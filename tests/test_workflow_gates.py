"""Semantic checks on the job gates in .github/workflows/test.yml.

The e2e, Docker e2e and examples jobs exchange the workflow's OIDC token for a
Claude API token through ./.github/actions/setup-claude-auth, which needs the
three ANTHROPIC_* repository variables. Each of those jobs must run only on a
trusted event (a push, or a pull request whose head is this repository) *and*
when that configuration is present: a fork without the variables must skip
the jobs, not fail in the auth step -- and a skipped job means no real-API
validation happened, which is not the same as the offline suite passing.

The workflow is read with a real YAML parser and each job's ``if`` is
evaluated with GitHub's expression semantics (for the operator subset the
file uses) under the event/variable combinations that matter. The YAML is
never inspected as text.
"""

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOW_PATH = WORKFLOWS_DIR / "test.yml"
LINT_WORKFLOW_PATH = WORKFLOWS_DIR / "lint.yml"

# A manual run of Test or Lint is a fallback that verifies the offline jobs
# only (see TestManualDispatch). These workflows must not grow that button:
# they react to events, or are callable only, and run on demand they would
# publish, spend API credit or post to Slack. publish.yml and
# pypi-quota-check.yml have offered workflow_dispatch all along.
MANUAL_EVENT = "workflow_dispatch"
NO_MANUAL_DISPATCH = (
    "auto-release.yml",
    "build-and-publish.yml",
    "build-wheel-check.yml",
    "claude.yml",
    "claude-code-review.yml",
    "claude-issue-triage.yml",
    "slack-issue-notification.yml",
)

# The jobs that call the Claude API, and so need the auth action and its
# configuration; and the jobs that run offline and must never be gated.
GATED_JOBS = ("test-e2e", "test-e2e-docker", "test-examples")
OFFLINE_JOBS = ("test", "test-mcp-v1-floor", "test-min-python")
AUTH_ACTION = "./.github/actions/setup-claude-auth"
REQUIRED_VARIABLES = (
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
)
# The least the auth action needs: read the checkout, mint the OIDC token.
GATED_PERMISSIONS = {"contents": "read", "id-token": "write"}

UPSTREAM = "anthropics/claude-agent-sdk-python"
FORK = "example-fork/claude-agent-sdk-python"
CONFIGURED = dict.fromkeys(REQUIRED_VARIABLES, "configured")

# Used when PyYAML is not installed. The path travels in the environment:
# under `bun -e`, process.argv does not carry script arguments.
_BUN_SCRIPT = (
    'const fs = require("fs");'
    'const text = fs.readFileSync(process.env.WORKFLOW_PATH, "utf8");'
    "process.stdout.write(JSON.stringify(Bun.YAML.parse(text)));"
)


def load_workflow(path: Path) -> dict[str, Any]:
    """Parse ``path`` with a real YAML parser.

    PyYAML (in the dev extra) when importable, else the Bun runtime's
    ``Bun.YAML`` (YAML 1.2). Having neither is a failure, not a skip: a gate
    regression that goes unobserved because this test skipped is the very
    thing the test exists to catch. The two parsers differ on YAML 1.1
    booleans -- PyYAML reads the top-level ``on`` key as True, Bun as "on" --
    so nothing here depends on that key.
    """
    assert path.is_file(), f"{path} does not exist"
    if importlib.util.find_spec("yaml") is not None:
        import yaml

        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        bun = shutil.which("bun")
        if bun is None:
            pytest.fail(
                "No YAML parser available: install the dev extra "
                "(pip install -e '.[dev]', which brings PyYAML) or the Bun runtime"
            )
        result = subprocess.run(
            [bun, "-e", _BUN_SCRIPT],
            env={**os.environ, "WORKFLOW_PATH": str(path)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            pytest.fail(f"Bun could not parse {path}: {result.stderr.strip()}")
        document = json.loads(result.stdout)
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


# --- GitHub Actions expressions: the subset the workflow uses -----------------
#
# Operators: ! == != && || and parentheses; operands: single-quoted strings
# (with '' as an escaped quote), numbers, true/false/null and dotted context
# paths. Semantics as documented by GitHub: string comparison ignores case;
# operands of different types are coerced to numbers before comparing; && and
# || return an operand rather than a boolean; false, 0, '' and null are the
# falsy values. An unset variable or an absent payload field reads as ''.

_TOKEN = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<lparen>\()"
    r"|(?P<rparen>\))"
    r"|(?P<and>&&)"
    r"|(?P<or>\|\|)"
    r"|(?P<eq>==)"
    r"|(?P<ne>!=)"
    r"|(?P<not>!)"
    r"|'(?P<string>(?:[^']|'')*)'"
    r"|(?P<number>-?\d+(?:\.\d+)?)"
    r"|(?P<path>[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*)"
)

_LITERALS: dict[str, Any] = {"true": True, "false": False, "null": None}


def _tokenize(expression: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(expression):
        match = _TOKEN.match(expression, position)
        if match is None or match.lastgroup is None:
            raise ValueError(
                f"unsupported syntax at {expression[position:]!r} in {expression!r}"
            )
        if match.lastgroup != "space":
            tokens.append((match.lastgroup, match.group(match.lastgroup)))
        position = match.end()
    return tokens


def _lookup(context: dict[str, Any], path: str) -> Any:
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return ""
        value = value[part]
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value != ""
    return True


def _as_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return 0.0
        try:
            return float(text)
        except ValueError:
            return math.nan
    return math.nan


def _equals(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.casefold() == right.casefold()
    if type(left) is type(right):
        return bool(left == right)
    # Mixed types: GitHub coerces both sides to numbers (NaN never equals).
    return _as_number(left) == _as_number(right)


class _Evaluator:
    """Recursive-descent interpreter: or > and > equality > unary > primary."""

    def __init__(self, expression: str, context: dict[str, Any]) -> None:
        self._tokens = _tokenize(expression)
        self._position = 0
        self._context = context

    def evaluate(self) -> Any:
        value = self._or()
        if self._position != len(self._tokens):
            raise ValueError(f"unexpected {self._tokens[self._position][1]!r}")
        return value

    def _peek(self) -> str | None:
        if self._position < len(self._tokens):
            return self._tokens[self._position][0]
        return None

    def _take(self, kind: str) -> str:
        if self._peek() != kind:
            found = (
                repr(self._tokens[self._position][1])
                if self._position < len(self._tokens)
                else "the end of the expression"
            )
            raise ValueError(f"expected {kind}, found {found}")
        text = self._tokens[self._position][1]
        self._position += 1
        return text

    def _or(self) -> Any:
        value = self._and()
        while self._peek() == "or":
            self._position += 1
            right = self._and()
            value = value if _truthy(value) else right
        return value

    def _and(self) -> Any:
        value = self._equality()
        while self._peek() == "and":
            self._position += 1
            right = self._equality()
            value = right if _truthy(value) else value
        return value

    def _equality(self) -> Any:
        value = self._unary()
        while self._peek() in ("eq", "ne"):
            negate = self._peek() == "ne"
            self._position += 1
            right = self._unary()
            equal = _equals(value, right)
            value = not equal if negate else equal
        return value

    def _unary(self) -> Any:
        if self._peek() == "not":
            self._position += 1
            return not _truthy(self._unary())
        return self._primary()

    def _primary(self) -> Any:
        kind = self._peek()
        if kind == "lparen":
            self._position += 1
            value = self._or()
            self._take("rparen")
            return value
        if kind == "string":
            return self._take("string").replace("''", "'")
        if kind == "number":
            text = self._take("number")
            return float(text) if "." in text else int(text)
        if kind == "path":
            path = self._take("path")
            if path in _LITERALS:
                return _LITERALS[path]
            return _lookup(self._context, path)
        raise ValueError(
            "expected a value, found "
            + (
                repr(self._tokens[self._position][1])
                if self._position < len(self._tokens)
                else "the end of the expression"
            )
        )


def evaluate(expression: str, context: dict[str, Any]) -> Any:
    """The value of a GitHub Actions expression under ``context``."""
    text = expression.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2]
    return _Evaluator(text, context).evaluate()


def job_runs(job: dict[str, Any], context: dict[str, Any]) -> bool:
    """Whether GitHub would run ``job`` under ``context`` (no ``if``: always)."""
    condition = job.get("if", True)
    if isinstance(condition, bool):
        return condition
    return _truthy(evaluate(str(condition), context))


def _context(
    *,
    event: str,
    repository: str,
    head_repository: str | None = None,
    variables: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The github and vars contexts GitHub builds for one event.

    ``repository`` is the repository the workflow runs in -- the fork's own
    name on a fork. A pull request's payload names the head repository; a
    push has no pull_request payload at all.
    """
    payload: dict[str, Any] = {}
    if event == "pull_request":
        payload["pull_request"] = {"head": {"repo": {"full_name": head_repository}}}
    return {
        "github": {"event_name": event, "repository": repository, "event": payload},
        "vars": dict(variables or {}),
    }


def _uses(job: dict[str, Any], action: str) -> bool:
    return any(step.get("uses") == action for step in job.get("steps", []))


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _permission_scopes(node: dict[str, Any]) -> set[str]:
    permissions = node.get("permissions", {})
    if isinstance(permissions, dict):
        return set(permissions)
    # The write-all shorthand grants every scope, id-token included.
    return {"id-token"} if permissions == "write-all" else set()


def triggers(document: dict[str, Any]) -> dict[str, Any]:
    """The workflow's ``on`` block, as a mapping of event name to settings.

    A YAML 1.1 parser (PyYAML) reads the bare key ``on`` as the boolean True;
    a YAML 1.2 parser (Bun.YAML) keeps the string "on". Accept either, so the
    tests do not depend on which parser loaded the file.
    """
    block = document["on"] if "on" in document else document.get(True)
    assert isinstance(block, dict), f"trigger block is {block!r}, not a mapping"
    return block


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return load_workflow(WORKFLOW_PATH)


@pytest.fixture(scope="module")
def jobs(workflow: dict[str, Any]) -> dict[str, Any]:
    assert isinstance(workflow.get("jobs"), dict), "workflow has no jobs mapping"
    return workflow["jobs"]


@pytest.fixture(scope="module")
def lint_workflow() -> dict[str, Any]:
    return load_workflow(LINT_WORKFLOW_PATH)


class TestExpressionSemantics:
    """The evaluator follows GitHub's rules for the operators the gates use.

    These pin the evaluator itself, so a gate test that passes does so
    because the workflow is right, not because the evaluator is lenient.
    """

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            # Strings compare without regard to case.
            ("'push' == 'PUSH'", True),
            ("'push' != 'pull_request'", True),
            # Unset variables and absent payload fields read as ''.
            ("vars.NOT_SET == ''", True),
            ("github.event.pull_request.head.repo.full_name == ''", True),
            ("vars.NOT_SET != ''", False),
            # Mixed types coerce to numbers: null and '' are both 0.
            ("null == ''", True),
            ("'1' == 1", True),
            # Negation and truthiness.
            ("!''", True),
            ("!'push'", False),
            ("!null", True),
            ("!0", True),
            # && and || hand back an operand, not a boolean.
            ("'' || 'fallback'", "fallback"),
            ("'first' || 'second'", "first"),
            ("'first' && 'second'", "second"),
            ("'' && 'second'", ""),
            # && binds tighter than ||; parentheses override.
            ("false || true && false", False),
            ("(false || true) && false", False),
            ("(true || false) && true", True),
            # Escaped quote inside a string; the optional ${{ }} wrapper.
            ("'it''s' == 'IT''S'", True),
            ("${{ 'a' == 'A' }}", True),
        ],
    )
    def test_value(self, expression: str, expected: Any) -> None:
        assert evaluate(expression, {"github": {"event": {}}, "vars": {}}) == expected

    def test_context_paths_resolve(self) -> None:
        context = _context(event="push", repository=UPSTREAM, variables=CONFIGURED)

        assert evaluate("github.event_name == 'push'", context) is True
        assert evaluate(
            "github.repository == 'ANTHROPICS/claude-agent-sdk-python'", context
        )
        assert evaluate("vars.ANTHROPIC_ORGANIZATION_ID != ''", context) is True
        assert evaluate("github.event.pull_request.head.repo.full_name", context) == ""

    @pytest.mark.parametrize(
        "expression",
        [
            # Functions are outside the supported subset.
            "contains(github.ref, 'main')",
            "startsWith(github.ref, 'refs/tags') && true",
            # GitHub strings are single-quoted only.
            'github.ref == "main"',
            # Dangling operator; unbalanced parenthesis.
            "github.event_name == 'push' ||",
            "(github.event_name == 'push'",
        ],
    )
    def test_unsupported_syntax_is_an_error_not_a_guess(self, expression: str) -> None:
        """An `if` the evaluator cannot read must fail these tests loudly,
        never evaluate to something that happens to look right."""
        with pytest.raises(ValueError):
            evaluate(expression, {})

    def test_a_job_without_a_condition_always_runs(self) -> None:
        assert job_runs(
            {"runs-on": "ubuntu-latest"}, _context(event="push", repository=FORK)
        )


class TestStructure:
    """What the parsed workflow says about which jobs are gated, and how."""

    def test_every_expected_job_exists(self, jobs: dict[str, Any]) -> None:
        assert set(GATED_JOBS + OFFLINE_JOBS) <= set(jobs)

    def test_the_gate_covers_exactly_the_jobs_that_authenticate(
        self, jobs: dict[str, Any]
    ) -> None:
        """The gate exists because of the auth action: every job that uses it
        is gated, and no gated job is gated for no reason."""
        authenticating = {name for name, job in jobs.items() if _uses(job, AUTH_ACTION)}
        assert authenticating == set(GATED_JOBS)

    @pytest.mark.parametrize("job_name", GATED_JOBS)
    def test_gated_job_has_a_condition(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        assert isinstance(jobs[job_name].get("if"), str)

    @pytest.mark.parametrize("job_name", GATED_JOBS)
    def test_gated_job_permissions(self, jobs: dict[str, Any], job_name: str) -> None:
        assert jobs[job_name].get("permissions") == GATED_PERMISSIONS

    def test_nothing_else_requests_an_id_token(
        self, workflow: dict[str, Any], jobs: dict[str, Any]
    ) -> None:
        assert "id-token" not in _permission_scopes(workflow)
        for name, job in jobs.items():
            if name not in GATED_JOBS:
                assert "id-token" not in _permission_scopes(job), name

    @pytest.mark.parametrize("job_name", OFFLINE_JOBS)
    def test_offline_job_is_not_gated(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        assert "if" not in jobs[job_name]

    @pytest.mark.parametrize(
        ("job_name", "expected"),
        [
            ("test-e2e", ["test"]),
            ("test-e2e-docker", ["test"]),
            ("test-examples", ["test-e2e"]),
        ],
    )
    def test_needs_chain(
        self, jobs: dict[str, Any], job_name: str, expected: list[str]
    ) -> None:
        assert _needs(jobs[job_name]) == expected


@pytest.mark.parametrize("job_name", GATED_JOBS)
class TestGate:
    """Where each API-calling job runs, and where it skips."""

    def test_push_on_this_repository_with_configuration_runs(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        context = _context(event="push", repository=UPSTREAM, variables=CONFIGURED)
        assert job_runs(jobs[job_name], context)

    def test_push_on_a_configured_fork_runs(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        """A fork that set up its own federation gets the jobs too: the gate
        is about the configuration being present, not about one repository
        name."""
        context = _context(event="push", repository=FORK, variables=CONFIGURED)
        assert job_runs(jobs[job_name], context)

    def test_same_repository_pull_request_with_configuration_runs(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        context = _context(
            event="pull_request",
            repository=UPSTREAM,
            head_repository=UPSTREAM,
            variables=CONFIGURED,
        )
        assert job_runs(jobs[job_name], context)

    def test_external_pull_request_is_skipped_even_with_configuration(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        """The trust boundary stays: the base repository's configuration is
        never spent on a pull request from another repository."""
        context = _context(
            event="pull_request",
            repository=UPSTREAM,
            head_repository=FORK,
            variables=CONFIGURED,
        )
        assert not job_runs(jobs[job_name], context)

    def test_push_on_an_unconfigured_fork_is_skipped(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        """A push to a fork is a trusted event, but without the variables the
        auth action cannot work: the job must skip, not fail."""
        context = _context(event="push", repository=FORK, variables={})
        assert not job_runs(jobs[job_name], context)

    def test_same_repository_pull_request_without_configuration_is_skipped(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        context = _context(
            event="pull_request",
            repository=UPSTREAM,
            head_repository=UPSTREAM,
            variables={},
        )
        assert not job_runs(jobs[job_name], context)

    @pytest.mark.parametrize("missing", REQUIRED_VARIABLES)
    def test_one_absent_variable_skips(
        self, jobs: dict[str, Any], job_name: str, missing: str
    ) -> None:
        variables = {k: v for k, v in CONFIGURED.items() if k != missing}
        context = _context(event="push", repository=UPSTREAM, variables=variables)
        assert not job_runs(jobs[job_name], context)

    @pytest.mark.parametrize("blank", REQUIRED_VARIABLES)
    def test_one_blank_variable_skips(
        self, jobs: dict[str, Any], job_name: str, blank: str
    ) -> None:
        """GitHub hands an unset variable to the expression as ''; a variable
        that exists but is blank must count as unset too."""
        context = _context(
            event="push", repository=UPSTREAM, variables={**CONFIGURED, blank: ""}
        )
        assert not job_runs(jobs[job_name], context)


class TestManualDispatch:
    """A manual run of Test or Lint verifies the offline suite, nothing more.

    ``workflow_dispatch`` on these two workflows is a fallback for when no run
    was started automatically: it exercises the offline jobs and the lint job
    on a chosen ref. It is not a way to run the real-API jobs -- their gate
    admits pushes and same-repository pull requests only, so under a manual
    event they are skipped even with the federation variables configured --
    and it says nothing about why an automatic run did not happen.
    """

    @pytest.fixture(params=["workflow", "lint_workflow"], ids=["test.yml", "lint.yml"])
    def manual_workflow(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        document: dict[str, Any] = request.getfixturevalue(request.param)
        return document

    def test_offers_a_manual_run_without_inputs(
        self, manual_workflow: dict[str, Any]
    ) -> None:
        block = triggers(manual_workflow)
        assert MANUAL_EVENT in block

        # No value, or settings without inputs: a verification run must not
        # ask for anything.
        inputs = (block[MANUAL_EVENT] or {}).get("inputs") or {}
        assert inputs == {}, "a manual verification run must need no inputs"

    def test_automatic_triggers_are_unchanged(
        self, manual_workflow: dict[str, Any]
    ) -> None:
        """The manual trigger is added next to the automatic ones, which stay
        exactly as they were: every pull request, and pushes to main."""
        block = triggers(manual_workflow)
        assert "pull_request" in block
        assert block["push"] == {"branches": ["main"]}

    @pytest.mark.parametrize("job_name", GATED_JOBS)
    def test_real_api_job_is_skipped_under_a_manual_event(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        """Pins existing behavior rather than changing it: the gate's event
        clause -- a push, or a same-repository pull request -- already
        excludes workflow_dispatch, whether or not the variables are set. A
        green manual run is therefore never real-API validation."""
        context = _context(
            event=MANUAL_EVENT, repository=UPSTREAM, variables=CONFIGURED
        )
        assert not job_runs(jobs[job_name], context)

    @pytest.mark.parametrize("job_name", OFFLINE_JOBS)
    def test_offline_job_runs_under_a_manual_event(
        self, jobs: dict[str, Any], job_name: str
    ) -> None:
        """Ungated, so a manual event runs them -- with no configuration and
        without requesting an identity token."""
        context = _context(event=MANUAL_EVENT, repository=UPSTREAM, variables={})
        assert job_runs(jobs[job_name], context)
        assert "id-token" not in _permission_scopes(jobs[job_name])

    def test_lint_job_runs_under_a_manual_event(
        self, lint_workflow: dict[str, Any]
    ) -> None:
        lint_jobs = lint_workflow["jobs"]
        assert set(lint_jobs) == {"lint"}

        context = _context(event=MANUAL_EVENT, repository=UPSTREAM, variables={})
        assert job_runs(lint_jobs["lint"], context)
        assert "id-token" not in _permission_scopes(lint_workflow)
        assert "id-token" not in _permission_scopes(lint_jobs["lint"])

    @pytest.mark.parametrize("name", NO_MANUAL_DISPATCH)
    def test_other_workflows_offer_no_manual_run(self, name: str) -> None:
        """Pins the trigger inventory of the workflows that must stay
        event-driven or callable-only (see NO_MANUAL_DISPATCH)."""
        assert MANUAL_EVENT not in triggers(load_workflow(WORKFLOWS_DIR / name))
