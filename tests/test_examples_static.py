"""Static checks for the scripts under ``examples/``.

The examples talk to the Claude API, so this suite never runs or imports them.
Instead every ``examples/**/*.py`` file is

* parsed and compiled, so a syntax error cannot ship;
* scanned for standard-library APIs that only exist on Python 3.11+ (the
  package declares ``requires-python = ">=3.10"``);
* checked so that every name it imports from ``claude_agent_sdk`` is really
  exported (``__all__``), and every name imported from a ``claude_agent_sdk``
  submodule really exists there;
* checked so that a hook callback only reports ``hookEventName`` values for
  events it is registered under.

``examples/streaming_mode_ipython.py`` is a sequence of copy-paste IPython
snippets with top-level ``await``: it is not an importable module, so it is
excluded from the compile/import checks and only parsed with top-level
``await`` allowed for the Python-version scan. The ``session_stores`` adapters
depend on optional clients (``boto3``, ``redis``, ``asyncpg``) and are only
parsed and compiled here, never imported.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import claude_agent_sdk

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

#: Copy-paste IPython snippets: top-level ``await`` is valid in an IPython cell
#: but not in a module, so these files can be neither compiled nor imported.
IPYTHON_SNIPPETS = frozenset({"streaming_mode_ipython.py"})

#: ``module.attribute`` pairs added in Python 3.11; the package supports 3.10.
PY311_ONLY_ATTRIBUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("asyncio", "timeout"),
        ("asyncio", "timeout_at"),
        ("asyncio", "TaskGroup"),
        ("datetime", "UTC"),
    }
)
#: Builtins added in Python 3.11.
PY311_ONLY_BUILTINS = frozenset({"ExceptionGroup", "BaseExceptionGroup"})

# ``except*`` has its own node type on 3.11+; on 3.10 it is a syntax error.
_TRY_STAR = getattr(ast, "TryStar", None)


def _example_files() -> list[Path]:
    return sorted(
        path for path in EXAMPLES_DIR.rglob("*.py") if path.name not in IPYTHON_SNIPPETS
    )


EXAMPLE_FILES = _example_files()
IPYTHON_FILES = sorted(EXAMPLES_DIR / name for name in IPYTHON_SNIPPETS)


def _relative(path: Path) -> str:
    return path.relative_to(EXAMPLES_DIR).as_posix()


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _parse_allowing_top_level_await(path: Path) -> ast.Module:
    tree = compile(
        path.read_text(encoding="utf-8"),
        str(path),
        "exec",
        flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )
    assert isinstance(tree, ast.Module)
    return tree


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """Map local names bound by ``import x`` / ``import x as y`` to module names."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    top_level = alias.name.split(".")[0]
                    aliases[top_level] = top_level
    return aliases


def _python_311_only_uses(tree: ast.Module) -> list[str]:
    """Describe every use of a Python 3.11+-only API in ``tree``."""
    aliases = _module_aliases(tree)
    uses: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = aliases.get(node.value.id)
            if module is not None and (module, node.attr) in PY311_ONLY_ATTRIBUTES:
                uses.append(f"line {node.lineno}: {module}.{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if (node.module, alias.name) in PY311_ONLY_ATTRIBUTES:
                    uses.append(
                        f"line {node.lineno}: from {node.module} import {alias.name}"
                    )
        elif isinstance(node, ast.Name) and node.id in PY311_ONLY_BUILTINS:
            uses.append(f"line {node.lineno}: {node.id}")
        elif _TRY_STAR is not None and isinstance(node, _TRY_STAR):
            uses.append(f"line {node.lineno}: except*")
    return uses


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _matcher_callbacks(matchers: ast.expr) -> list[str]:
    """Callback names listed in ``[HookMatcher(hooks=[cb, ...]), ...]``."""
    names: list[str] = []
    if not isinstance(matchers, (ast.List, ast.Tuple)):
        return names
    for matcher in matchers.elts:
        if not (isinstance(matcher, ast.Call) and _call_name(matcher) == "HookMatcher"):
            continue
        for keyword in matcher.keywords:
            if keyword.arg == "hooks" and isinstance(
                keyword.value, (ast.List, ast.Tuple)
            ):
                names.extend(
                    elt.id for elt in keyword.value.elts if isinstance(elt, ast.Name)
                )
    return names


def _registered_hook_events(tree: ast.Module) -> dict[str, set[str]]:
    """Map each hook callback name to the events it is registered under.

    Recognizes the documented shape
    ``ClaudeAgentOptions(hooks={"<Event>": [HookMatcher(hooks=[cb]), ...]})``.
    """
    registered: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call) and _call_name(node) == "ClaudeAgentOptions"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg != "hooks" or not isinstance(keyword.value, ast.Dict):
                continue
            for key, matchers in zip(
                keyword.value.keys, keyword.value.values, strict=True
            ):
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                for callback in _matcher_callbacks(matchers):
                    registered.setdefault(callback, set()).add(key.value)
    return registered


def _reported_hook_events(func: ast.AST) -> list[tuple[int, str]]:
    """``hookEventName`` string literals in dict literals inside ``func``."""
    reported: list[tuple[int, str]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "hookEventName"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                reported.append((value.lineno, value.value))
    return reported


def test_discovers_the_example_scripts() -> None:
    names = {_relative(path) for path in EXAMPLE_FILES}
    assert {
        "quick_start.py",
        "streaming_mode.py",
        "hooks.py",
        "include_partial_messages.py",
        "session_stores/s3_session_store.py",
    } <= names
    assert names.isdisjoint(IPYTHON_SNIPPETS)
    for path in IPYTHON_FILES:
        # The exclusion is deliberate; update IPYTHON_SNIPPETS if the file moves.
        assert path.is_file(), f"{_relative(path)} is listed as an IPython snippet"


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=_relative)
def test_parses_and_compiles(path: Path) -> None:
    tree = _parse(path)
    compile(tree, str(path), "exec")


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=_relative)
def test_uses_no_python_311_only_apis(path: Path) -> None:
    uses = _python_311_only_uses(_parse(path))
    assert uses == [], (
        f"{_relative(path)} uses APIs that do not exist on Python 3.10 "
        f"(the package's minimum): {uses}. Use anyio.fail_after / "
        "anyio.move_on_after or asyncio.wait_for instead of asyncio.timeout."
    )


@pytest.mark.parametrize("path", IPYTHON_FILES, ids=_relative)
def test_ipython_snippets_use_no_python_311_only_apis(path: Path) -> None:
    uses = _python_311_only_uses(_parse_allowing_top_level_await(path))
    assert uses == [], (
        f"{_relative(path)} uses APIs that do not exist on Python 3.10 "
        f"(the package's minimum): {uses}"
    )


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=_relative)
def test_claude_agent_sdk_imports_resolve(path: Path) -> None:
    missing: list[str] = []
    for node in ast.walk(_parse(path)):
        if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
            continue
        if node.module != "claude_agent_sdk" and not node.module.startswith(
            "claude_agent_sdk."
        ):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            if node.module == "claude_agent_sdk":
                exported = alias.name in claude_agent_sdk.__all__
            else:
                exported = hasattr(importlib.import_module(node.module), alias.name)
            if not exported:
                missing.append(
                    f"line {node.lineno}: from {node.module} import {alias.name}"
                )
    assert missing == [], (
        f"{_relative(path)} imports names the SDK does not export: {missing}"
    )


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=_relative)
def test_hook_callbacks_report_the_event_they_are_registered_for(path: Path) -> None:
    tree = _parse(path)
    registered = _registered_hook_events(tree)
    mismatches: list[str] = []
    for node in tree.body:
        if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in registered
        ):
            continue
        for lineno, event in _reported_hook_events(node):
            if event not in registered[node.name]:
                mismatches.append(
                    f"line {lineno}: {node.name}() is registered for "
                    f"{sorted(registered[node.name])} but reports "
                    f"hookEventName={event!r}"
                )
    assert mismatches == [], (
        f"{_relative(path)} has hook callbacks whose hookSpecificOutput names a "
        f"different event than the one they handle: {mismatches}"
    )
