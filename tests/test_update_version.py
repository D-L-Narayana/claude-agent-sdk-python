"""Tests for scripts/update_version.py: version validation and file writing."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "update_version.py"

# scripts/ is not a package, so load update_version.py by path
_spec = importlib.util.spec_from_file_location("update_version", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
update_version = importlib.util.module_from_spec(_spec)
sys.modules["update_version"] = update_version
_spec.loader.exec_module(update_version)

ORIGINAL_VERSION = "0.2.160"

PYPROJECT = (
    "[build-system]\n"
    'requires = ["hatchling"]\n'
    'build-backend = "hatchling.build"\n'
    "\n"
    "[project]\n"
    'name = "claude-agent-sdk"\n'
    f'version = "{ORIGINAL_VERSION}"\n'
    'description = "Python SDK for Claude Code"\n'
    'requires-python = ">=3.10"\n'
)

VERSION_PY = (
    '"""Version information for claude-agent-sdk."""\n'
    "\n"
    f'__version__ = "{ORIGINAL_VERSION}"\n'
)


class ProjectFiles:
    """A throwaway copy of the two files the script rewrites.

    Laid out exactly like the repository (pyproject.toml at the root,
    src/claude_agent_sdk/_version.py below it), so the same fixture serves the
    in-process tests, which pass the paths explicitly, and the command-line
    tests, which rely on the script's default relative paths from ``root``.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pyproject = root / "pyproject.toml"
        self.version_py = root / "src" / "claude_agent_sdk" / "_version.py"
        self.version_py.parent.mkdir(parents=True)
        self.pyproject.write_text(PYPROJECT)
        self.version_py.write_text(VERSION_PY)

    def update(self, version: str) -> None:
        update_version.update_version(
            version, pyproject_path=self.pyproject, version_path=self.version_py
        )

    def assert_untouched(self) -> None:
        assert self.pyproject.read_text() == PYPROJECT
        assert self.version_py.read_text() == VERSION_PY


@pytest.fixture
def project(tmp_path: Path) -> ProjectFiles:
    return ProjectFiles(tmp_path)


def import_version(path: Path) -> str:
    """Import the written file as Python and return its __version__.

    Fails on a file that is not valid Python, which is the whole point: the
    value goes into a real source file that `import claude_agent_sdk` runs.
    """
    spec = importlib.util.spec_from_file_location(f"_written_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version: str = module.__version__
    return version


class TestVersionPattern:
    """The grammar: MAJOR.MINOR.PATCH plus PEP 440 pre/post/dev segments."""

    @pytest.mark.parametrize(
        "version",
        [
            "0.2.161",
            "1.0.0",
            "10.20.30",
            "1.0.0a1",
            "1.0.0b2",
            "1.0.0rc1",
            "1.2.3.post1",
            "1.2.3.dev4",
            "1.2.3rc1.post2.dev3",
        ],
    )
    def test_admits_the_shapes_the_sdk_publishes(self, version: str) -> None:
        assert update_version.VERSION_PATTERN.fullmatch(version)

    @pytest.mark.parametrize(
        "version",
        [
            "0.2",
            "1",
            "1.2.3.4",
            "1.2.3-beta",
            "1.2.3-rc.1",
            "1.2.3+build.4",
            "1.2.3rc",
            "1.2.3.post",
            "1.2.3dev1",
            "1.2.3a1b2",
            "v1.2.3",
            "latest",
        ],
    )
    def test_rejects_other_shapes(self, version: str) -> None:
        assert not update_version.VERSION_PATTERN.fullmatch(version)

    @pytest.mark.parametrize(
        "char",
        [" ", ";", "$", "`", '"', "'", "(", ")", "&", "|", "\n", "\r", "\\"],
    )
    def test_admits_no_shell_or_quote_metacharacters(self, char: str) -> None:
        """The value lands inside a TOML string and a Python string literal."""
        assert not update_version.VERSION_PATTERN.fullmatch(char)
        assert not update_version.VERSION_PATTERN.fullmatch(f"1.2.3{char}")
        assert not update_version.VERSION_PATTERN.fullmatch(f"1.2.3{char}id")

    def test_only_ascii_digits(self) -> None:
        """A version is never spelled with other Unicode digits."""
        assert not update_version.VERSION_PATTERN.fullmatch("١.٢.٣")

    def test_is_unanchored(self) -> None:
        """Unanchored + fullmatch(): a future swap to match() then fails loudly on
        a prefix instead of silently accepting a trailing newline."""
        assert "^" not in update_version.VERSION_PATTERN.pattern
        assert "$" not in update_version.VERSION_PATTERN.pattern
        assert not update_version.VERSION_PATTERN.fullmatch("1.0.0\n")


class TestValidateVersion:
    @pytest.mark.parametrize(
        ("argument", "expected"),
        [
            ("0.2.161", "0.2.161"),
            ("0.2.161\n", "0.2.161"),
            ("0.2.161\r\n", "0.2.161"),
            ("  1.0.0rc1  ", "1.0.0rc1"),
            ("\t1.2.3.post1\n", "1.2.3.post1"),
        ],
    )
    def test_returns_the_stripped_value(self, argument: str, expected: str) -> None:
        assert update_version.validate_version(argument) == expected

    def test_leading_v_is_named_not_normalized(self) -> None:
        """Git tags carry the 'v'; the version files never do. Say so rather
        than silently writing a different string than the caller asked for."""
        with pytest.raises(ValueError, match="Invalid SDK version") as excinfo:
            update_version.validate_version("v0.2.161")

        message = str(excinfo.value)
        assert "'v0.2.161'" in message
        assert "Did you mean '0.2.161'?" in message

    def test_error_names_the_offending_value(self) -> None:
        with pytest.raises(ValueError, match="Invalid SDK version") as excinfo:
            update_version.validate_version("0.2.161; id")
        assert "0.2.161; id" in str(excinfo.value)

    def test_error_says_what_was_expected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            update_version.validate_version("latest")
        assert update_version.VERSION_PATTERN.pattern in str(excinfo.value)


class TestAcceptedVersions:
    """Valid versions round-trip through both files, which otherwise stay as
    they were."""

    @pytest.mark.parametrize(
        "version",
        [
            "0.2.161",
            "1.0.0a1",
            "1.0.0b2",
            "1.0.0rc1",
            "1.2.3.post1",
            "1.2.3.dev4",
            "2.0.0rc1.post1.dev2",
        ],
    )
    def test_round_trip(self, project: ProjectFiles, version: str) -> None:
        project.update(version)

        assert import_version(project.version_py) == version
        assert project.pyproject.read_text() == PYPROJECT.replace(
            ORIGINAL_VERSION, version
        )
        assert project.version_py.read_text() == VERSION_PY.replace(
            ORIGINAL_VERSION, version
        )

    @pytest.mark.parametrize(
        ("argument", "written"),
        [
            ("0.2.161\n", "0.2.161"),
            ("0.2.161\r\n", "0.2.161"),
            ("  1.0.0rc1  ", "1.0.0rc1"),
        ],
    )
    def test_surrounding_whitespace_is_stripped_before_writing(
        self, project: ProjectFiles, argument: str, written: str
    ) -> None:
        """The stripped value is written, so a trailing newline from `$(cat
        VERSION)` never lands inside the string literal."""
        project.update(argument)

        assert import_version(project.version_py) == written
        assert f'version = "{written}"\n' in project.pyproject.read_text()

    def test_prints_one_line_per_file(
        self, project: ProjectFiles, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project.update("0.2.161")

        out = capsys.readouterr().out
        assert "Updated pyproject.toml to version 0.2.161" in out
        assert "Updated _version.py to version 0.2.161" in out

    def test_only_the_first_version_assignment_in_pyproject_changes(
        self, project: ProjectFiles
    ) -> None:
        """[project].version comes first; a later table's `version` key and a
        comment that happens to spell the assignment are not the target."""
        before = (
            '# version = "0.0.0" is rewritten by scripts/update_version.py\n'
            + PYPROJECT
            + '\n[tool.x]\nversion = "9"\n'
        )
        project.pyproject.write_text(before)

        project.update("0.2.161")

        assert project.pyproject.read_text() == before.replace(
            f'\nversion = "{ORIGINAL_VERSION}"\n', '\nversion = "0.2.161"\n'
        )

    def test_crlf_line_endings_are_preserved(self, project: ProjectFiles) -> None:
        """A CRLF checkout must come back with CRLF: the script changes one
        string literal per file and nothing else."""
        project.pyproject.write_bytes(PYPROJECT.replace("\n", "\r\n").encode())
        project.version_py.write_bytes(VERSION_PY.replace("\n", "\r\n").encode())

        project.update("0.2.161")

        assert (
            project.pyproject.read_bytes()
            == (
                PYPROJECT.replace(ORIGINAL_VERSION, "0.2.161").replace("\n", "\r\n")
            ).encode()
        )
        assert (
            project.version_py.read_bytes()
            == (
                VERSION_PY.replace(ORIGINAL_VERSION, "0.2.161").replace("\n", "\r\n")
            ).encode()
        )

    def test_default_paths_are_the_repository_layout(
        self, project: ProjectFiles, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(project.root)

        update_version.update_version("0.2.161")

        assert import_version(project.version_py) == "0.2.161"
        assert 'version = "0.2.161"\n' in project.pyproject.read_text()


class TestRejectedVersions:
    """Invalid input raises before either file is touched."""

    @pytest.mark.parametrize(
        "version",
        [
            # A git-tag spelling, not a version.
            "v0.2.161",
            "V0.2.161",
            # Too few or too many components.
            "0.2",
            "1",
            "1.2.3.4",
            # Not versions at all.
            "latest",
            "stable",
            "next",
            # Shell-injection leftovers.
            "0.2.161; id",
            "$(id)",
            "`id`",
            "0.2.161 0.2.162",
            "../../etc/passwd",
            ".1.2.3",
            # Semver / local-version spellings PEP 440 does not use here.
            "1.2.3-beta",
            "1.2.3-rc.1",
            "1.2.3+build.4",
            "1.2.3dev1",
            # Empty and whitespace-only.
            "",
            " ",
            "\n",
            # Newlines *inside* the value: an extra source line in _version.py.
            # (A merely trailing newline is stripped -- see TestAcceptedVersions.)
            "0.2.161\nimport os",
            # Quote breakout: closes the string literal in a real source file.
            '0.2.161"',
            '0.2.161" + __import__("os").system("id") + "',
            # Backslashes: re.sub() replacement-escape processing, and invalid
            # Python escapes in the emitted literal.
            "0.2.161\\",
            "0.2.161\\n",
            "\\g<0>",
            # Flag-shaped.
            "-1.2.3",
            "--help",
            "-s",
        ],
    )
    def test_rejected_and_both_files_untouched(
        self, project: ProjectFiles, version: str
    ) -> None:
        with pytest.raises(ValueError, match="Invalid SDK version"):
            project.update(version)
        project.assert_untouched()

    def test_leading_v_is_named_not_normalized(self, project: ProjectFiles) -> None:
        with pytest.raises(ValueError) as excinfo:
            project.update("v0.2.161")

        assert "Did you mean '0.2.161'?" in str(excinfo.value)
        project.assert_untouched()

    def test_error_names_the_offending_value(self, project: ProjectFiles) -> None:
        with pytest.raises(ValueError) as excinfo:
            project.update("0.2.161; id")
        assert "0.2.161; id" in str(excinfo.value)


class TestMissingAssignment:
    """A file with nothing to replace fails the run, and neither file is
    written -- not even the one that did match."""

    def test_pyproject_without_a_version_line(self, project: ProjectFiles) -> None:
        no_version = PYPROJECT.replace(f'version = "{ORIGINAL_VERSION}"\n', "")
        project.pyproject.write_text(no_version)

        with pytest.raises(ValueError, match="No version assignment"):
            project.update("0.2.161")

        assert project.pyproject.read_text() == no_version
        assert project.version_py.read_text() == VERSION_PY

    def test_version_py_without_an_assignment(self, project: ProjectFiles) -> None:
        project.version_py.write_text("# no assignment here\n")

        with pytest.raises(ValueError, match="No __version__ assignment"):
            project.update("0.2.161")

        # pyproject.toml matched, but must not have been written: the two
        # files are validated together before either is touched.
        assert project.pyproject.read_text() == PYPROJECT
        assert project.version_py.read_text() == "# no assignment here\n"

    def test_single_quoted_assignment_is_not_matched(
        self, project: ProjectFiles
    ) -> None:
        """auto-release.yml reads the version back with a double-quoted
        pattern, so a reformatted file must fail here, not there."""
        project.version_py.write_text("__version__ = '0.2.160'\n")

        with pytest.raises(ValueError, match="No __version__ assignment"):
            project.update("0.2.161")

        assert project.pyproject.read_text() == PYPROJECT

    def test_error_names_the_file(self, project: ProjectFiles) -> None:
        project.version_py.write_text("")

        with pytest.raises(ValueError) as excinfo:
            project.update("0.2.161")

        assert str(project.version_py) in str(excinfo.value)


class TestReplacementIsLiteral:
    """The version reaches both files verbatim, with no escape interpretation.

    Validation already excludes every character these tests use, so they stub
    it out to exercise the write path directly. Without that, a plain-string
    re.sub() replacement -- which expands \\1 and \\g<0>, turns \\n into a
    newline, and raises on a bare trailing backslash -- would look identical to
    the callable replacement for all reachable input, and the guard would be
    unfalsifiable. This is the test that fails if someone swaps the callable
    back for an f-string.
    """

    @pytest.fixture
    def no_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(update_version, "validate_version", lambda version: version)

    @pytest.mark.usefixtures("no_validation")
    @pytest.mark.parametrize(
        "version",
        [
            "2.0.0\\1",
            "2.0.0\\g<0>",
            "2.0.0\\",
            "\\g<0>\\1",
            '2.0.0"',
            "2.0.0\\n",
            "2.0.0\nid",
        ],
    )
    def test_written_version_py_imports_back_to_the_exact_string(
        self, project: ProjectFiles, version: str
    ) -> None:
        project.update(version)
        assert import_version(project.version_py) == version

    @pytest.mark.usefixtures("no_validation")
    @pytest.mark.parametrize(
        ("version", "toml_line"),
        [
            # A backslash is doubled and a quote escaped, as a TOML basic
            # string (and a JSON string) requires.
            ("2.0.0\\g<0>", 'version = "2.0.0\\\\g<0>"'),
            ('2.0.0"', 'version = "2.0.0\\""'),
        ],
    )
    def test_pyproject_literal_is_escaped_like_a_toml_basic_string(
        self, project: ProjectFiles, version: str, toml_line: str
    ) -> None:
        project.update(version)
        assert f"\n{toml_line}\n" in project.pyproject.read_text()

    @pytest.mark.usefixtures("no_validation")
    def test_backreference_is_not_expanded_into_the_files(
        self, project: ProjectFiles
    ) -> None:
        """A string replacement would splice the matched assignment into itself."""
        project.update("2.0.0\\g<0>")

        assert project.version_py.read_text().count("__version__") == 1
        assert project.pyproject.read_text().count("\nversion = ") == 1

    def test_replacements_are_callables(
        self, monkeypatch: pytest.MonkeyPatch, project: ProjectFiles
    ) -> None:
        """Guard the mechanism itself, not only its observable output."""
        pyproject_spy = _SubnSpy(update_version.PYPROJECT_ASSIGNMENT)
        version_spy = _SubnSpy(update_version.VERSION_ASSIGNMENT)
        monkeypatch.setattr(update_version, "PYPROJECT_ASSIGNMENT", pyproject_spy)
        monkeypatch.setattr(update_version, "VERSION_ASSIGNMENT", version_spy)

        project.update("1.2.3")

        for spy in (pyproject_spy, version_spy):
            (repl,) = spy.replacements
            assert callable(repl), (
                f"re.sub replacement must be a callable to avoid escape "
                f"processing, got {repl!r}"
            )


class _SubnSpy:
    """Records the replacement handed to Pattern.subn(), then delegates."""

    def __init__(self, pattern: re.Pattern[str]) -> None:
        self._pattern = pattern
        self.replacements: list[object] = []

    def subn(self, repl: object, string: str, count: int = 0) -> tuple[str, int]:
        self.replacements.append(repl)
        return self._pattern.subn(repl, string, count=count)  # type: ignore[arg-type]


class TestCommandLine:
    """The script as the release workflows run it: `python scripts/update_version.py
    <version>` from the repository root.

    Every case runs in a throwaway cwd holding the repository's layout, so a
    regression in the guard corrupts the fixture rather than the repository.
    """

    def _run(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_valid_version_updates_both_files_and_exits_zero(
        self, project: ProjectFiles
    ) -> None:
        result = self._run(project.root, "0.2.161")

        assert result.returncode == 0, result.stderr
        assert "Updated pyproject.toml to version 0.2.161" in result.stdout
        assert "Updated _version.py to version 0.2.161" in result.stdout
        assert import_version(project.version_py) == "0.2.161"
        assert project.pyproject.read_text() == PYPROJECT.replace(
            ORIGINAL_VERSION, "0.2.161"
        )

    def test_whitespace_around_the_argument_is_stripped(
        self, project: ProjectFiles
    ) -> None:
        result = self._run(project.root, "1.0.0rc1\n")

        assert result.returncode == 0, result.stderr
        assert import_version(project.version_py) == "1.0.0rc1"

    @pytest.mark.parametrize(
        "version",
        ["v0.2.161", "0.2", "latest", "0.2.161; id", '0.2.161"; import os'],
    )
    def test_invalid_version_exits_one_without_writing(
        self, project: ProjectFiles, version: str
    ) -> None:
        result = self._run(project.root, version)

        assert result.returncode == 1
        assert "Invalid SDK version" in result.stderr
        assert "Traceback" not in result.stderr
        assert result.stdout == ""
        project.assert_untouched()

    def test_leading_v_suggests_the_fix(self, project: ProjectFiles) -> None:
        result = self._run(project.root, "v0.2.161")

        assert result.returncode == 1
        assert "Did you mean '0.2.161'?" in result.stderr
        project.assert_untouched()

    def test_error_line_names_the_offending_value(self, project: ProjectFiles) -> None:
        """Named in the one-line message, not merely in a traceback frame.

        Anchored to the start of a line: an uncaught raise renders the same
        text under `ValueError: `, which *contains* `Error: ` as a substring,
        so a plain `in` check would pass on the very traceback this guards
        against.
        """
        stderr = self._run(project.root, "0.2.161; id").stderr

        (error_line,) = [
            line for line in stderr.splitlines() if line.startswith("Error: ")
        ]
        assert "0.2.161; id" in error_line

    @pytest.mark.parametrize("args", [(), ("0.2.161", "0.2.162")])
    def test_wrong_argument_count_prints_usage_to_stderr(
        self, project: ProjectFiles, args: tuple[str, ...]
    ) -> None:
        result = self._run(project.root, *args)

        assert result.returncode == 1
        assert "Usage:" in result.stderr
        project.assert_untouched()

    def test_missing_assignment_exits_one_without_writing(
        self, project: ProjectFiles
    ) -> None:
        project.version_py.write_text("# no assignment here\n")

        result = self._run(project.root, "0.2.161")

        assert result.returncode == 1
        assert "No __version__ assignment" in result.stderr
        assert "Traceback" not in result.stderr
        assert project.pyproject.read_text() == PYPROJECT

    def test_missing_file_exits_one_with_a_message(self, project: ProjectFiles) -> None:
        project.version_py.unlink()

        result = self._run(project.root, "0.2.161")

        assert result.returncode == 1
        assert "_version.py" in result.stderr
        assert "Traceback" not in result.stderr
        assert project.pyproject.read_text() == PYPROJECT
