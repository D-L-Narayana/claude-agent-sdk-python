#!/usr/bin/env python3
"""Update the SDK version in pyproject.toml and src/claude_agent_sdk/_version.py.

build-and-publish.yml runs this with the workflow's ``inputs.version`` -- a
string typed into the GitHub UI, or computed by auto-release.yml -- so the
value is validated before it is written into a TOML string and a Python
string literal in files the release then commits and ships.
"""

import json
import re
import sys
from pathlib import Path

# MAJOR.MINOR.PATCH with the optional PEP 440 segments the SDK could publish:
# a pre-release (a1, b2, rc3), a post-release (.post1) and a dev release
# (.dev4). No leading "v" (that is the git tag's spelling), no semver "-beta"
# suffix, no "+local" part: pip would not accept them, and nothing admitted
# here is a character that could close the string literal the value lands in.
#
# Deliberately unanchored and matched with fullmatch(): with "^...$" a swap to
# match() would silently accept a trailing newline ("1.0.0\n"); unanchored,
# the same swap accepts an obvious prefix like "1.0.0; id" and fails loudly in
# tests. re.ASCII keeps \d at [0-9]: a version is not spelled with any other
# Unicode digits.
VERSION_PATTERN = re.compile(
    r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?", re.ASCII
)

# The two assignments this script rewrites, one per file. Each matches a whole
# line (MULTILINE ^) and never spans lines: [^"\r\n] cannot run on past a
# missing closing quote into the next line.
#
# pyproject.toml: `version = "..."` in [project], the first table in the file;
# count=1 below leaves a later table's `version` key alone.
PYPROJECT_ASSIGNMENT = re.compile(r'^version = "[^"\r\n]*"', re.MULTILINE)
# _version.py: the assignment auto-release.yml reads back with a double-quoted
# pattern to compute the next version, so a single-quoted file must fail here.
VERSION_ASSIGNMENT = re.compile(r'^__version__ = "[^"\r\n]*"', re.MULTILINE)

DEFAULT_PYPROJECT_PATH = Path("pyproject.toml")
DEFAULT_VERSION_PATH = Path("src/claude_agent_sdk/_version.py")

_EXPECTED = (
    "MAJOR.MINOR.PATCH, optionally followed by a PEP 440 pre-release "
    "(a1, b2, rc3), .postN or .devN segment"
)


def _rejection(version: str) -> str:
    """Why ``version`` is unusable.

    The caller prefixes "Invalid SDK version: ", so this reads as the rest of
    that sentence.
    """
    candidate = version.strip()

    # Git tags carry the "v"; the version files never do. Name the fix rather
    # than silently writing a different string than the caller asked for.
    if candidate[:1] in ("v", "V") and VERSION_PATTERN.fullmatch(candidate[1:]):
        return f"{candidate!r}. Did you mean {candidate[1:]!r}? (no leading 'v')"

    # Name the raw value, not the stripped one: if the whitespace is the
    # problem, the reader has to be able to see it.
    return f"{version!r}. Expected {_EXPECTED}, matching {VERSION_PATTERN.pattern}"


def validate_version(version: str) -> str:
    """Return the usable form of ``version``, or raise.

    Surrounding whitespace is stripped first -- a trailing newline from a
    ``$(cat VERSION)`` or a "\\r" from a CRLF checkout is unambiguous in
    intent -- and the stripped value is what the caller gets back and must
    write.

    Raises:
        ValueError: If the stripped value is not a fullmatch of VERSION_PATTERN.
    """
    candidate = version.strip()
    if VERSION_PATTERN.fullmatch(candidate):
        return candidate
    raise ValueError(f"Invalid SDK version: {_rejection(version)}")


def _read(path: Path) -> str:
    # Bytes, not read_text(): universal-newline decoding would turn a CRLF
    # checkout's line endings into "\n", and write_text() would re-encode them
    # with the platform's convention. This script changes one string literal
    # per file and nothing else.
    return path.read_bytes().decode("utf-8")


def _write(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8"))


def _replace_assignment(
    pattern: re.Pattern[str], content: str, literal: str, missing: str
) -> str:
    """``content`` with the first match of ``pattern`` replaced by ``literal``.

    Raises:
        ValueError: With ``missing`` as the message, if there is no match.
    """
    # A callable replacement, because re.sub() applies backslash-escape
    # processing to a *string* replacement -- \1 and \g<0> expand, \n becomes
    # a newline, and a bare trailing \ raises. A callable's return value is
    # used literally.
    new_content, count = pattern.subn(lambda _match: literal, content, count=1)
    if count != 1:
        raise ValueError(missing)
    return new_content


def update_version(
    new_version: str,
    *,
    pyproject_path: Path = DEFAULT_PYPROJECT_PATH,
    version_path: Path = DEFAULT_VERSION_PATH,
) -> None:
    """Write ``new_version`` into pyproject.toml and _version.py.

    Both files are read and checked before either is written, so a failure
    leaves the working tree exactly as it was -- never one file at the new
    version and the other at the old.

    Raises:
        ValueError: If ``new_version`` is not a valid SDK version, or if
            either file has no version assignment to replace.
        OSError: If either file cannot be read or written.
    """
    # Validate before touching the files: the value goes into a TOML string
    # and into a Python string literal that `import claude_agent_sdk` runs, so
    # an unvalidated value closes the literal and injects arbitrary code.
    new_version = validate_version(new_version)

    # json.dumps() rather than an f-string: it always emits a closed,
    # double-quoted, fully escaped literal -- valid both as a TOML basic string
    # and as a Python string -- so a widened VERSION_PATTERN could never make
    # either file unparseable. repr() is not an option: it emits single
    # quotes, which auto-release.yml's reader would not match.
    pyproject_literal = f"version = {json.dumps(new_version)}"
    version_literal = f"__version__ = {json.dumps(new_version)}"

    pyproject_content = _read(pyproject_path)
    version_content = _read(version_path)

    new_pyproject = _replace_assignment(
        PYPROJECT_ASSIGNMENT,
        pyproject_content,
        pyproject_literal,
        f'No version assignment (`version = "..."`) found in {pyproject_path}',
    )
    new_version_py = _replace_assignment(
        VERSION_ASSIGNMENT,
        version_content,
        version_literal,
        f'No __version__ assignment (`__version__ = "..."`) found in {version_path}',
    )

    _write(pyproject_path, new_pyproject)
    print(f"Updated pyproject.toml to version {new_version}")
    _write(version_path, new_version_py)
    print(f"Updated _version.py to version {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/update_version.py <version>", file=sys.stderr)
        sys.exit(1)

    # This runs as a release step, so report a bad version or an unreadable
    # file the way the other scripts here do -- one line on stderr, exit 1 --
    # instead of letting a traceback out.
    try:
        update_version(sys.argv[1])
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
