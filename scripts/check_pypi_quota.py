#!/usr/bin/env python3
"""Check PyPI storage usage for this project against the quota limits
documented at https://docs.pypi.org/project-management/storage-limits/.

Intended to run as a daily GitHub Actions job. Sets GitHub Actions outputs
when usage crosses the configured warning threshold, so the workflow can
post a Slack alert before publishes start failing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

PYPI_PROJECT_LIMIT_BYTES = 50 * 1024**3  # 50 GiB (increased from PyPI default 10 GiB)
PYPI_FILE_LIMIT_BYTES = 100 * 1024**2  # 100 MiB

DEFAULT_PACKAGE = "claude-agent-sdk"


def fetch_project_files(package: str) -> list[dict[str, Any]]:
    """The project's files from PyPI's JSON simple index (PEP 691), each entry
    carrying its ``size`` (PEP 700). The only network access in this script."""
    req = urllib.request.Request(
        f"https://pypi.org/simple/{package}/",
        headers={
            "Accept": "application/vnd.pypi.simple.v1+json",
            "User-Agent": "claude-agent-sdk-quota-check",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return cast("list[dict[str, Any]]", data.get("files", []))


def human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


@dataclass(frozen=True)
class QuotaReport:
    """What evaluate() concluded about one project's files.

    ``project_pct`` and ``file_pct`` are fractions of the respective limits
    (0.85 for 85 %), the form main() prints and writes to GITHUB_OUTPUT.
    ``summary`` is the Slack message body the workflow posts on an alert.
    """

    total: int
    largest_size: int
    largest_name: str
    project_pct: float
    file_pct: float
    over_project: bool
    over_file: bool
    alert: bool
    summary: str


def evaluate(
    files: list[dict[str, Any]],
    *,
    project_limit: int,
    file_limit: int,
    warn_threshold: float,
    package: str = DEFAULT_PACKAGE,
) -> QuotaReport:
    """Measure ``files`` against the limits. Pure: no network, no file I/O.

    ``files`` is what fetch_project_files() returns; an entry without a
    ``size`` counts as zero bytes. A limit counts as exceeded once usage
    reaches ``warn_threshold`` of it (``>=``). ``package`` is only named in
    the summary text.

    Raises:
        ValueError: If a limit is not positive -- the percentages would be
            meaningless, or a division by zero.
    """
    if project_limit <= 0:
        raise ValueError(f"project_limit must be positive, got {project_limit}")
    if file_limit <= 0:
        raise ValueError(f"file_limit must be positive, got {file_limit}")

    total: int = sum(f.get("size", 0) for f in files)
    largest = max(files, key=lambda f: f.get("size", 0), default={})
    largest_size: int = largest.get("size", 0)
    largest_name: str = largest.get("filename", "<none>")

    project_pct = total / project_limit
    file_pct = largest_size / file_limit

    over_project = project_pct >= warn_threshold
    over_file = file_pct >= warn_threshold

    summary = (
        f"*PyPI quota warning for `{package}`*\n"
        f"• Project: {human(total)} / {human(project_limit)} "
        f"({project_pct:.1%})"
        f"{' :rotating_light:' if over_project else ''}\n"
        f"• Largest file: {human(largest_size)} / "
        f"{human(file_limit)} ({file_pct:.1%})"
        f"{' :rotating_light:' if over_file else ''}\n"
        f"Consider yanking old releases or requesting a limit increase "
        f"before the next publish."
    )

    return QuotaReport(
        total=total,
        largest_size=largest_size,
        largest_name=largest_name,
        project_pct=project_pct,
        file_pct=file_pct,
        over_project=over_project,
        over_file=over_file,
        alert=over_project or over_file,
        summary=summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--project-limit",
        type=int,
        default=PYPI_PROJECT_LIMIT_BYTES,
        help="Total project size limit in bytes (default: 50 GiB)",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=PYPI_FILE_LIMIT_BYTES,
        help="Per-file size limit in bytes (default: PyPI's 100 MiB)",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.80,
        help="Fraction of limit that triggers a warning (default: 0.80)",
    )
    args = parser.parse_args()

    files = fetch_project_files(args.package)
    report = evaluate(
        files,
        project_limit=args.project_limit,
        file_limit=args.file_limit,
        warn_threshold=args.warn_threshold,
        package=args.package,
    )

    print(f"Package:        {args.package}")
    print(f"Files on PyPI:  {len(files)}")
    print(
        f"Project usage:  {human(report.total)} / {human(args.project_limit)} "
        f"({report.project_pct:.1%})"
    )
    print(
        f"Largest file:   {human(report.largest_size)} / {human(args.file_limit)} "
        f"({report.file_pct:.1%}) — {report.largest_name}"
    )

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with Path(gh_out).open("a", encoding="utf-8") as f:
            f.write(f"alert={'true' if report.alert else 'false'}\n")
            f.write(f"project_pct={report.project_pct:.3f}\n")
            f.write(f"file_pct={report.file_pct:.3f}\n")
            f.write("summary<<EOF\n")
            f.write(report.summary)
            f.write("\nEOF\n")

    if report.alert:
        which = []
        if report.over_project:
            which.append(f"project size at {report.project_pct:.1%} of limit")
        if report.over_file:
            which.append(f"largest file at {report.file_pct:.1%} of limit")
        print(f"::warning::PyPI quota threshold exceeded: {'; '.join(which)}")
    else:
        print("All quotas below warning threshold.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
