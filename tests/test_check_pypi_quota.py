"""Tests for scripts/check_pypi_quota.py: the pure evaluate() seam and main()."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "check_pypi_quota.py"

# scripts/ is not a package, so load check_pypi_quota.py by path
_spec = importlib.util.spec_from_file_location("check_pypi_quota", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
check_pypi_quota = importlib.util.module_from_spec(_spec)
sys.modules["check_pypi_quota"] = check_pypi_quota
_spec.loader.exec_module(check_pypi_quota)


def _file(name: str, size: int) -> dict[str, Any]:
    """One entry as the PyPI simple API (PEP 691 + PEP 700) lists it."""
    return {
        "filename": name,
        "size": size,
        "url": f"https://files.pythonhosted.org/packages/{name}",
        "hashes": {"sha256": "0" * 64},
    }


def _evaluate(
    files: list[dict[str, Any]],
    *,
    project_limit: int = 1000,
    file_limit: int = 500,
    warn_threshold: float = 0.80,
    **kwargs: Any,
) -> Any:
    """evaluate() with small round limits, so the expected numbers read easily."""
    return check_pypi_quota.evaluate(
        files,
        project_limit=project_limit,
        file_limit=file_limit,
        warn_threshold=warn_threshold,
        **kwargs,
    )


def _boom(*args: object, **kwargs: object) -> None:
    raise AssertionError("evaluate() must not reach the network")


class TestEvaluateTotals:
    def test_sums_every_file(self) -> None:
        report = _evaluate([_file("a.whl", 100), _file("b.whl", 250), _file("c", 50)])
        assert report.total == 400

    def test_identifies_the_largest_file(self) -> None:
        report = _evaluate([_file("a.whl", 100), _file("b.whl", 250), _file("c", 50)])
        assert report.largest_name == "b.whl"
        assert report.largest_size == 250

    def test_percentages_are_fractions_of_the_limits(self) -> None:
        report = _evaluate([_file("a.whl", 100), _file("b.whl", 250), _file("c", 50)])
        assert report.project_pct == pytest.approx(0.4)
        assert report.file_pct == pytest.approx(0.5)

    def test_a_file_without_a_size_counts_as_zero(self) -> None:
        """Older index responses may omit PEP 700's `size`."""
        report = _evaluate([{"filename": "x.whl"}])
        assert report.total == 0
        assert report.largest_size == 0
        assert report.largest_name == "x.whl"

    def test_empty_file_list(self) -> None:
        report = _evaluate([])

        assert report.total == 0
        assert report.largest_size == 0
        assert report.largest_name == "<none>"
        assert report.project_pct == 0.0
        assert report.file_pct == 0.0
        assert report.over_project is False
        assert report.over_file is False
        assert report.alert is False

    def test_report_is_immutable(self) -> None:
        report = _evaluate([_file("a.whl", 1)])
        with pytest.raises(AttributeError):
            report.alert = True


class TestEvaluateThresholds:
    def test_below_both_thresholds(self) -> None:
        report = _evaluate([_file("a.whl", 300), _file("b.whl", 200)])

        assert report.over_project is False
        assert report.over_file is False
        assert report.alert is False

    def test_at_the_project_threshold_alerts(self) -> None:
        """`>=`: hitting the line exactly is already a warning."""
        report = _evaluate([_file("a.whl", 300), _file("b.whl", 300), _file("c", 200)])

        assert report.project_pct == pytest.approx(0.8)
        assert report.over_project is True
        assert report.over_file is False
        assert report.alert is True

    def test_just_below_the_project_threshold_does_not_alert(self) -> None:
        report = _evaluate([_file("a.whl", 300), _file("b.whl", 300), _file("c", 199)])
        assert report.over_project is False
        assert report.alert is False

    def test_above_the_file_threshold_alerts(self) -> None:
        report = _evaluate([_file("big.whl", 450)])

        assert report.file_pct == pytest.approx(0.9)
        assert report.over_file is True
        assert report.over_project is False
        assert report.alert is True

    def test_both_over(self) -> None:
        report = _evaluate([_file("big.whl", 450), _file("small.whl", 400)])

        assert report.over_project is True
        assert report.over_file is True
        assert report.alert is True

    def test_threshold_is_configurable(self) -> None:
        files = [_file("a.whl", 300), _file("b.whl", 300)]

        assert _evaluate(files).alert is False
        assert _evaluate(files, warn_threshold=0.5).alert is True

    @pytest.mark.parametrize(
        "limits",
        [
            {"project_limit": 0},
            {"project_limit": -1},
            {"file_limit": 0},
            {"file_limit": -5},
        ],
    )
    def test_non_positive_limits_are_rejected(self, limits: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="positive"):
            _evaluate([_file("a.whl", 1)], **limits)


class TestEvaluateSummary:
    """The Slack text the workflow posts, built once here for GITHUB_OUTPUT."""

    def test_contains_both_percentages(self) -> None:
        report = _evaluate([_file("a.whl", 100), _file("b.whl", 250)])

        assert f"({report.project_pct:.1%})" in report.summary
        assert f"({report.file_pct:.1%})" in report.summary
        assert "(35.0%)" in report.summary
        assert "(50.0%)" in report.summary

    def test_contains_human_readable_sizes_and_limits(self) -> None:
        human = check_pypi_quota.human
        report = _evaluate(
            [_file("a.whl", 30 * 1024**2), _file("b.whl", 10 * 1024**2)],
            project_limit=50 * 1024**3,
            file_limit=100 * 1024**2,
        )

        assert f"{human(40 * 1024**2)} / {human(50 * 1024**3)}" in report.summary
        assert f"{human(30 * 1024**2)} / {human(100 * 1024**2)}" in report.summary

    def test_names_the_package(self) -> None:
        assert "`claude-agent-sdk`" in _evaluate([]).summary
        assert "`other-sdk`" in _evaluate([], package="other-sdk").summary

    def test_marks_only_the_exceeded_quota(self) -> None:
        project_only = _evaluate([_file("a", 300), _file("b", 300), _file("c", 200)])
        file_only = _evaluate([_file("big.whl", 450)])
        neither = _evaluate([_file("a.whl", 100)])

        project_line, file_line = (
            line for line in project_only.summary.splitlines() if line.startswith("•")
        )
        assert project_line.endswith(":rotating_light:")
        assert not file_line.endswith(":rotating_light:")

        project_line, file_line = (
            line for line in file_only.summary.splitlines() if line.startswith("•")
        )
        assert not project_line.endswith(":rotating_light:")
        assert file_line.endswith(":rotating_light:")

        assert ":rotating_light:" not in neither.summary

    def test_advises_the_remedy(self) -> None:
        summary = _evaluate([]).summary
        assert "yanking old releases" in summary
        assert "limit increase" in summary


class TestEvaluateIsPure:
    def test_touches_neither_the_network_nor_github_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(check_pypi_quota, "fetch_project_files", _boom)
        monkeypatch.setattr(check_pypi_quota.urllib.request, "urlopen", _boom)
        gh_out = tmp_path / "github_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))

        report = _evaluate([_file("big.whl", 450), _file("small.whl", 400)])

        assert report.alert is True
        assert not gh_out.exists()


class TestHuman:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            (0, "0.00 B"),
            (1023, "1023.00 B"),
            (1024, "1.00 KiB"),
            (100 * 1024**2, "100.00 MiB"),
            (50 * 1024**3, "50.00 GiB"),
            (3 * 1024**4, "3.00 TiB"),
            (2048 * 1024**4, "2048.00 TiB"),
        ],
    )
    def test_binary_units(self, size: int, expected: str) -> None:
        assert check_pypi_quota.human(size) == expected


class TestMain:
    """main() keeps its exact console and GITHUB_OUTPUT contract; the index
    fetch is replaced so no test here touches the network."""

    @pytest.fixture
    def run_main(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> Any:
        fetched: list[str] = []

        def run(
            files: list[dict[str, Any]], *argv: str, github_output: bool = True
        ) -> tuple[int, str, str | None, list[str]]:
            def fake_fetch(package: str) -> list[dict[str, Any]]:
                fetched.append(package)
                return files

            monkeypatch.setattr(check_pypi_quota, "fetch_project_files", fake_fetch)
            monkeypatch.setattr(check_pypi_quota.urllib.request, "urlopen", _boom)
            monkeypatch.setattr(sys, "argv", ["check_pypi_quota.py", *argv])
            gh_out = tmp_path / "github_output"
            if github_output:
                monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
            else:
                monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

            returncode = check_pypi_quota.main()

            out = capsys.readouterr().out
            written = gh_out.read_text() if gh_out.exists() else None
            return returncode, out, written, fetched

        return run

    LIMITS = ("--project-limit", "1000", "--file-limit", "500")

    def test_below_threshold_reports_all_clear(self, run_main: Any) -> None:
        returncode, out, gh_out, _ = run_main([_file("a.whl", 100)], *self.LIMITS)

        assert returncode == 0
        assert "All quotas below warning threshold." in out
        assert "::warning::" not in out
        assert gh_out is not None
        assert "alert=false\n" in gh_out
        assert "project_pct=0.100\n" in gh_out
        assert "file_pct=0.200\n" in gh_out
        assert "summary<<EOF\n" in gh_out
        assert gh_out.endswith("\nEOF\n")

    def test_prints_the_usage_table(self, run_main: Any) -> None:
        _, out, _, _ = run_main([_file("a.whl", 100)], *self.LIMITS)

        assert "Package:        claude-agent-sdk\n" in out
        assert "Files on PyPI:  1\n" in out
        assert "Project usage:  100.00 B / 1000.00 B (10.0%)\n" in out
        assert "Largest file:   100.00 B / 500.00 B (20.0%) — a.whl\n" in out

    def test_over_threshold_prints_a_workflow_warning(self, run_main: Any) -> None:
        files = [_file("big.whl", 450), _file("small.whl", 400)]

        returncode, out, gh_out, _ = run_main(files, *self.LIMITS)

        # Still exit 0: the workflow decides whether an alert fails the job.
        assert returncode == 0
        assert (
            "::warning::PyPI quota threshold exceeded: "
            "project size at 85.0% of limit; largest file at 90.0% of limit\n"
        ) in out
        assert "All quotas below warning threshold." not in out
        assert gh_out is not None
        assert "alert=true\n" in gh_out
        assert "project_pct=0.850\n" in gh_out
        assert "file_pct=0.900\n" in gh_out
        assert "*PyPI quota warning for `claude-agent-sdk`*" in gh_out
        assert gh_out.count(":rotating_light:") == 2

    def test_warning_names_only_the_exceeded_quota(self, run_main: Any) -> None:
        _, out, _, _ = run_main([_file("big.whl", 450)], *self.LIMITS)

        assert (
            "::warning::PyPI quota threshold exceeded: largest file at 90.0% of limit\n"
        ) in out
        assert "project size" not in out.split("::warning::")[1]

    def test_package_option_reaches_the_fetch_and_the_summary(
        self, run_main: Any
    ) -> None:
        _, out, gh_out, fetched = run_main(
            [_file("a.whl", 100)], "--package", "other-sdk", *self.LIMITS
        )

        assert fetched == ["other-sdk"]
        assert "Package:        other-sdk\n" in out
        assert gh_out is not None
        assert "*PyPI quota warning for `other-sdk`*" in gh_out

    def test_warn_threshold_option(self, run_main: Any) -> None:
        _, out, gh_out, _ = run_main(
            [_file("a.whl", 300)], *self.LIMITS, "--warn-threshold", "0.3"
        )

        assert "::warning::" in out
        assert gh_out is not None
        assert "alert=true\n" in gh_out

    def test_without_github_output_nothing_is_written(self, run_main: Any) -> None:
        returncode, out, gh_out, _ = run_main(
            [_file("a.whl", 100)], *self.LIMITS, github_output=False
        )

        assert returncode == 0
        assert "All quotas below warning threshold." in out
        assert gh_out is None

    def test_main_routes_through_evaluate(
        self, monkeypatch: pytest.MonkeyPatch, run_main: Any
    ) -> None:
        """The seam is only worth testing if main() actually uses it."""
        calls: list[dict[str, Any]] = []
        real_evaluate = check_pypi_quota.evaluate

        def spy(files: list[dict[str, Any]], **kwargs: Any) -> Any:
            calls.append({"files": files, **kwargs})
            return real_evaluate(files, **kwargs)

        monkeypatch.setattr(check_pypi_quota, "evaluate", spy)
        files = [_file("a.whl", 100)]

        run_main(files, *self.LIMITS, "--warn-threshold", "0.75")

        assert calls == [
            {
                "files": files,
                "project_limit": 1000,
                "file_limit": 500,
                "warn_threshold": 0.75,
                "package": "claude-agent-sdk",
            }
        ]
