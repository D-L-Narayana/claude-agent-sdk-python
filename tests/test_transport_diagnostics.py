"""Transport diagnostics: cached ``claude -v`` probe, stderr tail, stderr tee.

Three behaviors of :class:`SubprocessCLITransport` are covered here:

* the ``claude -v`` version probe runs once per CLI binary per process (keyed
  on path, size and mtime) and its result is exposed as ``cli_version``;
* the CLI's stderr is always piped, read line by line, and re-emitted on this
  process's ``sys.stderr`` when no ``options.stderr`` callback is set (unless
  ``CLAUDE_AGENT_SDK_INHERIT_STDERR=1`` asks for the old fd inheritance);
* a bounded tail of recent stderr lines rides along on the ``ProcessError``
  raised for a non-zero exit and on the ``CLIConnectionError`` raised when
  writing to a CLI that already died, so a crash at startup is diagnosable.

Every async test runs under both asyncio and trio (``anyio_backend`` in
conftest.py). Tests that spawn the shebang-based fake CLI are POSIX-only, like
the other fake-CLI tests in this suite.
"""

import io
import logging
import os
import sys
import textwrap
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from subprocess import PIPE
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from claude_agent_sdk._errors import CLIConnectionError, ProcessError
from claude_agent_sdk._internal.transport.subprocess_cli import (
    _ACTIVE_CHILDREN,
    _INHERIT_STDERR_ENV,
    _STDERR_TAIL_MAX_CHARS,
    _STDERR_TAIL_MAX_LINES,
    SubprocessCLITransport,
    _probe_cli_version,
    clear_cli_version_cache,
)
from claude_agent_sdk.types import ClaudeAgentOptions

pytestmark = pytest.mark.anyio

LOGGER_NAME = "claude_agent_sdk._internal.transport.subprocess_cli"
SKIP_VERSION_CHECK_ENV = "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"
# Never exists: the probe cache must not key on it (os.stat fails).
MISSING_CLI_PATH = "/nonexistent/claude-agent-sdk-tests/claude"

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="spawns a shebang script"
)

# A stand-in `claude` CLI. It answers `-v`, then follows FAKE_CLI_SCENARIO:
#   serve (default)   emit an init frame, answer control requests, exit on EOF
#   stderr-then-exit  two stderr lines, an init frame, then exit 1
#   exit-now          one stderr line and exit 3 before writing any stdout
FAKE_CLI = textwrap.dedent(
    """
    #!/usr/bin/env python3
    import json, os, sys

    if "-v" in sys.argv or "--version" in sys.argv:
        print("__VERSION__ (Claude Code)")
        sys.exit(0)

    scenario = os.environ.get("FAKE_CLI_SCENARIO", "serve")
    if scenario == "exit-now":
        print("fatal: no credentials found", file=sys.stderr, flush=True)
        sys.exit(3)
    if scenario == "stderr-then-exit":
        print("warn: first diagnostic line", file=sys.stderr, flush=True)
        print("warn: second diagnostic line", file=sys.stderr, flush=True)

    print(json.dumps({"type": "system", "subtype": "init", "session_id": "s",
                      "model": "m", "cwd": ".", "tools": [], "mcp_servers": [],
                      "permissionMode": "default", "apiKeySource": "none"}), flush=True)
    if scenario == "stderr-then-exit":
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "control_request":
            print(json.dumps({"type": "control_response",
                              "response": {"subtype": "success",
                                           "request_id": msg["request_id"],
                                           "response": {}}}), flush=True)
    """
).lstrip()


@pytest.fixture(autouse=True)
def _isolated_transport_globals(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start from an empty probe cache and the default environment, and keep
    this file's mock processes out of the module-global _ACTIVE_CHILDREN."""
    monkeypatch.delenv(SKIP_VERSION_CHECK_ENV, raising=False)
    monkeypatch.delenv(_INHERIT_STDERR_ENV, raising=False)
    clear_cli_version_cache()
    before = set(_ACTIVE_CHILDREN)
    try:
        yield
    finally:
        for extra in _ACTIVE_CHILDREN - before:
            _ACTIVE_CHILDREN.discard(extra)
        clear_cli_version_cache()


def _write_fake_cli(tmp_path: Path, version: str = "2.1.283") -> Path:
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLI.replace("__VERSION__", version))
    script.chmod(0o755)
    return script


def _transport(cli_path: Path | str, **options: Any) -> SubprocessCLITransport:
    return SubprocessCLITransport(
        prompt="hi", options=ClaudeAgentOptions(cli_path=str(cli_path), **options)
    )


def _spy_open_process() -> Any:
    """Patch anyio.open_process with a pass-through spy that records spawns."""
    return patch("anyio.open_process", side_effect=anyio.open_process)


def _probe_calls(mock_open: MagicMock) -> list[Any]:
    """The `claude -v` spawns among all recorded anyio.open_process calls."""
    return [c for c in mock_open.call_args_list if list(c.args[0])[1:] == ["-v"]]


def _mock_processes(
    version: bytes = b"2.1.283 (Claude Code)",
) -> tuple[MagicMock, MagicMock]:
    """(version probe, main process) mocks in the style used by test_transport."""
    version_process = MagicMock()
    version_process.stdout = MagicMock()
    version_process.stdout.receive = AsyncMock(return_value=version)
    version_process.terminate = MagicMock()
    version_process.wait = AsyncMock()

    process = MagicMock()
    process.returncode = None
    process.stdout = MagicMock()
    process.stderr = None
    process.stdin = MagicMock(aclose=AsyncMock())
    process.terminate = MagicMock()
    process.wait = AsyncMock()
    return version_process, process


class _ByteChunks:
    """A minimal byte receive stream for TextReceiveStream to wrap."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if not self._chunks:
            raise anyio.EndOfStream
        return self._chunks.pop(0)

    async def aclose(self) -> None:
        self._chunks.clear()


async def _aiter(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


async def _connect_and_close(transport: SubprocessCLITransport) -> None:
    await transport.connect()
    await transport.close()


class TestVersionProbeCache:
    @posix_only
    async def test_probe_cli_version_spawns_once_per_binary(
        self, tmp_path: Path
    ) -> None:
        script = _write_fake_cli(tmp_path)
        with _spy_open_process() as mock_open:
            assert await _probe_cli_version(str(script)) == "2.1.283"
            assert await _probe_cli_version(str(script)) == "2.1.283"
        assert mock_open.call_count == 1

    @posix_only
    async def test_two_transports_share_one_probe(self, tmp_path: Path) -> None:
        script = _write_fake_cli(tmp_path)
        transports = [_transport(script) for _ in range(2)]
        with _spy_open_process() as mock_open:
            for transport in transports:
                await _connect_and_close(transport)
        assert len(_probe_calls(mock_open)) == 1
        assert mock_open.call_count == 3  # one probe, two CLI spawns
        assert [t.cli_version for t in transports] == ["2.1.283", "2.1.283"]

    @posix_only
    async def test_mtime_change_invalidates_cache(self, tmp_path: Path) -> None:
        script = _write_fake_cli(tmp_path)
        with _spy_open_process() as mock_open:
            first = _transport(script)
            await _connect_and_close(first)
            st = script.stat()
            os.utime(script, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
            second = _transport(script)
            await _connect_and_close(second)
        assert len(_probe_calls(mock_open)) == 2
        assert (first.cli_version, second.cli_version) == ("2.1.283", "2.1.283")

    @posix_only
    async def test_size_change_invalidates_cache_even_with_same_mtime(
        self, tmp_path: Path
    ) -> None:
        script = _write_fake_cli(tmp_path)
        original = script.stat()
        with _spy_open_process() as mock_open:
            first = _transport(script)
            await _connect_and_close(first)
            # A longer version string changes the size; pin the mtime back so
            # the size alone has to invalidate the entry.
            script.write_text(FAKE_CLI.replace("__VERSION__", "12.1.283"))
            os.utime(script, ns=(original.st_atime_ns, original.st_mtime_ns))
            second = _transport(script)
            await _connect_and_close(second)
        assert len(_probe_calls(mock_open)) == 2
        assert first.cli_version == "2.1.283"
        assert second.cli_version == "12.1.283"

    async def test_nonexistent_path_is_probed_every_time(self) -> None:
        probe_a, cli_a = _mock_processes(b"2.0.0 (Claude Code)")
        probe_b, cli_b = _mock_processes(b"2.1.283 (Claude Code)")
        with patch(
            "anyio.open_process", side_effect=[probe_a, cli_a, probe_b, cli_b]
        ) as mock_open:
            first = _transport(MISSING_CLI_PATH)
            await _connect_and_close(first)
            second = _transport(MISSING_CLI_PATH)
            await _connect_and_close(second)
        assert len(_probe_calls(mock_open)) == 2
        assert first.cli_version == "2.0.0"
        assert second.cli_version == "2.1.283"

    @posix_only
    async def test_skip_env_skips_probe_and_leaves_version_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SKIP_VERSION_CHECK_ENV, "1")
        script = _write_fake_cli(tmp_path)
        with _spy_open_process() as mock_open:
            transport = _transport(script)
            await _connect_and_close(transport)
        assert _probe_calls(mock_open) == []
        assert mock_open.call_count == 1
        assert transport.cli_version is None

    @posix_only
    async def test_clear_cache_forces_reprobe(self, tmp_path: Path) -> None:
        script = _write_fake_cli(tmp_path)
        with _spy_open_process() as mock_open:
            await _connect_and_close(_transport(script))
            clear_cli_version_cache()
            await _connect_and_close(_transport(script))
        assert len(_probe_calls(mock_open)) == 2

    @posix_only
    async def test_minimum_version_warning_once_per_binary(
        self, tmp_path: Path
    ) -> None:
        script = _write_fake_cli(tmp_path, version="1.0.0")
        with patch(f"{LOGGER_NAME}.logger") as mock_logger:
            for _ in range(2):
                transport = _transport(script)
                await _connect_and_close(transport)
        assert mock_logger.warning.call_count == 1
        args = mock_logger.warning.call_args.args
        assert args[1] == "1.0.0"
        assert args[2] == str(script)
        assert transport.cli_version == "1.0.0"

    @posix_only
    async def test_verbatim_prompts_warning_on_every_connect(
        self, tmp_path: Path
    ) -> None:
        script = _write_fake_cli(tmp_path, version="2.1.247")
        with (
            patch(f"{LOGGER_NAME}.logger") as mock_logger,
            _spy_open_process() as mock_open,
        ):
            for _ in range(2):
                transport = _transport(script, verbatim_prompts=True)
                await _connect_and_close(transport)
        verbatim = [
            c
            for c in mock_logger.warning.call_args_list
            if "verbatim_prompts" in c.args[0]
        ]
        assert len(verbatim) == 2
        assert all(c.args[-1] == "2.1.248" for c in verbatim)
        assert len(_probe_calls(mock_open)) == 1  # evaluated from the cache
        assert transport.cli_version == "2.1.247"


class TestStderrTail:
    @posix_only
    async def test_process_error_carries_stderr_tail(self, tmp_path: Path) -> None:
        script = _write_fake_cli(tmp_path)
        transport = _transport(script, env={"FAKE_CLI_SCENARIO": "stderr-then-exit"})
        await transport.connect()
        try:
            with pytest.raises(ProcessError) as exc_info:
                async for _ in transport.read_messages():
                    pass
        finally:
            await transport.close()
        error = exc_info.value
        assert error.exit_code == 1
        assert error.stderr is not None
        assert "warn: first diagnostic line" in error.stderr
        assert "warn: second diagnostic line" in error.stderr
        assert "warn: first diagnostic line" in str(error)
        assert "warn: second diagnostic line" in str(error)
        assert "Check stderr output for details" not in str(error)

    @posix_only
    async def test_write_to_dead_process_names_exit_code_and_tail(
        self, tmp_path: Path
    ) -> None:
        script = _write_fake_cli(tmp_path)
        transport = _transport(script, env={"FAKE_CLI_SCENARIO": "exit-now"})
        await transport.connect()
        try:
            process = transport._process
            assert process is not None
            with anyio.fail_after(10):
                await process.wait()
            with pytest.raises(CLIConnectionError) as exc_info:
                await transport.write('{"type": "control_request"}\n')
        finally:
            await transport.close()
        message = str(exc_info.value)
        assert "Cannot write to terminated process (exit code: 3)" in message
        assert "fatal: no credentials found" in message

    async def test_failed_stdin_write_reports_exit_and_tail(self) -> None:
        """On asyncio the exit code can lag behind the broken pipe; the write
        failure still has to name the exit and carry the stderr tail."""
        transport = _transport(MISSING_CLI_PATH)
        process = MagicMock(returncode=None)

        async def _wait() -> int:
            process.returncode = 1
            return 1

        process.wait = AsyncMock(side_effect=_wait)
        transport._process = process
        transport._ready = True
        transport._stdin_stream = MagicMock(
            send=AsyncMock(side_effect=BrokenPipeError("pipe closed"))
        )
        transport._stderr_stream = _aiter(["Error: API key invalid\n"])  # type: ignore[assignment]
        await transport._handle_stderr()

        with pytest.raises(CLIConnectionError) as exc_info:
            await transport.write("{}\n")
        message = str(exc_info.value)
        assert "Failed to write to process stdin" in message
        assert "exited with code 1" in message
        assert "Error: API key invalid" in message

    async def test_process_error_keeps_placeholder_without_stderr(self) -> None:
        transport = _transport(MISSING_CLI_PATH)
        transport._process = MagicMock()
        transport._process.wait = AsyncMock(return_value=1)
        transport._stdout_stream = _aiter([])  # type: ignore[assignment]
        with pytest.raises(ProcessError) as exc_info:
            async for _ in transport.read_messages():
                pass
        assert exc_info.value.stderr == "Check stderr output for details"

    async def test_tail_is_none_before_any_line(self) -> None:
        assert _transport(MISSING_CLI_PATH)._stderr_tail_text() is None

    async def test_tail_keeps_only_the_most_recent_lines(self) -> None:
        transport = _transport(MISSING_CLI_PATH)
        transport._stderr_stream = _aiter([f"line-{i:03d}\n" for i in range(100)])  # type: ignore[assignment]
        await transport._handle_stderr()
        tail = transport._stderr_tail_text()
        assert tail is not None
        lines = tail.split("\n")
        assert len(lines) == _STDERR_TAIL_MAX_LINES
        assert lines[0] == f"line-{100 - _STDERR_TAIL_MAX_LINES:03d}"
        assert lines[-1] == "line-099"

    async def test_tail_is_bounded_in_characters(self) -> None:
        transport = _transport(MISSING_CLI_PATH)
        lines = [f"{i:02d}" + "x" * 998 for i in range(20)]  # 1000 chars each
        transport._stderr_stream = _aiter([line + "\n" for line in lines])  # type: ignore[assignment]
        await transport._handle_stderr()
        tail = transport._stderr_tail_text()
        assert tail is not None
        assert len(tail) <= _STDERR_TAIL_MAX_CHARS
        assert tail.count("\n") == 7  # 8 lines fit, the 9th would not
        assert tail.endswith(lines[-1])

    async def test_single_oversized_line_is_truncated_not_dropped(self) -> None:
        transport = _transport(MISSING_CLI_PATH)
        transport._stderr_stream = _aiter(["E" * (_STDERR_TAIL_MAX_CHARS * 2) + "\n"])  # type: ignore[assignment]
        await transport._handle_stderr()
        tail = transport._stderr_tail_text()
        assert tail is not None
        assert len(tail) == _STDERR_TAIL_MAX_CHARS
        assert set(tail) == {"E"}


class TestStderrTee:
    @posix_only
    async def test_lines_reach_sys_stderr_without_callback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _write_fake_cli(tmp_path)
        transport = _transport(script, env={"FAKE_CLI_SCENARIO": "stderr-then-exit"})
        await transport.connect()
        try:
            with pytest.raises(ProcessError):
                async for _ in transport.read_messages():
                    pass
        finally:
            await transport.close()
        err = capsys.readouterr().err
        assert "warn: first diagnostic line\n" in err
        assert "warn: second diagnostic line\n" in err

    @posix_only
    async def test_callback_receives_lines_and_nothing_is_teed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _write_fake_cli(tmp_path)
        received: list[str] = []
        transport = _transport(
            script,
            stderr=received.append,
            env={"FAKE_CLI_SCENARIO": "stderr-then-exit"},
        )
        await transport.connect()
        try:
            with pytest.raises(ProcessError) as exc_info:
                async for _ in transport.read_messages():
                    pass
        finally:
            await transport.close()
        assert received == [
            "warn: first diagnostic line",
            "warn: second diagnostic line",
        ]
        assert "diagnostic line" not in capsys.readouterr().err
        # The tail is kept for diagnostics whether or not a callback is set.
        assert exc_info.value.stderr is not None
        assert "warn: first diagnostic line" in exc_info.value.stderr

    async def test_stderr_is_piped_and_read_by_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        version_process, process = _mock_processes()
        process.stderr = _ByteChunks([b"mock line one\nmock line two\n"])
        with patch(
            "anyio.open_process", side_effect=[version_process, process]
        ) as mock_open:
            transport = _transport(MISSING_CLI_PATH)
            await transport.connect()
            assert mock_open.call_args_list[1].kwargs["stderr"] is PIPE
            assert transport._stderr_task is not None
            with anyio.fail_after(5):
                await transport._stderr_task.wait()
            await transport.close()
        assert transport._stderr_tail_text() == "mock line one\nmock line two"
        assert "mock line one\nmock line two\n" in capsys.readouterr().err

    async def test_inherit_env_restores_fd_inheritance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_INHERIT_STDERR_ENV, "1")
        version_process, process = _mock_processes()
        process.stderr = _ByteChunks([b"never read\n"])
        with patch(
            "anyio.open_process", side_effect=[version_process, process]
        ) as mock_open:
            transport = _transport(MISSING_CLI_PATH)
            await transport.connect()
            assert mock_open.call_args_list[1].kwargs["stderr"] is None
            assert transport._stderr_task is None
            assert transport._stderr_stream is None
            await transport.close()
        assert transport._stderr_tail_text() is None

    async def test_inherit_env_does_not_silence_a_callback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_INHERIT_STDERR_ENV, "1")
        received: list[str] = []
        version_process, process = _mock_processes()
        process.stderr = _ByteChunks([b"for the callback\n"])
        with patch(
            "anyio.open_process", side_effect=[version_process, process]
        ) as mock_open:
            transport = _transport(MISSING_CLI_PATH, stderr=received.append)
            await transport.connect()
            assert mock_open.call_args_list[1].kwargs["stderr"] is PIPE
            assert transport._stderr_task is not None
            with anyio.fail_after(5):
                await transport._stderr_task.wait()
            await transport.close()
        assert received == ["for the callback"]

    async def test_tee_tolerates_missing_or_closed_sys_stderr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = _transport(MISSING_CLI_PATH)
        monkeypatch.setattr(sys, "stderr", None)
        transport._stderr_stream = _aiter(["one\n"])  # type: ignore[assignment]
        await transport._handle_stderr()
        closed = io.StringIO()
        closed.close()
        monkeypatch.setattr(sys, "stderr", closed)
        transport._stderr_stream = _aiter(["two\n"])  # type: ignore[assignment]
        await transport._handle_stderr()
        assert transport._stderr_tail_text() == "one\ntwo"

    async def test_unreadable_stderr_stream_is_only_a_debug_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        version_process, process = _mock_processes()
        process.stderr = MagicMock()  # receive() is not awaitable: reader fails
        with patch("anyio.open_process", side_effect=[version_process, process]):
            transport = _transport(MISSING_CLI_PATH)
            with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
                await transport.connect()
                assert transport._stderr_task is not None
                with anyio.fail_after(5):
                    await transport._stderr_task.wait()
            await transport.close()
        records = [r for r in caplog.records if r.name == LOGGER_NAME]
        assert any("stderr stream read failed" in r.getMessage() for r in records)
        assert all(r.levelno <= logging.DEBUG for r in records)
        assert transport._stderr_tail_text() is None
