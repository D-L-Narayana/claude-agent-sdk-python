"""End-to-end workflow against a stand-in CLI: hooks, session store, export, sync.

The stand-in speaks the CLI's stream-json protocol well enough to drive the SDK
through one real multi-feature workflow without the Claude Code binary or an API
key: it answers the version probe and control requests, writes a transcript to
``$CLAUDE_CONFIG_DIR/projects/<project>/<session>.jsonl`` the way the CLI does,
mirrors it with ``transcript_mirror`` frames, asks a PreToolUse hook a question,
and reports session state. A second scenario dies at startup after writing to
stderr.

Every async test here runs under both asyncio and trio (``anyio_backend`` in
conftest.py).
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import anyio
import pytest

from claude_agent_sdk import (
    AbortSignal,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    InitMessage,
    ProcessError,
    ResultMessage,
    SQLiteSessionStore,
    TextBlock,
    UnknownBlock,
    UserMessage,
    export_session_from_store,
    get_session_messages,
    list_sessions_from_store,
    project_key_for_directory,
    query,
    sync_session_to_store,
    to_sdk_message,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(sys.platform == "win32", reason="spawns a shebang script"),
]

SESSION_ID = "4f1d2c3b-5a6e-4f70-8a9b-0c1d2e3f4a5b"

FAKE_CLI = textwrap.dedent(
    """
    #!/usr/bin/env python3
    import json, os, re, sys, uuid

    if "-v" in sys.argv or "--version" in sys.argv:
        print("2.1.283 (Claude Code)")
        sys.exit(0)

    SESSION = os.environ["FAKE_SESSION_ID"]
    scenario = os.environ.get("FAKE_SCENARIO", "ok")

    def emit(obj):
        print(json.dumps(obj), flush=True)

    def truthy(name):
        return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

    def state(value):
        if truthy("CLAUDE_CODE_SDK_READS_SESSION_STATE"):
            emit({"type": "system", "subtype": "session_state_changed", "state": value,
                  "uuid": "u-" + value, "session_id": SESSION, "sdk_host_only": True})

    emit({"type": "system", "subtype": "init", "session_id": SESSION, "model": "m",
          "cwd": os.getcwd(), "tools": ["Write"], "mcp_servers": [],
          "permissionMode": "default", "apiKeySource": "none",
          "slash_commands": ["commit"], "claude_code_version": "2.1.283"})

    if scenario == "crash":
        sys.stderr.write("Error: credential missing for this account\\n")
        sys.stderr.write("hint: run claude login\\n")
        sys.stderr.flush()
        sys.exit(1)

    # The transcript the real CLI would keep under CLAUDE_CONFIG_DIR.
    project_key = re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(os.getcwd()))
    project_dir = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "projects", project_key)
    os.makedirs(project_dir, exist_ok=True)
    transcript = os.path.join(project_dir, SESSION + ".jsonl")

    def record(entry):
        with open(transcript, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\\n")
        emit({"type": "transcript_mirror", "filePath": transcript, "entries": [entry]})

    hook_id = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "control_request":
            request = msg["request"]
            if request.get("subtype") == "initialize":
                for matchers in (request.get("hooks") or {}).values():
                    for matcher in matchers:
                        hook_id = hook_id or matcher["hookCallbackIds"][0]
            emit({"type": "control_response",
                  "response": {"subtype": "success", "request_id": msg["request_id"],
                               "response": {}}})
        elif kind == "user":
            state("running")
            text = msg["message"]["content"]
            user_uuid = str(uuid.uuid4())
            record({"type": "user", "uuid": user_uuid, "parentUuid": None,
                    "sessionId": SESSION, "timestamp": "2026-01-01T00:00:00.000Z",
                    "cwd": os.getcwd(), "message": {"role": "user", "content": text}})
            emit({"type": "control_request", "request_id": "hook-1",
                  "request": {"subtype": "hook_callback", "callback_id": hook_id,
                              "input": {"hook_event_name": "PreToolUse",
                                        "tool_name": "Write", "tool_input": {}},
                              "tool_use_id": "toolu_1"}})
            # Answer arrives as a control_response below.
            pending_user = user_uuid
        elif kind == "control_response" and msg["response"].get("request_id") == "hook-1":
            content = [{"type": "text", "text": "DONE"},
                       {"type": "mystery_block", "payload": {"n": 1}}]
            assistant_uuid = str(uuid.uuid4())
            record({"type": "assistant", "uuid": assistant_uuid, "parentUuid": pending_user,
                    "sessionId": SESSION, "timestamp": "2026-01-01T00:00:01.000Z",
                    "message": {"role": "assistant", "model": "m", "content": content}})
            emit({"type": "assistant", "parent_tool_use_id": None, "session_id": SESSION,
                  "uuid": assistant_uuid,
                  "message": {"role": "assistant", "model": "m", "content": content}})
            emit({"type": "result", "subtype": "success", "duration_ms": 1,
                  "duration_api_ms": 1, "is_error": False, "num_turns": 1,
                  "session_id": SESSION, "result": "DONE", "uuid": str(uuid.uuid4())})
            state("idle")
    """
).lstrip()


def _write_fake_cli(tmp_path: Path) -> Path:
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLI)
    script.chmod(0o755)
    return script


def _options(
    tmp_path: Path, config_dir: Path, project: Path, **extra: Any
) -> ClaudeAgentOptions:
    env = {"CLAUDE_CONFIG_DIR": str(config_dir), "FAKE_SESSION_ID": SESSION_ID}
    env.update(extra.pop("env", {}))
    return ClaudeAgentOptions(
        cli_path=str(_write_fake_cli(tmp_path)),
        cwd=str(project),
        env=env,
        **extra,
    )


async def test_client_workflow_hooks_store_export_and_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    seen_signals: list[Any] = []

    async def hook(
        input_data: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        seen_signals.append(context["signal"])
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            }
        }

    async with SQLiteSessionStore(tmp_path / "sessions.db") as store:
        options = _options(
            tmp_path,
            config_dir,
            project,
            hooks={"PreToolUse": [HookMatcher(hooks=[hook])]},
            session_store=store,
        )
        messages: list[Any] = []
        with anyio.fail_after(30):
            async with ClaudeSDKClient(options=options) as client:
                await client.query("go")
                async for message in client.receive_response():
                    messages.append(message)
                assert client.session_id == SESSION_ID
                assert client.cli_version == "2.1.283"

        # --- live stream: typed init, hook signal, unknown block preserved, result
        assert isinstance(messages[0], InitMessage)
        assert messages[0].model == "m"
        assert messages[0].permission_mode == "default"
        assert messages[0].slash_commands == ["commit"]
        assert seen_signals and isinstance(seen_signals[0], AbortSignal)
        assert seen_signals[0].aborted is False
        assistant = next(m for m in messages if isinstance(m, AssistantMessage))
        assert any(
            isinstance(b, TextBlock) and b.text == "DONE" for b in assistant.content
        )
        unknown = [b for b in assistant.content if isinstance(b, UnknownBlock)]
        assert unknown and unknown[0].type == "mystery_block"
        assert unknown[0].data["payload"] == {"n": 1}
        result = messages[-1]
        assert isinstance(result, ResultMessage) and result.session_id == SESSION_ID

        # --- the mirror reached the durable store (flushed before the result)
        listed = await list_sessions_from_store(store, directory=str(project))
        assert [s.session_id for s in listed] == [SESSION_ID]
        assert listed[0].summary == "go"

        # --- store -> disk export into a fresh config dir, readable by the disk helpers
        export_dir = tmp_path / "exported-config"
        export_dir.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(export_dir))
        exported = await export_session_from_store(
            store, SESSION_ID, directory=str(project)
        )
        assert exported.exists() and exported.name == f"{SESSION_ID}.jsonl"
        if os.name != "nt":
            assert exported.stat().st_mode & 0o777 == 0o600
        exported_messages = get_session_messages(SESSION_ID, directory=str(project))
        assert [m.type for m in exported_messages] == ["user", "assistant"]
        typed = [to_sdk_message(m) for m in exported_messages]
        assert isinstance(typed[0], UserMessage) and typed[0].content == "go"
        assert isinstance(typed[1], AssistantMessage)
        assert typed[1].content[0] == TextBlock(text="DONE")

        # --- disk -> second store sync is complete and idempotent
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
        async with SQLiteSessionStore(tmp_path / "second.db") as second:
            first_pass = await sync_session_to_store(
                SESSION_ID, second, directory=str(project)
            )
            assert first_pass.appended == 2 and first_pass.skipped == 0
            second_pass = await sync_session_to_store(
                SESSION_ID, second, directory=str(project)
            )
            assert second_pass.appended == 0 and second_pass.skipped == 2
            assert await second.load(
                {"project_key": first_pass.project_key, "session_id": SESSION_ID}
            ) == await store.load(
                {"project_key": first_pass.project_key, "session_id": SESSION_ID}
            )


async def test_query_surfaces_stderr_tail_when_cli_dies(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    options = _options(tmp_path, config_dir, project, env={"FAKE_SCENARIO": "crash"})

    with anyio.fail_after(30), pytest.raises(ProcessError) as exc_info:
        async for _ in query(prompt="go", options=options):
            pass

    err = exc_info.value
    assert err.exit_code == 1
    assert err.stderr is not None and "credential missing" in err.stderr
    assert "credential missing" in str(err)


async def test_invalid_option_combination_fails_before_spawn(tmp_path: Path) -> None:
    options = ClaudeAgentOptions(
        cli_path=str(tmp_path / "never-run"),
        resume=SESSION_ID,
        continue_conversation=True,
    )
    with pytest.raises(ValueError, match="continue_conversation"):
        async with ClaudeSDKClient(options=options):
            pass
    with pytest.raises(ValueError, match="continue_conversation"):
        async for _ in query(prompt="go", options=options):
            pass
    assert not (tmp_path / "never-run").exists()


async def test_export_refuses_target_created_during_store_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``overwrite=False`` must never replace a transcript that another local
    writer created while the export was awaiting the store: the export fails
    with ``FileExistsError`` and the writer's bytes survive untouched. Uses the
    real SQLite store and runs on both backends.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    project_key = project_key_for_directory(str(project))
    target = config_dir / "projects" / project_key / f"{SESSION_ID}.jsonl"
    sentinel = b'{"type":"user","message":{"content":"concurrent writer"}}\n'
    loaded = anyio.Event()
    release = anyio.Event()

    class PausedStore(SQLiteSessionStore):
        async def load(self, key: Any) -> Any:
            entries = await super().load(key)
            loaded.set()
            await release.wait()
            return entries

    outcome: dict[str, Any] = {}

    async def run_export() -> None:
        try:
            await export_session_from_store(
                store, SESSION_ID, directory=str(project), include_subagents=False
            )
        except FileExistsError as exc:
            outcome["error"] = exc

    async with PausedStore(tmp_path / "store.db") as store:
        await store.append(
            {"project_key": project_key, "session_id": SESSION_ID},
            [
                {
                    "type": "user",
                    "uuid": "entry-one",
                    "sessionId": SESSION_ID,
                    "message": {"role": "user", "content": "stored transcript"},
                }
            ],
        )
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_export)
                await loaded.wait()
                # A second local writer creates the target after the export's
                # precondition check but before its store load returns.
                target.parent.mkdir(parents=True)
                target.write_bytes(sentinel)
                release.set()

    assert isinstance(outcome.get("error"), FileExistsError)
    assert target.read_bytes() == sentinel
    # Nothing of the refused export is left behind next to the writer's file.
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]


async def test_export_refuses_subagent_file_created_during_store_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same guarantee for subagent transcripts: a ``subagents/*.jsonl``
    another writer creates while the export is reading the store is never
    replaced. The export fails with ``FileExistsError``, commits no main
    transcript, and leaves nothing of its own behind — including the writer's
    directories, which it must not remove.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    project_key = project_key_for_directory(str(project))
    project_dir = config_dir / "projects" / project_key
    main_target = project_dir / f"{SESSION_ID}.jsonl"
    sub_target = project_dir / SESSION_ID / "subagents" / "agent-x.jsonl"
    sentinel = b'{"type":"assistant","message":{"content":"writer owns this"}}\n'
    listed = anyio.Event()
    release = anyio.Event()

    class PausedStore(SQLiteSessionStore):
        async def list_subkeys(self, key: Any) -> Any:
            subkeys = await super().list_subkeys(key)
            listed.set()
            await release.wait()
            return subkeys

    outcome: dict[str, Any] = {}

    async def run_export() -> None:
        try:
            await export_session_from_store(store, SESSION_ID, directory=str(project))
        except FileExistsError as exc:
            outcome["error"] = exc

    def entry(uid: str, text: str) -> dict[str, Any]:
        return {
            "type": "user",
            "uuid": uid,
            "sessionId": SESSION_ID,
            "message": {"role": "user", "content": text},
        }

    async with PausedStore(tmp_path / "store.db") as store:
        main_key: dict[str, Any] = {
            "project_key": project_key,
            "session_id": SESSION_ID,
        }
        await store.append(main_key, [entry("u1", "main transcript")])  # type: ignore[arg-type]
        await store.append(
            {**main_key, "subpath": "subagents/agent-x"},  # type: ignore[arg-type]
            [entry("s1", "stored subagent transcript")],
        )
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_export)
                await listed.wait()
                # The writer lands the subagent transcript while the export is
                # still reading the store.
                sub_target.parent.mkdir(parents=True)
                sub_target.write_bytes(sentinel)
                release.set()

    assert isinstance(outcome.get("error"), FileExistsError)
    assert sub_target.read_bytes() == sentinel
    assert not main_target.exists()
    remaining = sorted(
        p.relative_to(project_dir).as_posix()
        for p in project_dir.rglob("*")
        if p.is_file()
    )
    assert remaining == [f"{SESSION_ID}/subagents/agent-x.jsonl"]


def test_fake_cli_speaks_the_protocol(tmp_path: Path) -> None:
    """Guard the fixture: a broken stand-in would make the tests above pass or
    fail for the wrong reason."""
    import subprocess

    script = _write_fake_cli(tmp_path)
    assert subprocess.run(
        [str(script), "-v"], capture_output=True, text=True, check=True
    ).stdout.startswith("2.1.283")
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    proc = subprocess.run(
        [str(script)],
        input=json.dumps(
            {
                "type": "control_request",
                "request_id": "1",
                "request": {"subtype": "initialize", "hooks": None},
            }
        )
        + "\n",
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "CLAUDE_CONFIG_DIR": str(config_dir),
            "FAKE_SESSION_ID": SESSION_ID,
        },
    )
    lines = [json.loads(line) for line in proc.stdout.splitlines()]
    assert lines[0]["subtype"] == "init"
    assert lines[1]["response"]["request_id"] == "1"
