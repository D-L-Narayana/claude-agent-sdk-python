"""Tests for ``export_session_from_store`` (SessionStore → local JSONL export).

The export is the inverse of ``import_session_to_store``: it writes a
store-held session to ``<CLAUDE_CONFIG_DIR>/projects/<project_key>/`` in the
exact layout the CLI and the disk-reading helpers (``get_session_messages``,
``list_sessions``, ``list_subagents``, ``get_subagent_messages``) expect.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import uuid as uuid_mod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from claude_agent_sdk import (
    InMemorySessionStore,
    get_session_messages,
    get_session_messages_from_store,
    get_subagent_messages,
    import_session_to_store,
    list_sessions,
    list_subagents,
    project_key_for_directory,
    rename_session_via_store,
)
from claude_agent_sdk._internal.session_export import export_session_from_store
from claude_agent_sdk._internal.session_store import file_path_to_session_key
from claude_agent_sdk.types import SessionKey, SessionStore

pytestmark = pytest.mark.anyio

SESSION_ID = "550e8400-e29b-41d4-a716-446655440000"
AGENT_META = {
    "agentType": "general-purpose",
    "toolUseId": "toolu_01",
    "parentAgentId": "a-parent",
}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def claude_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``CLAUDE_CONFIG_DIR`` at an empty temp dir.

    Deliberately does NOT pre-create ``projects/`` — the export must create
    the parent directories itself.
    """
    config_dir = tmp_path / ".claude"
    config_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    return config_dir


@pytest.fixture
def cwd(tmp_path: Path) -> Path:
    d = tmp_path / "project"
    d.mkdir()
    return d


@pytest.fixture
def project_key(cwd: Path) -> str:
    return project_key_for_directory(cwd)


def _user(text: str, uid: str, parent: str | None, sid: str) -> dict[str, Any]:
    return {
        "type": "user",
        "uuid": uid,
        "parentUuid": parent,
        "sessionId": sid,
        "timestamp": "2024-01-01T00:00:00.000Z",
        "message": {"role": "user", "content": text},
    }


def _assistant(text: str, uid: str, parent: str, sid: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": uid,
        "parentUuid": parent,
        "sessionId": sid,
        "timestamp": "2024-01-01T00:00:01.000Z",
        "message": {
            "role": "assistant",
            "model": "test-model",
            "content": [{"type": "text", "text": text}],
        },
    }


def _two_turns(sid: str) -> list[dict[str, Any]]:
    u1, a1, u2, a2 = (str(uuid_mod.uuid4()) for _ in range(4))
    return [
        _user("prompt 0", u1, None, sid),
        _assistant("reply 0", a1, u1, sid),
        _user("prompt 1", u2, a1, sid),
        _assistant("reply 1", a2, u2, sid),
    ]


async def _seed(
    store: SessionStore, project_key: str, sid: str = SESSION_ID
) -> list[dict[str, Any]]:
    """Append a two-turn user/assistant chain under the main key."""
    entries = _two_turns(sid)
    await store.append({"project_key": project_key, "session_id": sid}, entries)  # type: ignore[arg-type]
    return entries


async def _seed_with_subagents(
    store: SessionStore, project_key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Seed main transcript + a direct subagent (with agent_metadata) + a
    nested workflow subagent (without metadata).

    Returns ``(main_entries, sub_entries, deep_entries)``. The synthetic
    ``agent_metadata`` entry is last so an import of the exported tree
    reproduces ``sub_entries`` exactly.
    """
    entries = await _seed(store, project_key)
    su, sa = str(uuid_mod.uuid4()), str(uuid_mod.uuid4())
    sub_entries: list[dict[str, Any]] = [
        _user("sub prompt", su, None, SESSION_ID),
        _assistant("sub reply", sa, su, SESSION_ID),
        {**AGENT_META, "type": "agent_metadata"},
    ]
    sub_key: SessionKey = {
        "project_key": project_key,
        "session_id": SESSION_ID,
        "subpath": "subagents/agent-abc",
    }
    await store.append(sub_key, sub_entries)  # type: ignore[arg-type]
    deep_entries = [_user("deep prompt", str(uuid_mod.uuid4()), None, SESSION_ID)]
    deep_key: SessionKey = {
        "project_key": project_key,
        "session_id": SESSION_ID,
        "subpath": "subagents/workflows/run-1/agent-deep",
    }
    await store.append(deep_key, deep_entries)  # type: ignore[arg-type]
    return entries, sub_entries, deep_entries


def _files_under(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()}


def _read_jsonl(path: Path) -> list[Any]:
    return [
        json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln
    ]


def _assert_mode_0600(path: Path) -> None:
    """Exported transcripts and sidecars are owner-read/write only (POSIX)."""
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path


# ---------------------------------------------------------------------------
# Main transcript
# ---------------------------------------------------------------------------


class TestMainTranscript:
    async def test_export_writes_cli_layout_readable_by_disk_helpers(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        store = InMemorySessionStore()
        entries = await _seed(store, project_key)

        path = await export_session_from_store(store, SESSION_ID, directory=str(cwd))

        # <CLAUDE_CONFIG_DIR>/projects/<sanitized realpath(cwd)>/<sid>.jsonl —
        # the file the CLI reads for ``claude --resume <sid>`` from ``cwd``.
        expected = claude_config_dir / "projects" / project_key / f"{SESSION_ID}.jsonl"
        assert path == expected
        assert path.is_file()
        _assert_mode_0600(path)
        # One compact JSON object per line, round-tripping the store entries.
        assert _read_jsonl(path) == entries
        # The on-disk path maps back to the key the live mirror would use.
        assert file_path_to_session_key(
            str(path), str(claude_config_dir / "projects")
        ) == {"project_key": project_key, "session_id": SESSION_ID}

        msgs = get_session_messages(SESSION_ID, directory=str(cwd))
        assert [m.uuid for m in msgs] == [e["uuid"] for e in entries]
        assert [m.type for m in msgs] == ["user", "assistant", "user", "assistant"]
        assert [m.message for m in msgs] == [e["message"] for e in entries]
        assert all(m.session_id == SESSION_ID for m in msgs)
        # Disk and store reads agree on the exported session.
        assert msgs == await get_session_messages_from_store(
            store, SESSION_ID, directory=str(cwd)
        )

        sessions = list_sessions(directory=str(cwd), include_worktrees=False)
        assert [s.session_id for s in sessions] == [SESSION_ID]
        assert sessions[0].summary == "prompt 0"
        assert sessions[0].first_prompt == "prompt 0"
        assert sessions[0].created_at == 1704067200000
        assert sessions[0].file_size == path.stat().st_size

    async def test_directory_defaults_to_cwd(
        self,
        claude_config_dir: Path,
        cwd: Path,
        project_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = InMemorySessionStore()
        await _seed(store, project_key)
        monkeypatch.chdir(cwd)

        # overwrite=True on a not-yet-existing target is simply a write.
        path = await export_session_from_store(store, SESSION_ID, overwrite=True)

        assert (
            path == claude_config_dir / "projects" / project_key / f"{SESSION_ID}.jsonl"
        )
        assert path.is_file()
        assert len(get_session_messages(SESSION_ID, directory=str(cwd))) == 4

    async def test_existing_target_requires_overwrite(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        class SpyStore(InMemorySessionStore):
            def __init__(self) -> None:
                super().__init__()
                self.loads = 0

            async def load(self, key):  # type: ignore[override]
                self.loads += 1
                return await super().load(key)

        store = SpyStore()
        entries = await _seed(store, project_key)
        path = await export_session_from_store(store, SESSION_ID, directory=str(cwd))
        original = path.read_bytes()
        assert store.loads == 1

        with pytest.raises(FileExistsError, match=SESSION_ID):
            await export_session_from_store(store, SESSION_ID, directory=str(cwd))
        assert path.read_bytes() == original
        # Refused before the store is consulted — no wasted (remote) load.
        assert store.loads == 1

        # The store moved on (rename) — overwrite=True re-exports current state.
        await rename_session_via_store(store, SESSION_ID, "Renamed", directory=str(cwd))
        again = await export_session_from_store(
            store, SESSION_ID, directory=str(cwd), overwrite=True
        )
        assert again == path
        assert store.loads == 2
        lines = _read_jsonl(path)
        assert lines[:4] == entries
        assert lines[4]["type"] == "custom-title"
        assert lines[4]["customTitle"] == "Renamed"
        _assert_mode_0600(path)
        sessions = list_sessions(directory=str(cwd), include_worktrees=False)
        assert [s.session_id for s in sessions] == [SESSION_ID]
        assert sessions[0].custom_title == "Renamed"
        assert sessions[0].summary == "Renamed"

    async def test_type_key_hoisted_for_key_reordering_adapters(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        """The ``SessionStore.load`` contract lets adapters reorder object keys
        (Postgres JSONB does). ``list_sessions`` keys tag extraction on a
        ``{"type":"tag"`` line prefix, so the export must hoist ``type`` first
        — matching what ``get_session_info_from_store`` does for the same
        store — or disk and store reads of the same session would disagree."""

        class ReorderingStore(InMemorySessionStore):
            async def load(self, key):  # type: ignore[override]
                entries = await super().load(key)
                if entries is None:
                    return None
                return [dict(sorted(e.items())) for e in entries]

        store = ReorderingStore()
        await _seed(store, project_key)
        await store.append(
            {"project_key": project_key, "session_id": SESSION_ID},
            [{"type": "tag", "tag": "exp", "sessionId": SESSION_ID}],
        )

        path = await export_session_from_store(store, SESSION_ID, directory=str(cwd))

        lines = path.read_text(encoding="utf-8").splitlines()
        assert all(ln.startswith('{"type":"') for ln in lines)
        sessions = list_sessions(directory=str(cwd), include_worktrees=False)
        assert [s.session_id for s in sessions] == [SESSION_ID]
        assert sessions[0].tag == "exp"
        assert sessions[0].summary == "prompt 0"

    async def test_non_uuid_session_id_raises_value_error(
        self, claude_config_dir: Path, cwd: Path
    ) -> None:
        class SpyStore(InMemorySessionStore):
            def __init__(self) -> None:
                super().__init__()
                self.loads = 0

            async def load(self, key):  # type: ignore[override]
                self.loads += 1
                return await super().load(key)

        store = SpyStore()
        for bad in ("not-a-uuid", "", "../../etc/passwd"):
            with pytest.raises(ValueError, match="Invalid session_id"):
                await export_session_from_store(store, bad, directory=str(cwd))
        # Rejected before the store is consulted and before anything is written.
        assert store.loads == 0
        assert not (claude_config_dir / "projects").exists()

    async def test_missing_session_raises_file_not_found(
        self, claude_config_dir: Path, cwd: Path, project_key: str, tmp_path: Path
    ) -> None:
        store = InMemorySessionStore()
        with pytest.raises(FileNotFoundError, match=SESSION_ID):
            await export_session_from_store(store, SESSION_ID, directory=str(cwd))

        # An emptied session (load() -> []) is equally "not there".
        await store.append({"project_key": project_key, "session_id": SESSION_ID}, [])
        with pytest.raises(FileNotFoundError, match=SESSION_ID):
            await export_session_from_store(store, SESSION_ID, directory=str(cwd))

        # The key is derived from ``directory`` — a session stored under one
        # project is not found via another.
        other = tmp_path / "other-project"
        other.mkdir()
        await _seed(store, project_key)
        with pytest.raises(FileNotFoundError, match=SESSION_ID):
            await export_session_from_store(store, SESSION_ID, directory=str(other))

        assert not (claude_config_dir / "projects").exists()


# ---------------------------------------------------------------------------
# Subagents
# ---------------------------------------------------------------------------


class TestSubagents:
    async def test_subagent_transcripts_and_sidecars_written(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        store = InMemorySessionStore()
        _entries, sub_entries, deep_entries = await _seed_with_subagents(
            store, project_key
        )

        path = await export_session_from_store(store, SESSION_ID, directory=str(cwd))

        session_dir = path.with_suffix("")
        agent_file = session_dir / "subagents" / "agent-abc.jsonl"
        meta_file = session_dir / "subagents" / "agent-abc.meta.json"
        deep_file = (
            session_dir / "subagents" / "workflows" / "run-1" / "agent-deep.jsonl"
        )
        assert agent_file.is_file()
        assert meta_file.is_file()
        assert deep_file.is_file()
        for f in (agent_file, meta_file, deep_file):
            _assert_mode_0600(f)
        # The agent_metadata entry is not a transcript line — it becomes the
        # .meta.json sidecar (synthetic ``type`` discriminator stripped).
        assert _read_jsonl(agent_file) == sub_entries[:2]
        assert json.loads(meta_file.read_text(encoding="utf-8")) == AGENT_META
        assert _read_jsonl(deep_file) == deep_entries
        assert not deep_file.with_name("agent-deep.meta.json").exists()

        assert sorted(list_subagents(SESSION_ID, directory=str(cwd))) == ["abc", "deep"]
        msgs = get_subagent_messages(SESSION_ID, "abc", directory=str(cwd))
        assert [m.uuid for m in msgs] == [e["uuid"] for e in sub_entries[:2]]
        assert all(m.parent_tool_use_id == "toolu_01" for m in msgs)
        assert all(m.parent_agent_id == "a-parent" for m in msgs)
        deep_msgs = get_subagent_messages(SESSION_ID, "deep", directory=str(cwd))
        assert [m.uuid for m in deep_msgs] == [deep_entries[0]["uuid"]]
        assert deep_msgs[0].parent_tool_use_id is None
        assert deep_msgs[0].parent_agent_id is None

    async def test_exported_files_are_owner_only_regardless_of_umask(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        """Every exported file — main transcript, subagent transcripts and
        ``.meta.json`` sidecars — ends up with mode 0o600 under a fresh
        ``CLAUDE_CONFIG_DIR``. The mode is set explicitly (chmod after write,
        preserved by the rename into place), not inherited from the process
        umask: with the most permissive umask the files would otherwise be
        world-readable 0o666."""
        store = InMemorySessionStore()
        await _seed_with_subagents(store, project_key)

        old_umask = os.umask(0o000)  # most permissive: would yield 0o666 files
        try:
            path = await export_session_from_store(
                store, SESSION_ID, directory=str(cwd)
            )
        finally:
            os.umask(old_umask)

        session_dir = path.with_suffix("")
        written = _files_under(claude_config_dir)
        assert written == {
            path,
            session_dir / "subagents" / "agent-abc.jsonl",
            session_dir / "subagents" / "agent-abc.meta.json",
            session_dir / "subagents" / "workflows" / "run-1" / "agent-deep.jsonl",
        }
        for f in written:
            _assert_mode_0600(f)

    async def test_store_without_list_subkeys_writes_only_main(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        class MinimalStore(SessionStore):
            """Only the required append/load — list_subkeys is the Protocol
            default, i.e. absent."""

            def __init__(self) -> None:
                self._data: dict[tuple[str, str, str | None], list[Any]] = {}

            async def append(self, key, entries):  # type: ignore[override]
                k = (key["project_key"], key["session_id"], key.get("subpath"))
                self._data.setdefault(k, []).extend(entries)

            async def load(self, key):  # type: ignore[override]
                return self._data.get(
                    (key["project_key"], key["session_id"], key.get("subpath"))
                )

        store = MinimalStore()
        await _seed(store, project_key)
        hidden_key: SessionKey = {
            "project_key": project_key,
            "session_id": SESSION_ID,
            "subpath": "subagents/agent-hidden",
        }
        await store.append(
            hidden_key, [_user("x", str(uuid_mod.uuid4()), None, SESSION_ID)]
        )

        path = await export_session_from_store(store, SESSION_ID, directory=str(cwd))

        assert path.is_file()
        assert not path.with_suffix("").exists()
        assert list_subagents(SESSION_ID, directory=str(cwd)) == []
        assert _files_under(claude_config_dir) == {path}

    async def test_include_subagents_false_skips_subkeys(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        class SpyStore(InMemorySessionStore):
            def __init__(self) -> None:
                super().__init__()
                self.list_subkeys_calls = 0

            async def list_subkeys(self, key):  # type: ignore[override]
                self.list_subkeys_calls += 1
                return await super().list_subkeys(key)

        store = SpyStore()
        await _seed_with_subagents(store, project_key)

        path = await export_session_from_store(
            store, SESSION_ID, directory=str(cwd), include_subagents=False
        )

        assert store.list_subkeys_calls == 0
        assert path.is_file()
        assert not path.with_suffix("").exists()
        assert _files_under(claude_config_dir) == {path}

    async def test_unsafe_subpaths_skipped_with_warning(
        self,
        claude_config_dir: Path,
        cwd: Path,
        project_key: str,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        unsafe = ["../escape", "/etc/passwd", "", "subagents/../../x", "C:\\abs"]

        class EvilStore(InMemorySessionStore):
            async def list_subkeys(self, key):  # type: ignore[override]
                return [*unsafe, "subagents/agent-ok"]

            async def load(self, key):  # type: ignore[override]
                if key.get("subpath") in unsafe:
                    # Unsafe subpaths must be rejected before any load.
                    raise AssertionError(f"loaded unsafe subpath {key!r}")
                return await super().load(key)

        store = EvilStore()
        await _seed(store, project_key)
        ok_key: SessionKey = {
            "project_key": project_key,
            "session_id": SESSION_ID,
            "subpath": "subagents/agent-ok",
        }
        ok_entries = [_user("ok", str(uuid_mod.uuid4()), None, SESSION_ID)]
        await store.append(ok_key, ok_entries)  # type: ignore[arg-type]

        caplog.set_level(logging.WARNING, logger="claude_agent_sdk")
        path = await export_session_from_store(store, SESSION_ID, directory=str(cwd))

        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and "unsafe subpath" in r.getMessage()
        ]
        for bad in unsafe:
            assert any(repr(bad) in w for w in warnings), (bad, warnings)

        session_dir = path.with_suffix("")
        ok_file = session_dir / "subagents" / "agent-ok.jsonl"
        assert ok_file.is_file()
        assert _read_jsonl(ok_file) == ok_entries
        # Nothing was written outside the session directory — the whole temp
        # tree holds exactly the main transcript and the one safe subagent.
        assert _files_under(tmp_path) == {path, ok_file}
        assert not (
            claude_config_dir / "projects" / project_key / "escape.jsonl"
        ).exists()
        assert not (tmp_path / "escape.jsonl").exists()


# ---------------------------------------------------------------------------
# Timeouts and error wrapping
# ---------------------------------------------------------------------------


class TestTimeoutsAndErrors:
    async def test_hanging_load_times_out(
        self, claude_config_dir: Path, cwd: Path
    ) -> None:
        class SlowStore(SessionStore):
            async def append(self, key, entries):  # type: ignore[override]
                pass

            async def load(self, key):  # type: ignore[override]
                await anyio.sleep(3600)
                return None

        with pytest.raises(
            RuntimeError,
            match=(
                rf"SessionStore\.load\(\) for session {SESSION_ID} timed out after "
                r"50ms during session export"
            ),
        ):
            await export_session_from_store(
                SlowStore(), SESSION_ID, directory=str(cwd), load_timeout_ms=50
            )
        assert not (claude_config_dir / "projects").exists()

    async def test_load_failure_is_wrapped_with_context(
        self, claude_config_dir: Path, cwd: Path
    ) -> None:
        class BrokenStore(SessionStore):
            async def append(self, key, entries):  # type: ignore[override]
                pass

            async def load(self, key):  # type: ignore[override]
                raise OSError("network down")

        with pytest.raises(RuntimeError, match=r"SessionStore\.load\(\).*network down"):
            await export_session_from_store(
                BrokenStore(), SESSION_ID, directory=str(cwd)
            )
        assert not (claude_config_dir / "projects").exists()

    async def test_subkey_failure_leaves_no_partial_session(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        """A store failure while exporting subagents must not leave a main
        transcript behind — ``list_sessions`` / ``--resume`` would otherwise
        see a session silently missing its subagent history, and a retry
        without ``overwrite`` would hit ``FileExistsError``."""

        class HungSubkeysStore(InMemorySessionStore):
            async def list_subkeys(self, key):  # type: ignore[override]
                await anyio.sleep(3600)
                return []

        store = HungSubkeysStore()
        await _seed(store, project_key)

        with pytest.raises(RuntimeError, match=r"list_subkeys\(\).*timed out"):
            await export_session_from_store(
                store, SESSION_ID, directory=str(cwd), load_timeout_ms=50
            )

        main = claude_config_dir / "projects" / project_key / f"{SESSION_ID}.jsonl"
        assert not main.exists()
        assert list_sessions(directory=str(cwd), include_worktrees=False) == []
        # No staging leftovers, and the project directory created only for
        # this export is gone again.
        assert _files_under(claude_config_dir) == set()
        assert not main.parent.exists()
        # The retry succeeds without overwrite once the store is healthy.
        healthy = InMemorySessionStore()
        await _seed(healthy, project_key)
        assert (
            await export_session_from_store(healthy, SESSION_ID, directory=str(cwd))
            == main
        )


# ---------------------------------------------------------------------------
# Round trip with import_session_to_store
# ---------------------------------------------------------------------------


class TestRoundTrip:
    async def test_export_then_import_reproduces_entries(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        store = InMemorySessionStore()
        entries, sub_entries, deep_entries = await _seed_with_subagents(
            store, project_key
        )
        await export_session_from_store(store, SESSION_ID, directory=str(cwd))

        fresh = InMemorySessionStore()
        await import_session_to_store(SESSION_ID, fresh, directory=str(cwd))

        main_key: SessionKey = {"project_key": project_key, "session_id": SESSION_ID}
        assert fresh.get_entries(main_key) == entries
        assert (
            fresh.get_entries({**main_key, "subpath": "subagents/agent-abc"})
            == sub_entries
        )
        assert (
            fresh.get_entries(
                {**main_key, "subpath": "subagents/workflows/run-1/agent-deep"}
            )
            == deep_entries
        )
        assert sorted(await fresh.list_subkeys(main_key)) == sorted(
            await store.list_subkeys(main_key)
        )


# ---------------------------------------------------------------------------
# Concurrent local writers
# ---------------------------------------------------------------------------

SENTINEL = b'{"type":"user","message":{"content":"concurrent writer"}}\n'
WRITER_SUB = b'{"type":"user","uuid":"writer-sub","message":{"content":"writer"}}\n'


class _PausedStore(InMemorySessionStore):
    """``InMemorySessionStore`` that pauses one store call so a test can act as
    a concurrent local writer while ``export_session_from_store`` is awaiting
    the store.

    ``pause_on`` names the call: ``"load"`` (the main transcript load),
    ``"list_subkeys"`` or ``"subagent_load"`` (the first subpath load). The
    paused call sets ``paused`` and blocks until ``release`` is set.
    """

    def __init__(self, pause_on: str) -> None:
        super().__init__()
        self.pause_on = pause_on
        self.paused = anyio.Event()
        self.release = anyio.Event()

    async def _pause(self) -> None:
        if not self.paused.is_set():
            self.paused.set()
            await self.release.wait()

    async def load(self, key):  # type: ignore[override]
        entries = await super().load(key)
        is_sub = key.get("subpath") is not None
        if (self.pause_on == "load" and not is_sub) or (
            self.pause_on == "subagent_load" and is_sub
        ):
            await self._pause()
        return entries

    async def list_subkeys(self, key):  # type: ignore[override]
        subkeys = await super().list_subkeys(key)
        if self.pause_on == "list_subkeys":
            await self._pause()
        return subkeys


async def _export_while_paused(
    store: _PausedStore, writer: Callable[[], None], **kwargs: Any
) -> BaseException | None:
    """Run the export in a task; once the store has paused, run ``writer`` (the
    concurrent local writer), release the store and return the export's
    exception (``None`` if it succeeded)."""
    outcome: list[BaseException | None] = []

    async def _export() -> None:
        try:
            await export_session_from_store(store, SESSION_ID, **kwargs)
        except Exception as e:
            outcome.append(e)
        else:
            outcome.append(None)

    with anyio.fail_after(30):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_export)
            await store.paused.wait()
            writer()
            store.release.set()
    return outcome[0]


def _disable_hardlinks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the export see a filesystem without hard links (EPERM), so it has
    to take its exclusive-create fallback for every committed file."""
    from claude_agent_sdk._internal import session_export

    def _no_links(staged: Path, target: Path) -> None:
        raise OSError(errno.EPERM, "hard links not permitted", str(target))

    monkeypatch.setattr(session_export, "_hardlink", _no_links, raising=False)


class TestConcurrentWriters:
    """``overwrite=False`` must never replace a file that appeared at a target
    path after the precondition check — e.g. the CLI or another process
    writing the same session while the export awaits the store — and a
    refused export must leave none of its own files behind and none of the
    writer's files missing or modified."""

    @pytest.mark.parametrize(
        "hardlinks", [True, False], ids=["hardlink", "exclusive-create"]
    )
    async def test_main_target_created_during_load_is_preserved(
        self,
        claude_config_dir: Path,
        cwd: Path,
        project_key: str,
        monkeypatch: pytest.MonkeyPatch,
        hardlinks: bool,
    ) -> None:
        if not hardlinks:
            _disable_hardlinks(monkeypatch)
        store = _PausedStore(pause_on="load")
        await _seed(store, project_key)
        project_dir = claude_config_dir / "projects" / project_key
        target = project_dir / f"{SESSION_ID}.jsonl"

        def writer() -> None:
            # Appears after the export's precondition check, during its
            # awaited store.load().
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(SENTINEL)

        err = await _export_while_paused(
            store, writer, directory=str(cwd), include_subagents=False
        )

        assert isinstance(err, FileExistsError), (
            f"expected FileExistsError, got {err!r}"
        )
        assert str(target) in str(err)
        assert "overwrite=True" in str(err)
        assert target.read_bytes() == SENTINEL
        # Nothing of this export remains: no transcript, no staging dir, no
        # session dir.
        assert _files_under(claude_config_dir) == {target}
        assert not list(project_dir.glob(".export-*"))
        assert not (project_dir / SESSION_ID).exists()

    async def test_main_target_created_during_subagent_phase_is_preserved(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        store = _PausedStore(pause_on="list_subkeys")
        await _seed_with_subagents(store, project_key)
        project_dir = claude_config_dir / "projects" / project_key
        target = project_dir / f"{SESSION_ID}.jsonl"
        session_dir = project_dir / SESSION_ID
        writer_sub = session_dir / "subagents" / "agent-writer.jsonl"

        def writer() -> None:
            writer_sub.parent.mkdir(parents=True, exist_ok=True)
            writer_sub.write_bytes(WRITER_SUB)
            target.write_bytes(SENTINEL)

        err = await _export_while_paused(store, writer, directory=str(cwd))

        assert isinstance(err, FileExistsError), (
            f"expected FileExistsError, got {err!r}"
        )
        assert str(target) in str(err)
        assert target.read_bytes() == SENTINEL
        assert writer_sub.read_bytes() == WRITER_SUB
        # The subagent files this export committed ahead of the refused main
        # transcript are gone again, along with the directories only it
        # created; the writer's files are untouched.
        assert _files_under(claude_config_dir) == {target, writer_sub}
        assert not (session_dir / "subagents" / "workflows").exists()
        assert not list(project_dir.glob(".export-*"))

    @pytest.mark.parametrize(
        "conflict",
        ["subagents/agent-abc.jsonl", "subagents/workflows/run-1/agent-deep.jsonl"],
        ids=["first-subagent", "later-subagent"],
    )
    @pytest.mark.parametrize(
        "hardlinks", [True, False], ids=["hardlink", "exclusive-create"]
    )
    async def test_subagent_file_created_during_pause_is_preserved(
        self,
        claude_config_dir: Path,
        cwd: Path,
        project_key: str,
        monkeypatch: pytest.MonkeyPatch,
        conflict: str,
        hardlinks: bool,
    ) -> None:
        """A writer's subagent transcript at a path the store would also write
        is never replaced. ``later-subagent`` makes the conflict land after
        this export has already committed ``agent-abc.jsonl`` and its sidecar,
        so those must be undone."""
        if not hardlinks:
            _disable_hardlinks(monkeypatch)
        store = _PausedStore(pause_on="subagent_load")
        await _seed_with_subagents(store, project_key)
        project_dir = claude_config_dir / "projects" / project_key
        target = project_dir / f"{SESSION_ID}.jsonl"
        session_dir = project_dir / SESSION_ID
        writer_file = session_dir / conflict

        def writer() -> None:
            writer_file.parent.mkdir(parents=True, exist_ok=True)
            writer_file.write_bytes(WRITER_SUB)
            target.write_bytes(SENTINEL)

        err = await _export_while_paused(store, writer, directory=str(cwd))

        assert isinstance(err, FileExistsError), (
            f"expected FileExistsError, got {err!r}"
        )
        assert str(writer_file) in str(err)
        assert writer_file.read_bytes() == WRITER_SUB
        assert target.read_bytes() == SENTINEL
        assert _files_under(claude_config_dir) == {writer_file, target}
        assert not list(project_dir.glob(".export-*"))

    async def test_overwrite_true_replaces_target_created_during_load(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        """``overwrite=True`` keeps its documented replace semantics even for a
        target that appeared during the export."""
        store = _PausedStore(pause_on="load")
        entries = await _seed(store, project_key)
        project_dir = claude_config_dir / "projects" / project_key
        target = project_dir / f"{SESSION_ID}.jsonl"

        def writer() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(SENTINEL)

        err = await _export_while_paused(
            store, writer, directory=str(cwd), include_subagents=False, overwrite=True
        )

        assert err is None
        assert _read_jsonl(target) == entries
        _assert_mode_0600(target)
        assert not list(project_dir.glob(".export-*"))

    async def test_overwrite_true_replaces_known_files_and_keeps_unknown(
        self, claude_config_dir: Path, cwd: Path, project_key: str
    ) -> None:
        store = InMemorySessionStore()
        entries, sub_entries, deep_entries = await _seed_with_subagents(
            store, project_key
        )
        project_dir = claude_config_dir / "projects" / project_key
        target = project_dir / f"{SESSION_ID}.jsonl"
        session_dir = project_dir / SESSION_ID
        stale = session_dir / "subagents" / "agent-abc.jsonl"
        stale_meta = session_dir / "subagents" / "agent-abc.meta.json"
        deep = session_dir / "subagents" / "workflows" / "run-1" / "agent-deep.jsonl"
        unknown_sub = session_dir / "subagents" / "agent-other.jsonl"
        notes = session_dir / "notes.txt"
        for path, data in (
            (target, SENTINEL),
            (stale, b"stale\n"),
            (stale_meta, b"{}\n"),
            (unknown_sub, WRITER_SUB),
            (notes, b"keep me\n"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        path = await export_session_from_store(
            store, SESSION_ID, directory=str(cwd), overwrite=True
        )

        assert path == target
        assert _read_jsonl(target) == entries
        assert _read_jsonl(stale) == sub_entries[:2]
        assert json.loads(stale_meta.read_text(encoding="utf-8")) == AGENT_META
        assert _read_jsonl(deep) == deep_entries
        # Files the store does not know about are left exactly as they were.
        assert unknown_sub.read_bytes() == WRITER_SUB
        assert notes.read_bytes() == b"keep me\n"
        for f in (target, stale, stale_meta, deep):
            _assert_mode_0600(f)
        assert not list(project_dir.glob(".export-*"))

    async def test_exclusive_create_fallback_exports_full_layout(
        self,
        claude_config_dir: Path,
        cwd: Path,
        project_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Where hard links are unavailable the export still produces the full
        layout with 0o600 modes via its exclusive-create fallback."""
        _disable_hardlinks(monkeypatch)
        store = InMemorySessionStore()
        entries, sub_entries, deep_entries = await _seed_with_subagents(
            store, project_key
        )

        path = await export_session_from_store(store, SESSION_ID, directory=str(cwd))

        session_dir = path.with_suffix("")
        assert _read_jsonl(path) == entries
        assert (
            _read_jsonl(session_dir / "subagents" / "agent-abc.jsonl")
            == (sub_entries[:2])
        )
        assert (
            json.loads(
                (session_dir / "subagents" / "agent-abc.meta.json").read_text(
                    encoding="utf-8"
                )
            )
            == AGENT_META
        )
        assert (
            _read_jsonl(
                session_dir / "subagents" / "workflows" / "run-1" / "agent-deep.jsonl"
            )
            == deep_entries
        )
        for f in _files_under(claude_config_dir):
            _assert_mode_0600(f)
        assert not list(path.parent.glob(".export-*"))
