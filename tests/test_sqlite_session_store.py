"""Tests for :class:`claude_agent_sdk.stores.SQLiteSessionStore`.

Covers the shipped conformance harness (file-backed and ``":memory:"``), the
public ``*_from_store`` / ``*_via_store`` session helpers (including the
``list_session_summaries`` fast path, which must never fall back to
``load()``), ``materialize_resume_session``, and the durability, concurrency
and lifecycle guarantees the store documents. Every async test runs under
both asyncio and trio (see ``conftest.py``).
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import uuid as uuid_mod
from pathlib import Path
from typing import Any

import anyio
import pytest

from claude_agent_sdk import (
    ClaudeAgentOptions,
    delete_session_via_store,
    fork_session_via_store,
    get_session_info_from_store,
    get_session_messages_from_store,
    get_subagent_messages_from_store,
    list_sessions_from_store,
    list_subagents_from_store,
    project_key_for_directory,
    rename_session_via_store,
    tag_session_via_store,
)
from claude_agent_sdk._internal.session_resume import materialize_resume_session
from claude_agent_sdk._internal.session_store_validation import (
    validate_session_store_options,
)
from claude_agent_sdk.stores import SQLiteSessionStore
from claude_agent_sdk.testing import run_session_store_conformance
from claude_agent_sdk.types import SessionKey, SessionStoreEntry

pytestmark = pytest.mark.anyio

DIR = "/workspace/project"
PROJECT_KEY = project_key_for_directory(DIR)
SESSION_ID = "550e8400-e29b-41d4-a716-446655440000"
SESSION_ID_2 = "660e8400-e29b-41d4-a716-446655440000"
_KEY: SessionKey = {"project_key": "proj", "session_id": "sess"}
# ``_user()`` stamps 2024-01-01T00:00:00Z on every entry; as epoch milliseconds.
_T0_MS = 1_704_067_200_000


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


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
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


async def _seed_chain(store: SQLiteSessionStore, sid: str, n: int = 2) -> list[str]:
    """Append ``n`` user/assistant pairs and return their UUIDs in order."""
    key: SessionKey = {"project_key": PROJECT_KEY, "session_id": sid}
    uuids: list[str] = []
    parent: str | None = None
    entries: list[Any] = []
    for i in range(n):
        u = str(uuid_mod.uuid4())
        a = str(uuid_mod.uuid4())
        entries.append(_user(f"prompt {i}", u, parent, sid))
        entries.append(_assistant(f"reply {i}", a, u, sid))
        uuids.extend([u, a])
        parent = a
    await store.append(key, entries)
    return uuids


def _raw_rows(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
    """Inspect a *closed* database file directly (the schema is documented API)."""
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


class _CountingLoadStore(SQLiteSessionStore):
    """Counts ``load()`` calls so tests can prove the summary fast path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.load_calls = 0

    async def load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        self.load_calls += 1
        return await super().load(key)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "sessions.db"


@pytest.fixture(params=["file", "memory"])
def any_db_path(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    if request.param == "memory":
        return ":memory:"
    return str(tmp_path / "sessions.db")


@pytest.fixture
def project_cwd(tmp_path: Path) -> Path:
    d = tmp_path / "project"
    d.mkdir()
    return d


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``~`` and clear auth env so resume never touches real config."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    # Clearing the auth env is what makes _copy_auth_files() consult the
    # macOS Keychain — stub it so a logged-in macOS host is never read.
    monkeypatch.setattr(
        "claude_agent_sdk._internal.session_resume._read_keychain_credentials",
        lambda: None,
    )
    return home


# ---------------------------------------------------------------------------
# Conformance harness
# ---------------------------------------------------------------------------


class TestConformance:
    async def test_file_db_passes_conformance(self, tmp_path: Path) -> None:
        counter = itertools.count()
        created: list[SQLiteSessionStore] = []

        def make() -> SQLiteSessionStore:
            # A fresh file per contract so contracts are isolated from each other.
            store = SQLiteSessionStore(tmp_path / f"conformance-{next(counter)}.db")
            created.append(store)
            return store

        await run_session_store_conformance(make)
        assert len(created) > 1
        # The harness closed every store it created (SQLite stores own a
        # connection), so each one now refuses further operations.
        for store in created:
            with pytest.raises(RuntimeError, match="SQLiteSessionStore is closed"):
                await store.load(_KEY)

    async def test_memory_db_passes_conformance(self) -> None:
        await run_session_store_conformance(lambda: SQLiteSessionStore(":memory:"))

    def test_options_validation_accepts_continue_conversation(
        self, db_path: Path
    ) -> None:
        # list_sessions() is implemented, so continue_conversation is allowed.
        validate_session_store_options(
            ClaudeAgentOptions(
                session_store=SQLiteSessionStore(db_path), continue_conversation=True
            )
        )


# ---------------------------------------------------------------------------
# list_sessions_from_store fast path (summary sidecars)
# ---------------------------------------------------------------------------


class TestSummaryFastPath:
    async def test_list_sessions_from_store_never_calls_load(
        self, db_path: Path
    ) -> None:
        async with _CountingLoadStore(db_path) as store:
            sid_a = str(uuid_mod.uuid4())
            sid_b = str(uuid_mod.uuid4())
            await _seed_chain(store, sid_a)
            await _seed_chain(store, sid_b)

            sessions = await list_sessions_from_store(store, directory=DIR)
            assert {s.session_id for s in sessions} == {sid_a, sid_b}
            for s in sessions:
                assert s.first_prompt == "prompt 0"
                assert s.summary == "prompt 0"
                assert s.created_at == _T0_MS
            # Sorted by storage mtime descending: the later seed comes first.
            mtimes = [s.last_modified for s in sessions]
            assert mtimes == sorted(mtimes, reverse=True)
            assert sessions[0].session_id == sid_b
            # Complete, fresh sidecars → zero per-session load() calls.
            assert store.load_calls == 0

    async def test_summary_mtime_never_lags_list_sessions(self, db_path: Path) -> None:
        async with _CountingLoadStore(db_path) as store:
            sid = str(uuid_mod.uuid4())
            key: SessionKey = {"project_key": PROJECT_KEY, "session_id": sid}
            await store.append(key, [_user("first", "u1", None, sid)])
            await store.append(key, [_assistant("reply", "a1", "u1", sid)])
            # Subagent rows neither touch the summary nor list_sessions().
            await store.append(
                {**key, "subpath": "subagents/agent-1"},
                [_user("sub", "s1", None, sid)],
            )

            listed = {
                e["session_id"]: e["mtime"]
                for e in await store.list_sessions(PROJECT_KEY)
            }
            summaries = {
                s["session_id"]: s["mtime"]
                for s in await store.list_session_summaries(PROJECT_KEY)
            }
            assert set(listed) == set(summaries) == {sid}
            # Same storage clock, stamped inside the same transaction: the
            # sidecar can never look stale to the fast-path freshness check.
            assert summaries[sid] >= listed[sid]

            sessions = await list_sessions_from_store(store, directory=DIR)
            assert [s.session_id for s in sessions] == [sid]
            assert sessions[0].first_prompt == "first"
            assert sessions[0].last_modified == summaries[sid]
            assert store.load_calls == 0

    async def test_rename_and_tag_are_folded_into_summaries(
        self, db_path: Path
    ) -> None:
        async with _CountingLoadStore(db_path) as store:
            sid = str(uuid_mod.uuid4())
            await _seed_chain(store, sid)
            await rename_session_via_store(store, sid, "My Title", directory=DIR)
            await tag_session_via_store(store, sid, "exp", directory=DIR)

            [info] = await list_sessions_from_store(store, directory=DIR)
            assert info.custom_title == "My Title"
            assert info.summary == "My Title"
            assert info.tag == "exp"
            assert info.first_prompt == "prompt 0"
            assert store.load_calls == 0

            # The load()-based single-session read agrees with the sidecar.
            direct = await get_session_info_from_store(store, sid, directory=DIR)
            assert direct is not None
            assert direct.custom_title == "My Title"
            assert direct.tag == "exp"
            assert store.load_calls == 1


# ---------------------------------------------------------------------------
# Public session helpers round-trip through the SQLite store
# ---------------------------------------------------------------------------


class TestSessionHelpersRoundTrip:
    async def test_get_session_messages(self, db_path: Path) -> None:
        async with SQLiteSessionStore(db_path) as store:
            sid = str(uuid_mod.uuid4())
            uuids = await _seed_chain(store, sid, n=2)
            msgs = await get_session_messages_from_store(store, sid, directory=DIR)
            assert [m.uuid for m in msgs] == uuids
            assert [m.type for m in msgs] == ["user", "assistant"] * 2
            assert msgs[0].message == {"role": "user", "content": "prompt 0"}

    async def test_fork_round_trip(self, db_path: Path) -> None:
        async with _CountingLoadStore(db_path) as store:
            sid = str(uuid_mod.uuid4())
            src_uuids = await _seed_chain(store, sid, n=2)

            result = await fork_session_via_store(
                store, sid, directory=DIR, title="Forked"
            )
            assert result.session_id != sid

            forked = await store.load(
                {"project_key": PROJECT_KEY, "session_id": result.session_id}
            )
            assert forked is not None
            msg_entries = [e for e in forked if e["type"] in ("user", "assistant")]
            assert len(msg_entries) == 4
            for e in msg_entries:
                assert e["sessionId"] == result.session_id
                assert e["uuid"] not in src_uuids
                assert e["forkedFrom"]["sessionId"] == sid
            assert forked[-1]["type"] == "custom-title"
            assert forked[-1]["customTitle"] == "Forked"

            msgs = await get_session_messages_from_store(
                store, result.session_id, directory=DIR
            )
            assert len(msgs) == 4

            # Both sessions are listed from sidecars alone; the fork is newest
            # and carries its custom title.
            store.load_calls = 0
            sessions = await list_sessions_from_store(store, directory=DIR)
            assert [s.session_id for s in sessions] == [result.session_id, sid]
            assert sessions[0].summary == "Forked"
            assert sessions[0].custom_title == "Forked"
            assert sessions[1].summary == "prompt 0"
            assert store.load_calls == 0

    async def test_delete_cascades_to_subagents_and_summaries(
        self, db_path: Path
    ) -> None:
        async with SQLiteSessionStore(db_path) as store:
            sid = str(uuid_mod.uuid4())
            other = str(uuid_mod.uuid4())
            await _seed_chain(store, sid)
            await _seed_chain(store, other)
            sub_key: SessionKey = {
                "project_key": PROJECT_KEY,
                "session_id": sid,
                "subpath": "subagents/agent-abc123",
            }
            await store.append(
                sub_key, [_user("sub", str(uuid_mod.uuid4()), None, sid)]
            )

            await delete_session_via_store(store, sid, directory=DIR)

            assert (
                await store.load({"project_key": PROJECT_KEY, "session_id": sid})
                is None
            )
            assert await store.load(sub_key) is None
            assert (
                await store.list_subkeys(
                    {"project_key": PROJECT_KEY, "session_id": sid}
                )
                == []
            )
            assert [
                s["session_id"] for s in await store.list_session_summaries(PROJECT_KEY)
            ] == [other]
            sessions = await list_sessions_from_store(store, directory=DIR)
            assert [s.session_id for s in sessions] == [other]

    async def test_subagents_round_trip(self, db_path: Path) -> None:
        async with SQLiteSessionStore(db_path) as store:
            sid = str(uuid_mod.uuid4())
            await _seed_chain(store, sid)
            sub_key: SessionKey = {
                "project_key": PROJECT_KEY,
                "session_id": sid,
                "subpath": "subagents/agent-abc123",
            }
            u = str(uuid_mod.uuid4())
            a = str(uuid_mod.uuid4())
            await store.append(
                sub_key,
                [
                    {
                        "type": "agent_metadata",
                        "agentType": "gp",
                        "toolUseId": "toolu_1",
                    },
                    _user("sub prompt", u, None, sid),
                    _assistant("sub reply", a, u, sid),
                ],
            )

            assert await list_subagents_from_store(store, sid, directory=DIR) == [
                "abc123"
            ]
            msgs = await get_subagent_messages_from_store(
                store, sid, "abc123", directory=DIR
            )
            assert [m.uuid for m in msgs] == [u, a]
            assert all(m.parent_tool_use_id == "toolu_1" for m in msgs)
            # The main transcript is untouched by the subagent rows.
            main = await get_session_messages_from_store(store, sid, directory=DIR)
            assert len(main) == 4


# ---------------------------------------------------------------------------
# Resume materialization
# ---------------------------------------------------------------------------


class TestResumeMaterialization:
    async def test_resume_writes_jsonl_and_subagent_sidecar(
        self, db_path: Path, project_cwd: Path, isolated_home: Path
    ) -> None:
        project_key = project_key_for_directory(project_cwd)
        entries: list[SessionStoreEntry] = [
            {
                "type": "user",
                "uuid": "u1",
                "message": {"role": "user", "content": "hi"},
            },
            {"type": "assistant", "uuid": "a1"},
        ]
        async with SQLiteSessionStore(db_path) as store:
            await store.append(
                {"project_key": project_key, "session_id": SESSION_ID}, entries
            )
            await store.append(
                {
                    "project_key": project_key,
                    "session_id": SESSION_ID,
                    "subpath": "subagents/agent-abc",
                },
                [
                    {"type": "agent_metadata", "toolUseId": "toolu_1"},
                    {"type": "user", "uuid": "s1"},
                ],
            )

            opts = ClaudeAgentOptions(
                cwd=project_cwd, session_store=store, resume=SESSION_ID
            )
            m = await materialize_resume_session(opts)
            assert m is not None
            try:
                assert m.resume_session_id == SESSION_ID
                project_dir = m.config_dir / "projects" / project_key
                main = project_dir / f"{SESSION_ID}.jsonl"
                lines = main.read_text(encoding="utf-8").splitlines()
                assert [json.loads(line) for line in lines] == entries

                sub = project_dir / SESSION_ID / "subagents" / "agent-abc.jsonl"
                sub_lines = sub.read_text(encoding="utf-8").splitlines()
                assert [json.loads(line) for line in sub_lines] == [
                    {"type": "user", "uuid": "s1"}
                ]
                meta = sub.with_name("agent-abc.meta.json")
                assert json.loads(meta.read_text(encoding="utf-8")) == {
                    "toolUseId": "toolu_1"
                }
            finally:
                await m.cleanup()
            assert not m.config_dir.exists()

    async def test_continue_conversation_resolves_most_recent_session(
        self, db_path: Path, project_cwd: Path, isolated_home: Path
    ) -> None:
        project_key = project_key_for_directory(project_cwd)
        async with SQLiteSessionStore(db_path) as store:
            await store.append(
                {"project_key": project_key, "session_id": SESSION_ID},
                [{"type": "user", "uuid": "u1"}],
            )
            await store.append(
                {"project_key": project_key, "session_id": SESSION_ID_2},
                [{"type": "user", "uuid": "u2"}],
            )
            opts = ClaudeAgentOptions(
                cwd=project_cwd, session_store=store, continue_conversation=True
            )
            m = await materialize_resume_session(opts)
            assert m is not None
            try:
                # Strictly increasing storage mtimes make "most recent" exact.
                assert m.resume_session_id == SESSION_ID_2
            finally:
                await m.cleanup()

    async def test_unknown_session_is_not_materialized(
        self, db_path: Path, project_cwd: Path
    ) -> None:
        async with SQLiteSessionStore(db_path) as store:
            opts = ClaudeAgentOptions(
                cwd=project_cwd, session_store=store, resume=SESSION_ID
            )
            assert await materialize_resume_session(opts) is None


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    async def test_concurrent_appends_to_one_key_keep_per_task_order(
        self, db_path: Path
    ) -> None:
        async with SQLiteSessionStore(db_path) as store:

            async def worker(task: int) -> None:
                await store.append(
                    _KEY, [{"type": "x", "task": task, "i": i} for i in range(3)]
                )

            async with anyio.create_task_group() as tg:
                for task in range(20):
                    tg.start_soon(worker, task)

            loaded = await store.load(_KEY)
            assert loaded is not None
            assert len(loaded) == 60
            for task in range(20):
                assert [e["i"] for e in loaded if e["task"] == task] == [0, 1, 2]
            # One transaction per append(): a batch's rows are never interleaved
            # with another task's rows.
            for j in range(0, 60, 3):
                assert len({e["task"] for e in loaded[j : j + 3]}) == 1
            [listed] = await store.list_sessions("proj")

        # Every append got its own storage mtime (strictly monotonic clock),
        # and list_sessions() reports the newest of them.
        [(distinct,)] = _raw_rows(
            db_path, "SELECT COUNT(DISTINCT mtime) FROM claude_session_entries"
        )
        assert distinct == 20
        [(max_mtime,)] = _raw_rows(
            db_path, "SELECT MAX(mtime) FROM claude_session_entries"
        )
        assert listed["mtime"] == max_mtime

    async def test_concurrent_appends_to_different_keys_interleave_safely(
        self, any_db_path: str
    ) -> None:
        async with SQLiteSessionStore(any_db_path) as store:

            async def worker(s: int) -> None:
                key: SessionKey = {"project_key": "proj", "session_id": f"s{s}"}
                for i in range(5):
                    await store.append(key, [{"type": "x", "s": s, "i": i}])

            async with anyio.create_task_group() as tg:
                for s in range(10):
                    tg.start_soon(worker, s)

            for s in range(10):
                loaded = await store.load(
                    {"project_key": "proj", "session_id": f"s{s}"}
                )
                assert loaded == [{"type": "x", "s": s, "i": i} for i in range(5)]
            listed = await store.list_sessions("proj")
            assert sorted(e["session_id"] for e in listed) == [
                f"s{s}" for s in range(10)
            ]
            assert len({e["mtime"] for e in listed}) == 10
            summaries = await store.list_session_summaries("proj")
            assert sorted(s["session_id"] for s in summaries) == [
                f"s{s}" for s in range(10)
            ]


# ---------------------------------------------------------------------------
# Durability and lifecycle
# ---------------------------------------------------------------------------


class TestDurabilityAndLifecycle:
    async def test_entries_and_summaries_survive_reopen(self, db_path: Path) -> None:
        sid = str(uuid_mod.uuid4())
        async with SQLiteSessionStore(db_path) as store:
            uuids = await _seed_chain(store, sid)
            before = await store.list_sessions(PROJECT_KEY)

        async with _CountingLoadStore(db_path) as reopened:
            msgs = await get_session_messages_from_store(reopened, sid, directory=DIR)
            assert [m.uuid for m in msgs] == uuids
            assert reopened.load_calls == 1
            assert await reopened.list_sessions(PROJECT_KEY) == before

            # Summary sidecars are durable too: the fast path needs no load().
            [info] = await list_sessions_from_store(reopened, directory=DIR)
            assert info.first_prompt == "prompt 0"
            assert reopened.load_calls == 1

            # The storage clock resumes strictly after the persisted mtimes.
            await reopened.append(
                {"project_key": PROJECT_KEY, "session_id": sid},
                [_user("more", "u9", None, sid)],
            )
            [after] = await reopened.list_sessions(PROJECT_KEY)
            assert after["mtime"] > before[0]["mtime"]

    async def test_aclose_is_idempotent(self, db_path: Path) -> None:
        store = SQLiteSessionStore(db_path)
        await store.append(_KEY, [{"type": "x"}])
        await store.aclose()
        await store.aclose()
        # A store that was never used closes cleanly as well.
        await SQLiteSessionStore(":memory:").aclose()

    async def test_operations_after_aclose_raise(self, db_path: Path) -> None:
        store = SQLiteSessionStore(db_path)
        await store.append(_KEY, [{"type": "x"}])
        await store.aclose()

        ops = [
            lambda: store.append(_KEY, [{"type": "x"}]),
            lambda: store.load(_KEY),
            lambda: store.list_sessions("proj"),
            lambda: store.list_session_summaries("proj"),
            lambda: store.delete(_KEY),
            lambda: store.list_subkeys(_KEY),
        ]
        for op in ops:
            with pytest.raises(RuntimeError, match="SQLiteSessionStore is closed"):
                await op()

        # The data is intact for a new store instance on the same file.
        async with SQLiteSessionStore(db_path) as fresh:
            assert await fresh.load(_KEY) == [{"type": "x"}]

    async def test_async_context_manager_closes(self, db_path: Path) -> None:
        async with SQLiteSessionStore(db_path) as store:
            assert store.path == str(db_path)
            await store.append(_KEY, [{"type": "x"}])
        with pytest.raises(RuntimeError, match="SQLiteSessionStore is closed"):
            await store.load(_KEY)

    async def test_memory_stores_are_isolated(self) -> None:
        async with (
            SQLiteSessionStore(":memory:") as a,
            SQLiteSessionStore(":memory:") as b,
        ):
            assert a.path == ":memory:"
            await a.append(_KEY, [{"type": "x"}])
            assert await a.load(_KEY) == [{"type": "x"}]
            assert await b.load(_KEY) is None

    async def test_append_empty_creates_nothing(self, any_db_path: str) -> None:
        async with SQLiteSessionStore(any_db_path) as store:
            await store.append(_KEY, [])
            assert await store.load(_KEY) is None
            assert await store.list_sessions("proj") == []
            assert await store.list_session_summaries("proj") == []
            assert await store.list_subkeys(_KEY) == []

    async def test_mtimes_are_epoch_ms_and_strictly_increasing(self) -> None:
        async with SQLiteSessionStore(":memory:") as store:
            for i in range(5):
                await store.append(
                    {"project_key": "proj", "session_id": f"s{i}"}, [{"type": "x"}]
                )
            listed = sorted(
                await store.list_sessions("proj"), key=lambda e: e["session_id"]
            )
            mtimes = [e["mtime"] for e in listed]
            assert all(isinstance(m, int) and m > 1e12 for m in mtimes)
            assert mtimes == sorted(mtimes)
            assert len(set(mtimes)) == 5
            summaries = {
                s["session_id"]: s["mtime"]
                for s in await store.list_session_summaries("proj")
            }
            assert summaries == {e["session_id"]: e["mtime"] for e in listed}

    async def test_missing_parent_directory_is_not_created(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "missing" / "sessions.db"
        with pytest.raises(sqlite3.OperationalError):
            async with SQLiteSessionStore(path):
                pass
        assert not path.parent.exists()

        store = SQLiteSessionStore(path)
        with pytest.raises(sqlite3.OperationalError):
            await store.append(_KEY, [{"type": "x"}])
        assert not path.parent.exists()
        # Once the directory exists the same instance opens lazily and works.
        path.parent.mkdir()
        await store.append(_KEY, [{"type": "x"}])
        assert await store.load(_KEY) == [{"type": "x"}]
        await store.aclose()

    async def test_unicode_entries_round_trip(self, db_path: Path) -> None:
        text = "héllo wörld 🌍 日本語 —   line separator"
        entry = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": text},
        }
        async with SQLiteSessionStore(db_path) as store:
            await store.append(_KEY, [entry])
            assert await store.load(_KEY) == [entry]
        # Stored as UTF-8 text rather than ASCII escapes (ensure_ascii=False).
        [(raw,)] = _raw_rows(db_path, "SELECT entry FROM claude_session_entries")
        assert "日本語" in raw
        assert "\\u" not in raw

    async def test_lone_surrogate_entry_round_trips(self, db_path: Path) -> None:
        # A lone surrogate cannot be stored as UTF-8 text; the store must fall
        # back to an escaped encoding rather than failing the append.
        entry = {"type": "x", "text": "bad \ud800 surrogate"}
        async with SQLiteSessionStore(db_path) as store:
            await store.append(_KEY, [entry])
            assert await store.load(_KEY) == [entry]

    async def test_non_serializable_batch_is_rejected_atomically(
        self, db_path: Path
    ) -> None:
        async with SQLiteSessionStore(db_path) as store:
            with pytest.raises(TypeError):
                await store.append(
                    _KEY, [{"type": "x", "n": 1}, {"type": "x", "blob": b"\x00"}]
                )
            # Nothing from the failed batch landed, and the store is usable.
            assert await store.load(_KEY) is None
            assert await store.list_sessions("proj") == []
            assert await store.list_session_summaries("proj") == []
            await store.append(_KEY, [{"type": "x", "n": 2}])
            assert await store.load(_KEY) == [{"type": "x", "n": 2}]

    async def test_custom_table_prefix(self, db_path: Path) -> None:
        async with SQLiteSessionStore(db_path, table_prefix="my_store") as store:
            await store.append(_KEY, [{"type": "x"}])
            assert await store.load(_KEY) == [{"type": "x"}]
        names = {
            row[0]
            for row in _raw_rows(
                db_path, "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"my_store_entries", "my_store_summaries"} <= names
        assert not any(name.startswith("claude_session") for name in names)

    async def test_delete_subpath_keeps_main_and_summary(self, db_path: Path) -> None:
        async with SQLiteSessionStore(db_path) as store:
            await store.append(_KEY, [{"type": "x", "customTitle": "T"}])
            sub: SessionKey = {**_KEY, "subpath": "subagents/agent-1"}
            await store.append(sub, [{"type": "x"}])

            await store.delete(sub)

            assert await store.load(sub) is None
            assert await store.load(_KEY) == [{"type": "x", "customTitle": "T"}]
            assert await store.list_subkeys(_KEY) == []
            assert [
                s["session_id"] for s in await store.list_session_summaries("proj")
            ] == ["sess"]

    async def test_list_subkeys_in_first_appearance_order(self, db_path: Path) -> None:
        async with SQLiteSessionStore(db_path) as store:
            await store.append(
                {**_KEY, "subpath": "subagents/agent-b"}, [{"type": "x"}]
            )
            await store.append(
                {**_KEY, "subpath": "subagents/agent-a"}, [{"type": "x"}]
            )
            await store.append(
                {**_KEY, "subpath": "subagents/agent-b"}, [{"type": "x"}]
            )
            assert await store.list_subkeys(_KEY) == [
                "subagents/agent-b",
                "subagents/agent-a",
            ]

    @pytest.mark.parametrize(
        "prefix", ["", "1abc", "bad-prefix", "x;drop", "a b", "é", 'x"y']
    )
    def test_invalid_table_prefix_rejected(self, prefix: str) -> None:
        with pytest.raises(ValueError, match="table_prefix"):
            SQLiteSessionStore(":memory:", table_prefix=prefix)

    def test_path_property_and_lazy_open(self, db_path: Path) -> None:
        assert SQLiteSessionStore(db_path).path == str(db_path)
        assert SQLiteSessionStore(str(db_path)).path == str(db_path)
        assert SQLiteSessionStore(":memory:").path == ":memory:"
        # Constructing a store does not touch the filesystem.
        assert not db_path.exists()

    def test_store_constructed_before_the_event_loop_starts(
        self, db_path: Path, anyio_backend: str
    ) -> None:
        # Typical application shape: the store is a module-level object that
        # is created before anyio.run()/asyncio.run() starts the event loop.
        store = SQLiteSessionStore(db_path)

        async def main() -> list[SessionStoreEntry] | None:
            async with store:
                await store.append(_KEY, [{"type": "x"}])
                return await store.load(_KEY)

        assert anyio.run(main, backend=anyio_backend) == [{"type": "x"}]
