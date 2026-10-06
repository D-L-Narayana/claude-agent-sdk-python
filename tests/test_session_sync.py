"""Tests for ``sync_session_to_store`` (idempotent local JSONL → SessionStore
catch-up after a live-mirror gap)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from claude_agent_sdk import (
    InMemorySessionStore,
    import_session_to_store,
    project_key_for_directory,
)
from claude_agent_sdk._internal.session_import import (
    SessionSyncReport,
    sync_session_to_store,
)
from claude_agent_sdk._internal.sessions import _split_agent_metadata
from claude_agent_sdk.types import SessionKey, SessionStoreEntry

SESSION_ID = "550e8400-e29b-41d4-a716-446655440001"
SUBPATH = "subagents/agent-abc"
META: dict[str, Any] = {"agentType": "coder", "worktreePath": "/tmp/wt"}
META_ENTRY = cast(SessionStoreEntry, {"type": "agent_metadata", **META})

_LOGGER = "claude_agent_sdk._internal.session_import"


@pytest.fixture
def cwd(tmp_path: Path) -> Path:
    d = tmp_path / "project"
    d.mkdir()
    return d


@pytest.fixture
def project_key(cwd: Path) -> str:
    return project_key_for_directory(cwd)


@pytest.fixture
def claude_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_key: str
) -> Path:
    """Create an isolated ~/.claude/projects/<project_key>/ tree."""
    config = tmp_path / "claude_config"
    project_dir = config / "projects" / project_key
    project_dir.mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    return project_dir


def _entry(i: int) -> SessionStoreEntry:
    return {"type": "user", "uuid": f"u{i}", "timestamp": f"2026-01-01T00:00:{i:02d}Z"}


def _tag(name: str) -> SessionStoreEntry:
    """A uuid-less entry, like the CLI's tag / title / mode marker lines."""
    return cast(SessionStoreEntry, {"type": "tag", "tag": name})


def _mode(mode: str) -> SessionStoreEntry:
    return cast(SessionStoreEntry, {"type": "permission-mode", "mode": mode})


def _write_jsonl(path: Path, entries: list[SessionStoreEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def _write_session(claude_dir: Path, *, main: int = 7, sub: int = 2) -> None:
    """Main transcript with ``main`` entries, one subagent transcript with
    ``sub`` entries and a ``.meta.json`` sidecar."""
    _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", [_entry(i) for i in range(main)])
    sub_dir = claude_dir / SESSION_ID / "subagents"
    _write_jsonl(sub_dir / "agent-abc.jsonl", [_entry(10 + i) for i in range(sub)])
    (sub_dir / "agent-abc.meta.json").write_text(json.dumps(META), encoding="utf-8")


def _main_key(project_key: str) -> SessionKey:
    return {"project_key": project_key, "session_id": SESSION_ID}


def _sub_key(project_key: str, subpath: str = SUBPATH) -> SessionKey:
    return {"project_key": project_key, "session_id": SESSION_ID, "subpath": subpath}


def _uuids(entries: list[SessionStoreEntry]) -> set[str]:
    return {e["uuid"] for e in entries if isinstance(e.get("uuid"), str)}


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """WARNING records emitted by the session_import module logger."""
    return [
        r for r in caplog.records if r.name == _LOGGER and r.levelno == logging.WARNING
    ]


async def _snapshot(store: InMemorySessionStore, project_key: str) -> dict[str, Any]:
    subkeys = sorted(await store.list_subkeys(_main_key(project_key)))
    return {
        "main": store.get_entries(_main_key(project_key)),
        "subkeys": subkeys,
        "subs": {sk: store.get_entries(_sub_key(project_key, sk)) for sk in subkeys},
    }


# ---------------------------------------------------------------------------
# Sync into an empty store — parity with import_session_to_store
# ---------------------------------------------------------------------------


class TestSyncIntoEmptyStore:
    @pytest.mark.anyio
    async def test_matches_import_and_reports_counts(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_session(claude_dir)

        imported = InMemorySessionStore()
        await import_session_to_store(SESSION_ID, imported, directory=str(cwd))

        synced = InMemorySessionStore()
        report = await sync_session_to_store(SESSION_ID, synced, directory=str(cwd))

        assert isinstance(report, SessionSyncReport)
        assert report.session_id == SESSION_ID
        assert report.project_key == project_key
        assert (report.appended, report.skipped) == (7, 0)
        # 2 transcript entries + 1 agent_metadata entry for the sidecar.
        assert report.subagents == {SUBPATH: (3, 0)}

        assert await _snapshot(synced, project_key) == await _snapshot(
            imported, project_key
        )
        assert synced.get_entries(_main_key(project_key)) == [
            _entry(i) for i in range(7)
        ]
        assert synced.get_entries(_sub_key(project_key)) == [
            _entry(10),
            _entry(11),
            META_ENTRY,
        ]

    @pytest.mark.anyio
    async def test_batches_appends_by_batch_size(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        entries = [_entry(i) for i in range(5)]
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", entries)

        store = InMemorySessionStore()
        spy = AsyncMock(wraps=store.append)
        store.append = spy  # type: ignore[method-assign]

        report = await sync_session_to_store(
            SESSION_ID, store, directory=str(cwd), batch_size=2
        )

        assert (report.appended, report.skipped) == (5, 0)
        assert spy.await_count == 3  # 2 + 2 + 1
        key = _main_key(project_key)
        assert spy.await_args_list[0].args == (key, entries[0:2])
        assert spy.await_args_list[1].args == (key, entries[2:4])
        assert spy.await_args_list[2].args == (key, entries[4:5])

    @pytest.mark.anyio
    async def test_nonpositive_batch_size_uses_default(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        entries = [_entry(i) for i in range(3)]
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", entries)

        store = InMemorySessionStore()
        spy = AsyncMock(wraps=store.append)
        store.append = spy  # type: ignore[method-assign]

        report = await sync_session_to_store(
            SESSION_ID, store, directory=str(cwd), batch_size=0
        )

        assert report.appended == 3
        assert spy.await_count == 1
        assert store.get_entries(_main_key(project_key)) == entries

    @pytest.mark.anyio
    async def test_directory_none_keys_from_resolved_path_not_cwd(
        self,
        claude_dir: Path,
        project_key: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Same key derivation as import: the project_key comes from the
        directory the file was *found* in, not the process cwd."""
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", [_entry(0)])
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert project_key_for_directory(None) != project_key  # precondition

        store = InMemorySessionStore()
        report = await sync_session_to_store(SESSION_ID, store, directory=None)

        assert report.project_key == project_key
        assert store.get_entries(_main_key(project_key)) == [_entry(0)]


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


class TestIdempotence:
    @pytest.mark.anyio
    async def test_second_sync_appends_nothing_and_store_is_unchanged(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_session(claude_dir)
        store = InMemorySessionStore()

        first = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))
        assert first.appended == 7
        before = await _snapshot(store, project_key)

        second = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert (second.appended, second.skipped) == (0, 7)
        assert second.subagents == {SUBPATH: (0, 3)}
        assert await _snapshot(store, project_key) == before

    @pytest.mark.anyio
    async def test_sync_after_import_is_a_noop(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_session(claude_dir)
        store = InMemorySessionStore()
        await import_session_to_store(SESSION_ID, store, directory=str(cwd))
        before = await _snapshot(store, project_key)

        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert (report.appended, report.skipped) == (0, 7)
        assert report.subagents == {SUBPATH: (0, 3)}
        assert await _snapshot(store, project_key) == before


# ---------------------------------------------------------------------------
# Gap repair (simulated dropped mirror batch)
# ---------------------------------------------------------------------------


class TestGapRepair:
    @pytest.mark.anyio
    async def test_fills_gap_from_dropped_batch_in_the_middle(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        entries = [_entry(i) for i in range(7)]
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", entries)

        store = InMemorySessionStore()
        key = _main_key(project_key)
        # The live mirror delivered 0-2 and 5-6; the batch with 3-4 was dropped.
        await store.append(key, entries[0:3])
        await store.append(key, entries[5:7])

        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert (report.appended, report.skipped) == (2, 5)
        stored = store.get_entries(key)
        assert _uuids(stored) == {f"u{i}" for i in range(7)}
        # Previously-missing entries land at the end, in file order.
        assert stored == entries[0:3] + entries[5:7] + entries[3:5]

    @pytest.mark.anyio
    async def test_gap_entries_appended_in_file_order_in_batches(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        entries = [_entry(i) for i in range(8)]
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", entries)

        store = InMemorySessionStore()
        key = _main_key(project_key)
        await store.append(key, [entries[0], entries[7]])
        spy = AsyncMock(wraps=store.append)
        store.append = spy  # type: ignore[method-assign]

        report = await sync_session_to_store(
            SESSION_ID, store, directory=str(cwd), batch_size=2
        )

        assert (report.appended, report.skipped) == (6, 2)
        assert [c.args[1] for c in spy.await_args_list] == [
            entries[1:3],
            entries[3:5],
            entries[5:7],
        ]

    @pytest.mark.anyio
    async def test_in_file_duplicate_uuid_matches_import_and_never_duplicates(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        """Matching is against the store's content at the start of the run:
        into an empty store a repeated uuid lands exactly as import would
        write it; once the store holds that uuid, no further copy is added."""
        first = _entry(0)
        twin = cast(SessionStoreEntry, {**first, "timestamp": "2026-01-01T00:00:59Z"})
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", [first, twin, _entry(1)])
        key = _main_key(project_key)

        empty = InMemorySessionStore()
        report = await sync_session_to_store(SESSION_ID, empty, directory=str(cwd))
        assert (report.appended, report.skipped) == (3, 0)
        imported = InMemorySessionStore()
        await import_session_to_store(SESSION_ID, imported, directory=str(cwd))
        assert empty.get_entries(key) == imported.get_entries(key)

        seeded = InMemorySessionStore()
        await seeded.append(key, [twin])
        report = await sync_session_to_store(SESSION_ID, seeded, directory=str(cwd))
        assert (report.appended, report.skipped) == (1, 2)
        assert seeded.get_entries(key) == [twin, _entry(1)]


# ---------------------------------------------------------------------------
# uuid-less entries dedupe by deep equality
# ---------------------------------------------------------------------------


class TestUuidlessEntries:
    @pytest.mark.anyio
    async def test_deduped_by_deep_equality_and_resynced_only_when_absent(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        on_disk = [_entry(0), _tag("alpha"), _entry(1), _tag("beta")]
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", on_disk)

        store = InMemorySessionStore()
        key = _main_key(project_key)
        # Key order differs from the on-disk line (e.g. JSONB reordering) —
        # parsed objects are compared, so this is still "already present".
        reordered_alpha = cast(SessionStoreEntry, {"tag": "alpha", "type": "tag"})
        await store.append(key, [_entry(0), reordered_alpha])

        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert (report.appended, report.skipped) == (2, 2)
        assert store.get_entries(key) == [
            _entry(0),
            _tag("alpha"),
            _entry(1),
            _tag("beta"),
        ]

        second = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))
        assert (second.appended, second.skipped) == (0, 4)

    @pytest.mark.anyio
    async def test_uuidless_entry_that_differs_is_resynced(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", [_entry(0), _tag("alpha")])

        store = InMemorySessionStore()
        key = _main_key(project_key)
        similar = cast(SessionStoreEntry, {"type": "tag", "tag": "alpha", "extra": 1})
        await store.append(key, [_entry(0), similar])

        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert (report.appended, report.skipped) == (1, 1)
        assert store.get_entries(key) == [_entry(0), similar, _tag("alpha")]

    @pytest.mark.anyio
    async def test_repeated_uuidless_lines_are_matched_by_count(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        """Mode toggles write identical uuid-less lines (plan → default →
        plan). Each on-disk occurrence is matched against at most one stored
        entry, so syncing into an empty store equals import, a partial store
        gets exactly the missing occurrence, and the final state (last line
        wins) is preserved."""
        plan, default = _mode("plan"), _mode("default")
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", [plan, default, plan])
        key = _main_key(project_key)

        empty = InMemorySessionStore()
        report = await sync_session_to_store(SESSION_ID, empty, directory=str(cwd))
        assert (report.appended, report.skipped) == (3, 0)
        assert empty.get_entries(key) == [plan, default, plan]

        partial = InMemorySessionStore()
        await partial.append(key, [plan, default])
        report = await sync_session_to_store(SESSION_ID, partial, directory=str(cwd))
        assert (report.appended, report.skipped) == (1, 2)
        assert partial.get_entries(key) == [plan, default, plan]

        again = await sync_session_to_store(SESSION_ID, partial, directory=str(cwd))
        assert (again.appended, again.skipped) == (0, 3)
        assert partial.get_entries(key) == [plan, default, plan]


# ---------------------------------------------------------------------------
# Subagents and the .meta.json sidecar
# ---------------------------------------------------------------------------


class TestSubagents:
    @pytest.mark.anyio
    async def test_first_sync_appends_transcript_and_one_agent_metadata(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_session(claude_dir, main=1, sub=2)

        store = InMemorySessionStore()
        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert report.subagents == {SUBPATH: (3, 0)}
        assert store.get_entries(_sub_key(project_key)) == [
            _entry(10),
            _entry(11),
            META_ENTRY,
        ]
        assert await store.list_subkeys(_main_key(project_key)) == [SUBPATH]

    @pytest.mark.anyio
    async def test_second_sync_appends_nothing_for_subagent(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_session(claude_dir, main=1, sub=2)
        store = InMemorySessionStore()
        await sync_session_to_store(SESSION_ID, store, directory=str(cwd))
        before = store.get_entries(_sub_key(project_key))

        second = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert second.subagents == {SUBPATH: (0, 3)}
        assert store.get_entries(_sub_key(project_key)) == before

    @pytest.mark.anyio
    async def test_changed_sidecar_appends_new_agent_metadata(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_session(claude_dir, main=1, sub=2)
        store = InMemorySessionStore()
        await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        new_meta = {**META, "worktreePath": "/tmp/wt-moved"}
        (claude_dir / SESSION_ID / "subagents" / "agent-abc.meta.json").write_text(
            json.dumps(new_meta), encoding="utf-8"
        )

        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert report.subagents == {SUBPATH: (1, 2)}
        stored = store.get_entries(_sub_key(project_key))
        assert stored == [
            _entry(10),
            _entry(11),
            META_ENTRY,
            {"type": "agent_metadata", **new_meta},
        ]
        # Readers take the LAST agent_metadata entry, so the new sidecar wins.
        assert _split_agent_metadata(stored)[0] == {
            "type": "agent_metadata",
            **new_meta,
        }

        third = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))
        assert third.subagents == {SUBPATH: (0, 3)}

    @pytest.mark.anyio
    async def test_subagent_gap_is_filled_without_duplicating_metadata(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_session(claude_dir, main=1, sub=2)
        store = InMemorySessionStore()
        sub_key = _sub_key(project_key)
        # Live mirror delivered the first entry and the metadata; the second
        # transcript entry was dropped.
        await store.append(sub_key, [_entry(10), META_ENTRY])

        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert report.subagents == {SUBPATH: (1, 2)}
        assert store.get_entries(sub_key) == [_entry(10), META_ENTRY, _entry(11)]

    @pytest.mark.anyio
    async def test_sidecar_type_key_cannot_shadow_agent_metadata_marker(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", [_entry(0)])
        sub_dir = claude_dir / SESSION_ID / "subagents"
        _write_jsonl(sub_dir / "agent-abc.jsonl", [_entry(10)])
        (sub_dir / "agent-abc.meta.json").write_text(
            json.dumps({"type": "something-else", "toolUseId": "toolu_1"}),
            encoding="utf-8",
        )

        store = InMemorySessionStore()
        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert report.subagents == {SUBPATH: (2, 0)}
        assert store.get_entries(_sub_key(project_key))[1] == {
            "type": "agent_metadata",
            "toolUseId": "toolu_1",
        }

    @pytest.mark.anyio
    @pytest.mark.parametrize("sidecar", ["not json {", "[1, 2]", "42"])
    async def test_unusable_sidecar_is_treated_as_absent(
        self, claude_dir: Path, cwd: Path, project_key: str, sidecar: str
    ) -> None:
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", [_entry(0)])
        sub_dir = claude_dir / SESSION_ID / "subagents"
        _write_jsonl(sub_dir / "agent-abc.jsonl", [_entry(10)])
        (sub_dir / "agent-abc.meta.json").write_text(sidecar, encoding="utf-8")

        store = InMemorySessionStore()
        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert report.subagents == {SUBPATH: (1, 0)}
        assert store.get_entries(_sub_key(project_key)) == [_entry(10)]

    @pytest.mark.anyio
    async def test_nested_subagent_subpath(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", [_entry(0)])
        nested = claude_dir / SESSION_ID / "subagents" / "workflows" / "run-1"
        _write_jsonl(nested / "agent-def.jsonl", [_entry(20)])

        store = InMemorySessionStore()
        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        subpath = "subagents/workflows/run-1/agent-def"
        assert report.subagents == {subpath: (1, 0)}
        assert store.get_entries(_sub_key(project_key, subpath)) == [_entry(20)]

    @pytest.mark.anyio
    async def test_include_subagents_false_skips_subagents(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_session(claude_dir, main=2, sub=2)

        store = InMemorySessionStore()
        report = await sync_session_to_store(
            SESSION_ID, store, directory=str(cwd), include_subagents=False
        )

        assert (report.appended, report.skipped) == (2, 0)
        assert report.subagents == {}
        assert await store.list_subkeys(_main_key(project_key)) == []

    @pytest.mark.anyio
    async def test_no_subagents_dir_is_noop(
        self, claude_dir: Path, cwd: Path, project_key: str
    ) -> None:
        _write_jsonl(claude_dir / f"{SESSION_ID}.jsonl", [_entry(0)])

        store = InMemorySessionStore()
        report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert report.subagents == {}
        assert store.get_entries(_main_key(project_key)) == [_entry(0)]


# ---------------------------------------------------------------------------
# Validation / errors
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.anyio
    async def test_invalid_uuid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid session_id"):
            await sync_session_to_store("../../etc/passwd", InMemorySessionStore())

    @pytest.mark.anyio
    async def test_session_not_found_raises(self, claude_dir: Path, cwd: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            await sync_session_to_store(
                SESSION_ID, InMemorySessionStore(), directory=str(cwd)
            )


# ---------------------------------------------------------------------------
# Corrupt JSONL lines
# ---------------------------------------------------------------------------


class TestCorruptLines:
    @pytest.mark.anyio
    @pytest.mark.parametrize("bad_line", ["not json {", "[1, 2]", "42", '"text"'])
    async def test_corrupt_line_is_skipped_with_warning_and_rest_syncs(
        self,
        claude_dir: Path,
        cwd: Path,
        project_key: str,
        caplog: pytest.LogCaptureFixture,
        bad_line: str,
    ) -> None:
        path = claude_dir / f"{SESSION_ID}.jsonl"
        path.write_text(
            json.dumps(_entry(0))
            + "\n"
            + bad_line
            + "\n"
            + json.dumps(_entry(1))
            + "\n",
            encoding="utf-8",
        )

        store = InMemorySessionStore()
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            report = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert (report.appended, report.skipped) == (2, 0)
        assert store.get_entries(_main_key(project_key)) == [_entry(0), _entry(1)]

        warnings = _warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert str(path) in message
        assert "line 2" in message

    @pytest.mark.anyio
    async def test_corrupt_line_is_not_counted_and_resync_is_noop(
        self,
        claude_dir: Path,
        cwd: Path,
        project_key: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The corrupt line is neither appended nor skipped-counted, and a
        re-sync of the unchanged file warns again but appends nothing."""
        path = claude_dir / f"{SESSION_ID}.jsonl"
        path.write_text(
            json.dumps(_entry(0)) + "\n{truncated\n" + json.dumps(_entry(1)) + "\n",
            encoding="utf-8",
        )
        store = InMemorySessionStore()
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            first = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))
        assert (first.appended, first.skipped) == (2, 0)
        assert len(_warnings(caplog)) == 1
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            second = await sync_session_to_store(SESSION_ID, store, directory=str(cwd))

        assert (second.appended, second.skipped) == (0, 2)
        assert store.get_entries(_main_key(project_key)) == [_entry(0), _entry(1)]
        assert len(_warnings(caplog)) == 1
