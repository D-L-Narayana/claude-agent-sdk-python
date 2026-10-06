"""Replay or catch up a local on-disk session transcript into a :class:`SessionStore`.

This is the inverse of :mod:`session_resume` — where ``materialize_resume_session``
reads a store and writes a temp ``~/.claude`` tree, the functions here read the
local ``~/.claude/projects/<dir>/<sessionId>.jsonl`` (plus subagent
transcripts) and replay lines into ``store.append()``:

- :func:`import_session_to_store` appends every line (bulk migration; the
  adapter is expected to dedupe by ``uuid``).
- :func:`sync_session_to_store` first loads what the store already holds and
  appends only the missing lines — idempotent gap repair after a
  :class:`MirrorErrorMessage` reported a dropped live-mirror batch.

Mirrors the TypeScript SDK's ``importSessionToStore``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ..types import SessionKey, SessionStore, SessionStoreEntry
from .sessions import (
    _read_agent_metadata_sidecar,
    _resolve_session_file_path,
    _split_agent_metadata,
    _validate_uuid,
)
from .transcript_mirror_batcher import MAX_PENDING_BYTES, MAX_PENDING_ENTRIES

__all__ = ["SessionSyncReport", "import_session_to_store", "sync_session_to_store"]

logger = logging.getLogger(__name__)


@dataclass
class SessionSyncReport:
    """Outcome of :func:`sync_session_to_store`.

    ``appended``/``skipped`` count main-transcript entries; ``subagents`` maps
    each subagent ``subpath`` to its own ``(appended, skipped)`` pair, where
    the synthetic ``agent_metadata`` entry derived from the ``.meta.json``
    sidecar is counted too. A run that found nothing to do reports
    ``appended == 0`` everywhere.
    """

    session_id: str
    project_key: str
    appended: int
    skipped: int
    subagents: dict[str, tuple[int, int]] = field(default_factory=dict)


async def import_session_to_store(
    session_id: str,
    store: SessionStore,
    *,
    directory: str | None = None,
    include_subagents: bool = True,
    batch_size: int = MAX_PENDING_ENTRIES,
) -> None:
    """Replay a local session transcript into a :class:`SessionStore`.

    Streams the on-disk JSONL line-by-line and calls ``store.append(key, batch)``
    every ``batch_size`` entries (or 1 MiB of line bytes, whichever comes
    first). Useful for migrating existing local sessions to a remote store.
    Every line is appended, so adapters should treat ``entry["uuid"]`` as an
    idempotency key for re-import to be duplicate-safe; to catch a store up
    after a :class:`MirrorErrorMessage` without relying on adapter-side
    dedup, use :func:`sync_session_to_store` instead.

    Lines that are blank are skipped silently; lines that are not valid JSON
    objects (e.g. a torn write from a crash mid-append) are skipped with a
    ``logger.warning`` naming the file and line number, so one corrupt line
    cannot abort the import.

    The destination ``project_key`` is the name of the on-disk project
    directory the session file was found in — the same key
    :func:`file_path_to_session_key` (and thus ``TranscriptMirrorBatcher``)
    would have produced for the same file — so an imported session is
    indistinguishable from a live-mirrored one and resumable via
    ``query(options=ClaudeAgentOptions(session_store=store, resume=session_id))``
    from the original ``cwd``.

    Args:
        session_id: UUID of the session to import.
        store: Destination :class:`SessionStore`.
        directory: Project directory path (same semantics as
            :func:`list_sessions`). When omitted, all project directories are
            searched for the session file.
        include_subagents: If ``True`` (default), also import subagent
            transcripts under ``<sessionId>/subagents/**`` and their
            ``.meta.json`` sidecars.
        batch_size: Maximum entries per ``store.append()`` call. Default 500.

    Raises:
        ValueError: If ``session_id`` is not a valid UUID.
        FileNotFoundError: If the session JSONL cannot be found on disk.
    """
    resolved, project_key = _resolve_session(session_id, directory)
    batch_size = _effective_batch_size(batch_size)

    main_key: SessionKey = {"project_key": project_key, "session_id": session_id}
    await _append_jsonl_file_in_batches(resolved, main_key, store, batch_size)

    if not include_subagents:
        return

    for file_path, _subpath, sub_key in _iter_subagent_transcripts(
        resolved, session_id, project_key
    ):
        await _append_jsonl_file_in_batches(file_path, sub_key, store, batch_size)

        # The on-disk .jsonl does NOT contain agent_metadata entries — those
        # are only sent to live mirrors and persisted in the .meta.json
        # sidecar. Import the sidecar so materialize_resume_session() can
        # recreate it and resumed subagents keep their agentType/worktreePath.
        meta_entry = _read_agent_metadata_entry(file_path)
        if meta_entry is not None:
            await store.append(sub_key, [meta_entry])


async def sync_session_to_store(
    session_id: str,
    store: SessionStore,
    *,
    directory: str | None = None,
    include_subagents: bool = True,
    batch_size: int = MAX_PENDING_ENTRIES,
) -> SessionSyncReport:
    """Catch a :class:`SessionStore` up with the local on-disk session.

    Idempotent counterpart of :func:`import_session_to_store` for repairing a
    live-mirror gap: after a :class:`MirrorErrorMessage` (a batch the
    ``TranscriptMirrorBatcher`` dropped once its retry budget was exhausted)
    call this with the session id and the store only receives the entries it
    is missing. Running it again on an unchanged session appends nothing.

    Session resolution, ``project_key`` derivation, subagent ``subpath``
    derivation, batching and error behavior are identical to
    :func:`import_session_to_store`. For the main transcript and each
    subagent transcript the store's current entries are loaded once via
    ``store.load(key)`` (``None`` is treated as empty), then the on-disk lines
    are walked in file order:

    - an entry whose ``uuid`` is already present in the store is skipped;
    - an entry without a ``uuid`` (titles, tags, mode markers, ...) is
      skipped only when a deep-equal entry is already present — each stored
      entry accounts for at most one on-disk line, so repeated identical
      markers (``plan`` → ``default`` → ``plan``) are reproduced faithfully;
    - everything else is appended, in file order, in batches of
      ``batch_size`` entries or 1 MiB of line bytes.

    Entries present on disk but not in the store therefore land *after*
    whatever the store already holds. The SDK's store readers rebuild the
    conversation from ``parentUuid`` links, so storage order does not have to
    match disk order.

    A subagent's ``.meta.json`` sidecar is appended as a synthetic
    ``{"type": "agent_metadata", ...}`` entry only when it differs from the
    last ``agent_metadata`` entry already stored for that subpath (readers
    take the last one, so a changed sidecar wins without rewriting history).

    Blank lines are skipped silently; lines that are not valid JSON objects
    are skipped with a ``logger.warning`` naming the file and line number.
    Store exceptions propagate — because the sync is idempotent, simply run
    it again after the store recovers.

    Args:
        session_id: UUID of the session to sync.
        store: Destination :class:`SessionStore`. Must implement ``load``.
        directory: Project directory path (same semantics as
            :func:`list_sessions`). When omitted, all project directories are
            searched for the session file.
        include_subagents: If ``True`` (default), also sync subagent
            transcripts under ``<sessionId>/subagents/**`` and their
            ``.meta.json`` sidecars.
        batch_size: Maximum entries per ``store.append()`` call. Default 500.

    Returns:
        A :class:`SessionSyncReport` with appended/skipped counts for the main
        transcript and per subagent subpath.

    Raises:
        ValueError: If ``session_id`` is not a valid UUID.
        FileNotFoundError: If the session JSONL cannot be found on disk.
    """
    resolved, project_key = _resolve_session(session_id, directory)
    batch_size = _effective_batch_size(batch_size)

    main_key: SessionKey = {"project_key": project_key, "session_id": session_id}
    existing = await store.load(main_key) or []
    appended, skipped = await _sync_jsonl_file(
        resolved, main_key, store, batch_size, existing
    )
    report = SessionSyncReport(
        session_id=session_id,
        project_key=project_key,
        appended=appended,
        skipped=skipped,
    )

    if not include_subagents:
        return report

    for file_path, subpath, sub_key in _iter_subagent_transcripts(
        resolved, session_id, project_key
    ):
        existing = await store.load(sub_key) or []
        sub_appended, sub_skipped = await _sync_jsonl_file(
            file_path, sub_key, store, batch_size, existing
        )

        meta_entry = _read_agent_metadata_entry(file_path)
        if meta_entry is not None:
            # Readers use the LAST agent_metadata entry, so only a sidecar
            # that differs from it needs to be (re-)appended.
            last_meta, _transcript = _split_agent_metadata(existing)
            if last_meta == meta_entry:
                sub_skipped += 1
            else:
                await store.append(sub_key, [meta_entry])
                sub_appended += 1

        report.subagents[subpath] = (sub_appended, sub_skipped)

    return report


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_session(session_id: str, directory: str | None) -> tuple[Path, str]:
    """Validate ``session_id`` and locate its JSONL file.

    Returns ``(path, project_key)``. The key is the on-disk project directory
    name — matches ``file_path_to_session_key()`` / ``TranscriptMirrorBatcher``
    even when the resolver's search (``directory=None``) or worktree fallback
    found the file somewhere other than ``directory``.
    """
    if not _validate_uuid(session_id):
        raise ValueError(f"Invalid session_id: {session_id}")

    resolved = _resolve_session_file_path(session_id, directory)
    if resolved is None:
        raise FileNotFoundError(f"Session {session_id} not found")

    return resolved, resolved.parent.name


def _effective_batch_size(batch_size: int) -> int:
    return batch_size if batch_size > 0 else MAX_PENDING_ENTRIES


def _iter_subagent_transcripts(
    resolved: Path, session_id: str, project_key: str
) -> Iterator[tuple[Path, str, SessionKey]]:
    """Yield ``(file_path, subpath, key)`` for each subagent transcript.

    Subagent transcripts live at ``<projectDir>/<sessionId>/subagents/**``.
    ``subpath`` is the path relative to the session directory, '/'-joined,
    sans ``.jsonl`` — e.g. ``subagents/agent-abc`` or
    ``subagents/workflows/run-1/agent-def`` — matching
    ``file_path_to_session_key()`` so ``list_subkeys()`` and
    ``get_subagent_messages_from_store()`` round-trip.
    """
    session_dir = resolved.with_suffix("")
    for file_path in _collect_jsonl_files(session_dir / "subagents"):
        rel_parts = list(file_path.relative_to(session_dir).parts)
        rel_parts[-1] = rel_parts[-1][: -len(".jsonl")]
        subpath = "/".join(rel_parts)
        key: SessionKey = {
            "project_key": project_key,
            "session_id": session_id,
            "subpath": subpath,
        }
        yield file_path, subpath, key


def _read_agent_metadata_entry(transcript_path: Path) -> SessionStoreEntry | None:
    """The ``.meta.json`` sidecar beside a subagent transcript as the synthetic
    ``agent_metadata`` store entry, or ``None`` when the sidecar is missing,
    corrupt or not a JSON object (treated as absent; other read errors
    propagate)."""
    meta = _read_agent_metadata_sidecar(transcript_path)
    if meta is None:
        return None
    # Synthetic discriminator last so a stray "type" key in the CLI-owned
    # sidecar can never shadow it.
    return cast(SessionStoreEntry, {**meta, "type": "agent_metadata"})


def _iter_jsonl_entries(file_path: Path) -> Iterator[tuple[SessionStoreEntry, int]]:
    """Yield ``(entry, line_length)`` for each JSON-object line of a JSONL file.

    Blank lines are skipped silently. Lines that are not valid JSON, or whose
    JSON value is not an object, are skipped with a warning naming the file
    and 1-based line number (parity with ``_parse_transcript_entries``), so a
    single corrupt line — e.g. a torn write from a crash mid-append — cannot
    abort an import or sync.
    """
    with file_path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as e:  # json.JSONDecodeError is a ValueError
                logger.warning(
                    "Skipping malformed transcript line %d in %s: %s",
                    lineno,
                    file_path,
                    e,
                )
                continue
            if not isinstance(obj, dict):
                logger.warning(
                    "Skipping malformed transcript line %d in %s: expected a "
                    "JSON object, got %s",
                    lineno,
                    file_path,
                    type(obj).__name__,
                )
                continue
            yield cast(SessionStoreEntry, obj), len(line)


async def _append_jsonl_file_in_batches(
    file_path: Path,
    key: SessionKey,
    store: SessionStore,
    batch_size: int,
) -> None:
    """Stream-read a JSONL file, flushing to ``store.append()`` in batches of
    ``batch_size`` entries (or ``MAX_PENDING_BYTES`` of line text, whichever
    comes first). Blank and malformed lines are skipped (see
    :func:`_iter_jsonl_entries`)."""
    batch: list[SessionStoreEntry] = []
    nbytes = 0
    for entry, size in _iter_jsonl_entries(file_path):
        batch.append(entry)
        nbytes += size
        if len(batch) >= batch_size or nbytes >= MAX_PENDING_BYTES:
            await store.append(key, batch)
            batch = []
            nbytes = 0
    if batch:
        await store.append(key, batch)


async def _sync_jsonl_file(
    file_path: Path,
    key: SessionKey,
    store: SessionStore,
    batch_size: int,
    existing: list[Any],
) -> tuple[int, int]:
    """Append the lines of ``file_path`` that ``existing`` (the store's current
    entries for ``key``) does not already contain. Returns
    ``(appended, skipped)``.

    Matching is decided against ``existing`` as loaded at the start of the
    run — entries with a ``uuid`` by uuid, uuid-less entries by deep equality
    with each stored entry matching at most one on-disk line — so syncing
    into an empty store writes exactly what :func:`import_session_to_store`
    would.
    """
    existing_uuids: set[str] = set()
    # uuid-less stored entries not yet matched by an on-disk line.
    unmatched: list[dict[str, Any]] = []
    for raw in existing:
        if not isinstance(raw, dict):
            continue
        uuid = raw.get("uuid")
        if isinstance(uuid, str):
            existing_uuids.add(uuid)
        else:
            unmatched.append(raw)

    appended = 0
    skipped = 0
    batch: list[SessionStoreEntry] = []
    nbytes = 0
    for entry, size in _iter_jsonl_entries(file_path):
        uuid = entry.get("uuid")
        if isinstance(uuid, str):
            if uuid in existing_uuids:
                skipped += 1
                continue
        else:
            match = next((i for i, e in enumerate(unmatched) if e == entry), -1)
            if match >= 0:
                del unmatched[match]
                skipped += 1
                continue
        batch.append(entry)
        nbytes += size
        appended += 1
        if len(batch) >= batch_size or nbytes >= MAX_PENDING_BYTES:
            await store.append(key, batch)
            batch = []
            nbytes = 0
    if batch:
        await store.append(key, batch)
    return appended, skipped


def _collect_jsonl_files(base_dir: Path) -> Iterator[Path]:
    """Recursively yield all ``*.jsonl`` file paths under ``base_dir``.

    Yields nothing if ``base_dir`` does not exist. Sorted per directory so
    import order is deterministic across platforms.
    """
    try:
        dirents = sorted(base_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return
    for entry in dirents:
        if entry.is_dir():
            yield from _collect_jsonl_files(entry)
        elif entry.is_file() and entry.name.endswith(".jsonl"):
            yield entry
