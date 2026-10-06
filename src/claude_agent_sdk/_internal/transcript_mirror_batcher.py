"""Batching layer between ``transcript_mirror`` stdout frames and a SessionStore.

The CLI subprocess emits ``{"type": "transcript_mirror", "filePath": ..., "entries": [...]}``
frames interleaved with normal SDK messages. The receive loop peels these off
and hands them to :class:`TranscriptMirrorBatcher.enqueue`, which accumulates
them and flushes to :meth:`SessionStore.append` either when a ``result``
message arrives (explicit flush) or when the pending buffer exceeds size
thresholds (eager background flush). This keeps adapter latency off the
hot path during model streaming. :attr:`TranscriptMirrorBatcher.stats`
exposes lifetime counters (frames/entries enqueued, flushes, entries
flushed, batches failed and entries dropped) as an immutable
:class:`MirrorStats` snapshot.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import anyio

from ..types import SessionKey, SessionStore, SessionStoreEntry
from ._task_compat import TaskHandle, spawn_detached
from .session_store import file_path_to_session_key

logger = logging.getLogger(__name__)

# Eager-flush thresholds. Exported for tests.
MAX_PENDING_ENTRIES = 500
MAX_PENDING_BYTES = 1 << 20  # 1 MiB
SEND_TIMEOUT_SECONDS = 60.0

# Bounded retry for transient adapter failures. Backoff list length must be
# MAX_ATTEMPTS - 1 (one delay between each pair of attempts).
MIRROR_APPEND_MAX_ATTEMPTS = 3
MIRROR_APPEND_BACKOFF_S = (0.2, 0.8)


@dataclass
class _MirrorEntry:
    file_path: str
    entries: list[SessionStoreEntry]
    bytes: int


@dataclass(frozen=True)
class MirrorStats:
    """Point-in-time snapshot of :attr:`TranscriptMirrorBatcher.stats`.

    Immutable: read the property again for fresh numbers. A non-zero
    ``batches_failed``/``entries_dropped`` means the store is missing entries
    (each drop was also reported via ``on_error`` as a ``MirrorErrorMessage``)
    and ``sync_session_to_store()`` can repair the gap. Frames whose file
    path cannot be mapped to a :class:`SessionKey` are dropped with a warning
    and counted in neither ``entries_flushed`` nor ``entries_dropped``.
    """

    frames_enqueued: int = 0
    """``transcript_mirror`` frames passed to ``enqueue()``."""
    entries_enqueued: int = 0
    """Transcript entries contained in those frames."""
    flushes: int = 0
    """Drains (explicit ``flush()``/``close()`` or eager) that had frames."""
    entries_flushed: int = 0
    """Entries delivered by a successful ``store.append()``."""
    batches_failed: int = 0
    """Per-key batches dropped once the retry/timeout budget was exhausted."""
    entries_dropped: int = 0
    """Entries in those dropped batches (each is reported via ``on_error``)."""


@dataclass
class TranscriptMirrorBatcher:
    """Accumulates ``transcript_mirror`` frames and flushes them to a store.

    ``enqueue`` is fire-and-forget; ``flush`` is async. The pending queue is
    bounded — when it exceeds ``max_pending_entries`` or ``max_pending_bytes``
    an eager flush fires in the background so memory stays flat during long
    turns where no ``result`` (and thus no explicit ``flush()``) arrives.

    Adapter failures are retried (``MIRROR_APPEND_MAX_ATTEMPTS`` attempts
    total) with short backoff; timeouts are not retried since the in-flight
    call may still land. Only after the final attempt fails is the batch
    dropped and reported via ``on_error``. Failures never raise — the
    local-disk transcript is already durable so the session must continue
    unaffected. Adapters should dedupe by ``entry["uuid"]`` when present
    (some entry types lack a uuid) since a retried batch may partially
    overlap a prior partial write.
    """

    store: SessionStore
    projects_dir: str
    on_error: Callable[[SessionKey | None, str], Awaitable[None]]
    send_timeout: float = SEND_TIMEOUT_SECONDS
    max_pending_entries: int = MAX_PENDING_ENTRIES
    max_pending_bytes: int = MAX_PENDING_BYTES

    _pending: list[_MirrorEntry] = field(default_factory=list)
    _pending_entries: int = 0
    _pending_bytes: int = 0
    _flush_task: TaskHandle | None = None
    _lock: anyio.Lock = field(default_factory=anyio.Lock)

    # Lifetime counters behind the ``stats`` property. Only ever mutated
    # synchronously (no await between read and write), so concurrent drains
    # on one event loop cannot lose updates.
    _frames_enqueued: int = field(default=0, init=False, repr=False)
    _entries_enqueued: int = field(default=0, init=False, repr=False)
    _flushes: int = field(default=0, init=False, repr=False)
    _entries_flushed: int = field(default=0, init=False, repr=False)
    _batches_failed: int = field(default=0, init=False, repr=False)
    _entries_dropped: int = field(default=0, init=False, repr=False)

    @property
    def stats(self) -> MirrorStats:
        """Snapshot of the batcher's lifetime counters.

        Every read returns a new frozen :class:`MirrorStats`; a snapshot is
        never updated by later activity. Cheap enough to poll after each
        ``ResultMessage`` or at ``close()`` to decide whether a
        ``sync_session_to_store()`` catch-up is warranted.
        """
        return MirrorStats(
            frames_enqueued=self._frames_enqueued,
            entries_enqueued=self._entries_enqueued,
            flushes=self._flushes,
            entries_flushed=self._entries_flushed,
            batches_failed=self._batches_failed,
            entries_dropped=self._entries_dropped,
        )

    def enqueue(self, file_path: str, entries: list[SessionStoreEntry]) -> None:
        """Buffer a frame; schedule an eager flush if thresholds are exceeded."""
        self._frames_enqueued += 1
        self._entries_enqueued += len(entries)
        # Approximate wire size — one stringify per frame (not per entry) keeps
        # this cheap relative to the json.loads the transport already did.
        size = len(json.dumps(entries))
        self._pending.append(_MirrorEntry(file_path, entries, size))
        self._pending_entries += len(entries)
        self._pending_bytes += size
        if (
            self._pending_entries > self.max_pending_entries
            or self._pending_bytes > self.max_pending_bytes
        ):
            # Fire-and-forget on the current backend via the SDK's sniffio-
            # dispatched spawner; the lock in _drain() serializes against any
            # in-flight flush so append ordering holds. _drain() is contracted
            # never to raise; spawn_detached logs if that contract is ever
            # violated (parity with asyncio's unretrieved-exception warning).
            self._flush_task = spawn_detached(self._drain())

    async def flush(self) -> None:
        """Flush all pending entries, serialized after any in-flight eager flush."""
        await self._drain()

    async def close(self) -> None:
        """Final flush before teardown. Never raises.

        Shielded so the final batch still reaches the store when ``close()``
        runs under a cancelled scope (client disconnect / Ctrl+C at
        ``__aexit__``).
        """
        try:
            with anyio.CancelScope(shield=True):
                await self.flush()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"[TranscriptMirrorBatcher] close flush failed: {e}")

    async def _drain(self) -> None:
        """Detach the pending buffer, await any prior flush, then send.

        Detaching happens before acquiring the lock so ``enqueue`` can keep
        accumulating into a fresh buffer while a prior flush is in flight.
        Never raises — adapter and ``on_error`` callback errors are caught
        and logged.
        """
        items = self._pending
        self._pending = []
        self._pending_entries = 0
        self._pending_bytes = 0
        errors: list[tuple[SessionKey, str]] = []
        async with self._lock:
            if not items:
                return
            self._flushes += 1
            try:
                await self._do_flush(items, errors)
            except Exception as e:  # pragma: no cover - defensive
                # _do_flush already wraps store.append; this guards any
                # remaining unguarded path so the "Never raises" contract
                # holds against future regressions.
                logger.error("[TranscriptMirrorBatcher] _do_flush raised: %s", e)
                return
        # Report errors after releasing the lock so a slow on_error callback
        # cannot block subsequent drains (which only need the lock for
        # append-ordering).
        for key, msg in errors:
            try:
                await self.on_error(key, msg)
            except Exception as cb_err:  # pragma: no cover - defensive
                logger.error(
                    "[TranscriptMirrorBatcher] on_error callback raised: %s",
                    cb_err,
                )

    async def _do_flush(
        self, items: list[_MirrorEntry], errors: list[tuple[SessionKey, str]]
    ) -> None:
        # Coalesce by file_path so each unique file gets one append per flush
        # instead of one per enqueued frame. dict preserves first-seen order;
        # entries within a path keep enqueue order.
        by_path: dict[str, list[SessionStoreEntry]] = {}
        for item in items:
            bucket = by_path.get(item.file_path)
            if bucket is not None:
                bucket.extend(item.entries)
            else:
                by_path[item.file_path] = list(item.entries)

        for file_path, entries in by_path.items():
            if not entries:
                # Avoid creating phantom keys in adapters that touch storage
                # on append([]) — nothing to write.
                continue
            key = file_path_to_session_key(file_path, self.projects_dir)
            if key is None:
                logger.warning(
                    "[SessionStore] dropping mirror frame: filePath %s is not "
                    "under %s -- subprocess CLAUDE_CONFIG_DIR likely differs "
                    "from parent (custom env / container?)",
                    file_path,
                    self.projects_dir,
                )
                continue
            last_err: Exception | None = None
            succeeded = False
            for attempt in range(MIRROR_APPEND_MAX_ATTEMPTS):
                if attempt > 0:
                    await anyio.sleep(MIRROR_APPEND_BACKOFF_S[attempt - 1])
                try:
                    with anyio.fail_after(self.send_timeout):
                        await self.store.append(key, entries)
                    succeeded = True
                    self._entries_flushed += len(entries)
                    break
                except TimeoutError as e:
                    # Don't retry on timeout: the cancel scope cancels the
                    # task but cancellation is best-effort for adapters
                    # wrapping non-cancellable I/O, so the in-flight call may
                    # still land — a retry would launch a concurrent
                    # duplicate. Also keeps worst-case lock hold at
                    # ~send_timeout rather than ~3×send_timeout + backoff.
                    last_err = e
                    logger.debug(
                        "[TranscriptMirrorBatcher] append timed out after "
                        "%.1fs for %s — not retrying",
                        self.send_timeout,
                        file_path,
                    )
                    break
                except Exception as e:  # noqa: BLE001 - adapter is user code
                    last_err = e
                    logger.debug(
                        "[TranscriptMirrorBatcher] append attempt %d/%d failed "
                        "for %s: %s",
                        attempt + 1,
                        MIRROR_APPEND_MAX_ATTEMPTS,
                        file_path,
                        e,
                    )
            if not succeeded:
                self._batches_failed += 1
                self._entries_dropped += len(entries)
                logger.error(
                    "[TranscriptMirrorBatcher] flush failed for %s: %s",
                    file_path,
                    last_err,
                )
                errors.append((key, str(last_err)))
