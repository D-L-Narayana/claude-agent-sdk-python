"""SQLite-backed :class:`~claude_agent_sdk.SessionStore` built on the standard library.

:class:`SQLiteSessionStore` keeps every mirrored transcript in one SQLite
database file — no server, no client library, nothing to install — and
implements all optional :class:`~claude_agent_sdk.SessionStore` methods,
including the incremental summary sidecars that let
:func:`claude_agent_sdk.list_sessions_from_store` answer from a single query
instead of loading every session. It passes
:func:`claude_agent_sdk.testing.run_session_store_conformance`.

Usage::

    from claude_agent_sdk import ClaudeAgentOptions, query
    from claude_agent_sdk.stores import SQLiteSessionStore

    async with SQLiteSessionStore("sessions.db") as store:
        async for message in query(
            prompt="Hello!",
            options=ClaudeAgentOptions(session_store=store),
        ):
            ...  # transcript entries are mirrored into sessions.db

Schema (``table_prefix`` defaults to ``claude_session``)::

    CREATE TABLE claude_session_entries (
      project_key TEXT    NOT NULL,
      session_id  TEXT    NOT NULL,
      subpath     TEXT    NOT NULL DEFAULT '',  -- '' marks the main transcript
      seq         INTEGER PRIMARY KEY AUTOINCREMENT,
      entry       TEXT    NOT NULL,             -- one JSON transcript line
      mtime       INTEGER NOT NULL              -- storage write time, epoch ms
    );
    CREATE INDEX claude_session_entries_key_idx
      ON claude_session_entries (project_key, session_id, subpath, seq);
    CREATE TABLE claude_session_summaries (
      project_key TEXT    NOT NULL,
      session_id  TEXT    NOT NULL,
      mtime       INTEGER NOT NULL,             -- same clock as the entries
      data        TEXT    NOT NULL,             -- opaque fold_session_summary state
      PRIMARY KEY (project_key, session_id)
    );

Operational notes
-----------------

**Paths.** Relative paths resolve against the working directory at open time.
Parent directories are *not* created implicitly — opening
``"missing/dir/sessions.db"`` raises :class:`sqlite3.OperationalError`.
``":memory:"`` gives a private in-memory database that lives exactly as long
as the store instance.

**Durability.** File databases run in WAL mode with ``synchronous=NORMAL``:
committed appends survive process crashes; only an operating-system crash or
power loss can roll back the very last commits. SQLite keeps ``<path>-wal``
and ``<path>-shm`` next to the database while the store is open.

**Concurrency.** Operations are serialized through one lock and run on a
worker thread, so the event loop is never blocked and the store behaves the
same on asyncio and trio. Any number of tasks — and ``ClaudeSDKClient``
instances — in one process can share a store. A single writing process per
database file is recommended: other processes may read concurrently, and
concurrent writers are serialized by SQLite's file locks (waiting up to five
seconds for a busy database), but each process stamps writes with its own
clock, so a session written from several processes can occasionally look
stale to the summary fast path and be re-derived from its entries instead.

**Retention.** Nothing is deleted automatically. Expire sessions whose last
write is older than a cutoff (Unix epoch milliseconds) with a scheduled sweep
such as::

    DELETE FROM claude_session_entries
     WHERE session_id IN (SELECT session_id FROM claude_session_entries
                          GROUP BY project_key, session_id
                          HAVING MAX(mtime) < :cutoff_ms);
    DELETE FROM claude_session_summaries WHERE mtime < :cutoff_ms;

followed by ``VACUUM`` if the file should shrink. Local-disk transcripts under
``CLAUDE_CONFIG_DIR`` are swept independently by the CLI's
``cleanupPeriodDays`` setting.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import anyio

from .._internal.session_summary import fold_session_summary
from ..types import (
    SessionKey,
    SessionListSubkeysKey,
    SessionStore,
    SessionStoreEntry,
    SessionStoreListEntry,
    SessionSummaryEntry,
)

__all__ = ["SQLiteSessionStore"]

_T = TypeVar("_T")

#: Table names are interpolated into SQL (identifiers cannot be parameterized),
#: so the prefix must be a plain identifier.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_CLOSED_MESSAGE = "SQLiteSessionStore is closed"


def _dumps(obj: Any) -> str:
    """Serialize ``obj`` compactly as UTF-8-storable JSON text.

    Transcripts produced by a JavaScript runtime can contain lone surrogates,
    which SQLite cannot store as UTF-8 ``TEXT``; those fall back to ``\\uXXXX``
    escapes that :func:`json.loads` turns back into the identical string.
    """
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    if not text.isascii():
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            text = json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
    return text


@contextlib.contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Run the body as one write transaction; roll back on any failure.

    ``BEGIN IMMEDIATE`` takes the write lock up front, so the read-fold-write
    of a summary sidecar can never interleave with another writer.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise


class SQLiteSessionStore(SessionStore):
    """Durable :class:`~claude_agent_sdk.SessionStore` backed by one SQLite database.

    Args:
        path: Database file path, or ``":memory:"`` for a private in-memory
            database. The file is created on first use; its parent directories
            are not. See the module documentation for path, durability,
            concurrency and retention notes.
        table_prefix: Prefix of the two tables, ``<prefix>_entries`` and
            ``<prefix>_summaries``. Must match ``[A-Za-z_][A-Za-z0-9_]*``
            because it is interpolated into SQL identifiers; anything else
            raises :class:`ValueError`.

    The connection is opened lazily by the first operation (eagerly by
    ``async with``), so constructing a store never touches the filesystem.
    Release it with :meth:`aclose` or by using the store as an async context
    manager; afterwards every operation raises :class:`RuntimeError`.
    ``aclose()`` may be called any number of times.

    Storage write times (``mtime``) come from one strictly increasing epoch-ms
    clock per store instance, so back-to-back appends never share an mtime,
    :meth:`list_sessions` reflects the latest append, and a summary sidecar —
    stamped in the same transaction as its entries — is never older than the
    session it describes. On open the clock resumes after the newest mtime
    already persisted in the database.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        table_prefix: str = "claude_session",
    ) -> None:
        if not _IDENT_RE.fullmatch(table_prefix):
            raise ValueError(
                f"table_prefix {table_prefix!r} must match [A-Za-z_][A-Za-z0-9_]* "
                "(it is interpolated into SQL identifiers)"
            )
        self._path = os.fspath(path)
        self._entries = f"{table_prefix}_entries"
        self._summaries = f"{table_prefix}_summaries"
        self._conn: sqlite3.Connection | None = None
        self._closed = False
        self._lock = anyio.Lock()
        self._last_mtime = 0

    @property
    def path(self) -> str:
        """Database path as given to the constructor (``":memory:"`` included)."""
        return self._path

    # ------------------------------------------------------------------
    # SessionStore protocol
    # ------------------------------------------------------------------

    async def append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        self._ensure_not_closed()
        if not entries:
            # No rows, no summary, no phantom key.
            return
        project_key = key["project_key"]
        session_id = key["session_id"]
        subpath = key.get("subpath") or ""

        def write(conn: sqlite3.Connection) -> None:
            # Serialize before touching the database so a batch that cannot be
            # encoded (TypeError) is rejected whole and leaves nothing behind.
            texts = [_dumps(entry) for entry in entries]
            mtime = self._next_mtime()
            with _transaction(conn):
                conn.executemany(
                    f"INSERT INTO {self._entries}"
                    " (project_key, session_id, subpath, entry, mtime)"
                    " VALUES (?, ?, ?, ?, ?)",
                    [(project_key, session_id, subpath, text, mtime) for text in texts],
                )
                if not subpath:
                    # Fold the summary sidecar inside the same transaction and
                    # stamp it with the same storage mtime as the rows, so the
                    # list_sessions_from_store() fast path never sees a stale
                    # sidecar. Subagent transcripts do not contribute.
                    row = conn.execute(
                        f"SELECT mtime, data FROM {self._summaries}"
                        " WHERE project_key = ? AND session_id = ?",
                        (project_key, session_id),
                    ).fetchone()
                    prev: SessionSummaryEntry | None = None
                    if row is not None:
                        prev = {
                            "session_id": session_id,
                            "mtime": int(row[0]),
                            "data": json.loads(row[1]),
                        }
                    folded = fold_session_summary(prev, key, entries)
                    folded["mtime"] = mtime
                    conn.execute(
                        f"INSERT OR REPLACE INTO {self._summaries}"
                        " (project_key, session_id, mtime, data) VALUES (?, ?, ?, ?)",
                        (project_key, session_id, mtime, _dumps(folded["data"])),
                    )

        await self._run(write)

    async def load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        project_key = key["project_key"]
        session_id = key["session_id"]
        subpath = key.get("subpath") or ""

        def read(conn: sqlite3.Connection) -> list[SessionStoreEntry] | None:
            rows = conn.execute(
                f"SELECT entry FROM {self._entries}"
                " WHERE project_key = ? AND session_id = ? AND subpath = ?"
                " ORDER BY seq",
                (project_key, session_id, subpath),
            ).fetchall()
            if not rows:
                return None
            return [json.loads(row[0]) for row in rows]

        return await self._run(read)

    async def list_sessions(self, project_key: str) -> list[SessionStoreListEntry]:
        def read(conn: sqlite3.Connection) -> list[SessionStoreListEntry]:
            rows = conn.execute(
                f"SELECT session_id, MAX(mtime) FROM {self._entries}"
                " WHERE project_key = ? AND subpath = ''"
                " GROUP BY session_id",
                (project_key,),
            ).fetchall()
            return [
                {"session_id": session_id, "mtime": int(mtime)}
                for session_id, mtime in rows
            ]

        return await self._run(read)

    async def list_session_summaries(
        self, project_key: str
    ) -> list[SessionSummaryEntry]:
        def read(conn: sqlite3.Connection) -> list[SessionSummaryEntry]:
            rows = conn.execute(
                f"SELECT session_id, mtime, data FROM {self._summaries}"
                " WHERE project_key = ?",
                (project_key,),
            ).fetchall()
            return [
                {
                    "session_id": session_id,
                    "mtime": int(mtime),
                    "data": json.loads(data),
                }
                for session_id, mtime, data in rows
            ]

        return await self._run(read)

    async def delete(self, key: SessionKey) -> None:
        project_key = key["project_key"]
        session_id = key["session_id"]
        subpath = key.get("subpath") or ""

        def remove(conn: sqlite3.Connection) -> None:
            with _transaction(conn):
                if subpath:
                    # Targeted: only this subagent transcript.
                    conn.execute(
                        f"DELETE FROM {self._entries}"
                        " WHERE project_key = ? AND session_id = ? AND subpath = ?",
                        (project_key, session_id, subpath),
                    )
                    return
                # Cascade: main transcript, every subpath and the summary.
                conn.execute(
                    f"DELETE FROM {self._entries}"
                    " WHERE project_key = ? AND session_id = ?",
                    (project_key, session_id),
                )
                conn.execute(
                    f"DELETE FROM {self._summaries}"
                    " WHERE project_key = ? AND session_id = ?",
                    (project_key, session_id),
                )

        await self._run(remove)

    async def list_subkeys(self, key: SessionListSubkeysKey) -> list[str]:
        project_key = key["project_key"]
        session_id = key["session_id"]

        def read(conn: sqlite3.Connection) -> list[str]:
            rows = conn.execute(
                f"SELECT subpath FROM {self._entries}"
                " WHERE project_key = ? AND session_id = ? AND subpath <> ''"
                " GROUP BY subpath ORDER BY MIN(seq)",
                (project_key, session_id),
            ).fetchall()
            return [row[0] for row in rows]

        return await self._run(read)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the database connection.

        Idempotent. Every later operation raises
        ``RuntimeError("SQLiteSessionStore is closed")``. Waits for an
        in-flight operation to finish and is shielded from cancellation so
        the connection is released even while a task group is unwinding.
        """
        if self._closed:
            return
        self._closed = True
        with anyio.CancelScope(shield=True):
            async with self._lock:
                conn, self._conn = self._conn, None
                if conn is not None:
                    await anyio.to_thread.run_sync(conn.close)

    async def __aenter__(self) -> SQLiteSessionStore:
        # Open eagerly so a bad path fails at the ``async with`` line rather
        # than at the first append in the middle of a session.
        await self._run(lambda conn: None)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError(_CLOSED_MESSAGE)

    async def _run(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run ``fn(conn)`` on a worker thread with the store lock held."""
        self._ensure_not_closed()
        async with self._lock:
            # aclose() may have won the race for the lock.
            self._ensure_not_closed()

            def call() -> _T:
                return fn(self._connection())

            return await anyio.to_thread.run_sync(call)

    def _connection(self) -> sqlite3.Connection:
        """Return the connection, opening the database on first use.

        Runs on the worker thread with the lock held. A failed open (for
        example a missing parent directory) leaves the store usable, so the
        next operation retries.
        """
        conn = self._conn
        if conn is not None:
            return conn
        conn = sqlite3.connect(
            self._path, check_same_thread=False, isolation_level=None
        )
        try:
            if self._path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._entries} ("
                " project_key TEXT NOT NULL,"
                " session_id TEXT NOT NULL,"
                " subpath TEXT NOT NULL DEFAULT '',"
                " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
                " entry TEXT NOT NULL,"
                " mtime INTEGER NOT NULL)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._entries}_key_idx"
                f" ON {self._entries} (project_key, session_id, subpath, seq)"
            )
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._summaries} ("
                " project_key TEXT NOT NULL,"
                " session_id TEXT NOT NULL,"
                " mtime INTEGER NOT NULL,"
                " data TEXT NOT NULL,"
                " PRIMARY KEY (project_key, session_id))"
            )
            # Resume the storage clock strictly after anything already
            # persisted, so mtimes stay monotonic across close/reopen.
            self._last_mtime = max(self._last_mtime, self._persisted_clock(conn))
        except BaseException:
            conn.close()
            raise
        self._conn = conn
        return conn

    def _persisted_clock(self, conn: sqlite3.Connection) -> int:
        """Newest mtime already in the database (0 when empty).

        The latest row by ``seq`` carries the newest entry mtime (an O(log n)
        rowid lookup rather than a scan); the summaries table is small.
        """
        row = conn.execute(
            f"SELECT mtime FROM {self._entries} ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        latest_entry = int(row[0]) if row is not None else 0
        row = conn.execute(f"SELECT MAX(mtime) FROM {self._summaries}").fetchone()
        latest_summary = int(row[0]) if row is not None and row[0] is not None else 0
        return max(latest_entry, latest_summary)

    def _next_mtime(self) -> int:
        """Storage write time in Unix epoch ms, strictly increasing per instance.

        Same rule as ``InMemorySessionStore``: back-to-back appends within one
        millisecond still get distinct, ordered mtimes. Called with the store
        lock held.
        """
        now_ms = int(time.time() * 1000)
        if now_ms <= self._last_mtime:
            now_ms = self._last_mtime + 1
        self._last_mtime = now_ms
        return now_ms
