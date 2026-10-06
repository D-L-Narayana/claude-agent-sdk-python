"""Durable, dependency-free :class:`~claude_agent_sdk.SessionStore` implementations
shipped with the SDK.

Every store in this package is built on the Python standard library only (no
third-party database clients), so it can be used wherever the SDK itself runs:

- :class:`SQLiteSessionStore` — single-file store on the stdlib ``sqlite3``
  module. Passes :func:`claude_agent_sdk.testing.run_session_store_conformance`
  and maintains the per-session summary sidecars that make
  :func:`claude_agent_sdk.list_sessions_from_store` a single query.

The in-memory reference store (:class:`claude_agent_sdk.InMemorySessionStore`)
lives in the root package; the S3/Redis/Postgres reference adapters under
``examples/session_stores/`` need their respective client libraries and are
not shipped as part of the SDK.
"""

from .sqlite import SQLiteSessionStore

__all__ = ["SQLiteSessionStore"]
