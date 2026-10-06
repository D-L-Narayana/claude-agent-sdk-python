"""Test utilities for SDK extension authors.

Importing this subpackage does not require ``pytest`` — assertions use plain
``assert`` so the harness works under any test runner.

:func:`run_session_store_conformance` creates one store per behavioral
contract and, when a store defines ``aclose()``, calls it once that store is
done with (awaiting the result when it is awaitable), so adapters that own
connections — such as :class:`claude_agent_sdk.stores.SQLiteSessionStore` —
do not leak resources across the run. Stores without ``aclose`` are unaffected.
"""

from .session_store_conformance import run_session_store_conformance

__all__ = ["run_session_store_conformance"]
