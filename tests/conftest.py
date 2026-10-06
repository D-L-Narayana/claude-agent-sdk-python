"""Pytest configuration: run every ``@pytest.mark.anyio`` test under both
asyncio and trio, so CI catches backend-specific regressions in either."""

import pytest


@pytest.fixture(params=["asyncio", "trio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(autouse=True)
def _clear_cli_version_cache():  # type: ignore[no-untyped-def]
    """Keep the transport's per-binary CLI version cache from leaking across tests.

    The subprocess transport caches the ``claude -v`` probe per binary so repeated
    connects do not respawn it; tests that mock the probe must each start from an
    empty cache.
    """
    try:
        from claude_agent_sdk._internal.transport.subprocess_cli import (
            clear_cli_version_cache,
        )
    except ImportError:  # pragma: no cover - transport without the cache
        yield
        return
    clear_cli_version_cache()
    yield
    clear_cli_version_cache()
