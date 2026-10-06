"""Tests for the cooperative AbortSignal handed to hook and permission callbacks.

Every async test here runs under both asyncio and trio (``anyio_backend`` in
conftest.py); the synchronous ones need no event loop at all.
"""

import logging

import anyio
import pytest

from claude_agent_sdk._internal.abort_signal import AbortSignal

LOGGER_NAME = "claude_agent_sdk._internal.abort_signal"


class TestAbortSignalState:
    def test_defaults(self):
        signal = AbortSignal()
        assert signal.aborted is False
        assert signal.reason is None

    def test_abort_sets_state_and_reason(self):
        signal = AbortSignal()
        signal._abort("cancelled by Claude Code")
        assert signal.aborted is True
        assert signal.reason == "cancelled by Claude Code"

    def test_abort_without_reason(self):
        signal = AbortSignal()
        signal._abort()
        assert signal.aborted is True
        assert signal.reason is None

    def test_abort_is_idempotent_and_keeps_the_first_reason(self):
        signal = AbortSignal()
        signal._abort("first")
        signal._abort("second")
        signal._abort()
        assert signal.aborted is True
        assert signal.reason == "first"

    def test_repr_shows_state(self):
        signal = AbortSignal()
        assert repr(signal) == "AbortSignal(aborted=False, reason=None)"
        signal._abort("query closed")
        assert repr(signal) == "AbortSignal(aborted=True, reason='query closed')"


class TestAbortSignalWait:
    @pytest.mark.anyio
    async def test_wait_returns_once_aborted(self):
        signal = AbortSignal()
        seen: list[str | None] = []

        async def waiter():
            await signal.wait()
            seen.append(signal.reason)

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(waiter)
                # Let the waiter block on the signal before aborting it.
                await anyio.wait_all_tasks_blocked()
                assert seen == []
                signal._abort("stop")
        assert seen == ["stop"]

    @pytest.mark.anyio
    async def test_wait_returns_immediately_when_already_aborted(self):
        signal = AbortSignal()
        signal._abort("done")
        with anyio.fail_after(1):
            await signal.wait()
            # Aborting is permanent: a second wait returns as well.
            await signal.wait()
        assert signal.reason == "done"

    @pytest.mark.anyio
    async def test_wait_is_cancellable(self):
        signal = AbortSignal()
        with anyio.move_on_after(0.05) as scope:
            await signal.wait()
        assert scope.cancelled_caught
        # Cancelling the waiter does not abort the signal.
        assert signal.aborted is False
        assert signal.reason is None

    @pytest.mark.anyio
    async def test_all_waiters_are_released(self):
        signal = AbortSignal()
        released: list[str] = []

        async def waiter(name: str):
            await signal.wait()
            released.append(name)

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(waiter, "a")
                tg.start_soon(waiter, "b")
                await anyio.wait_all_tasks_blocked()
                signal._abort("stop")
        assert sorted(released) == ["a", "b"]

    @pytest.mark.anyio
    async def test_waiter_can_still_wait_after_a_cancelled_wait(self):
        """A cancelled ``wait()`` leaves the signal usable for a later wait."""
        signal = AbortSignal()
        with anyio.move_on_after(0.01):
            await signal.wait()
        signal._abort("stop")
        with anyio.fail_after(1):
            await signal.wait()


class TestAbortSignalCallbacks:
    def test_on_abort_fires_later_with_the_signal(self):
        signal = AbortSignal()
        fired: list[tuple[AbortSignal, str | None]] = []
        signal.on_abort(lambda s: fired.append((s, s.reason)))
        assert fired == []
        signal._abort("stop")
        assert fired == [(signal, "stop")]

    def test_on_abort_fires_immediately_if_already_aborted(self):
        signal = AbortSignal()
        signal._abort("stop")
        fired: list[str | None] = []
        signal.on_abort(lambda s: fired.append(s.reason))
        assert fired == ["stop"]

    def test_callbacks_fire_once_in_registration_order(self):
        signal = AbortSignal()
        order: list[str] = []
        signal.on_abort(lambda s: order.append("first"))
        signal.on_abort(lambda s: order.append("second"))
        signal._abort("stop")
        signal._abort("again")
        assert order == ["first", "second"]

    def test_raising_callback_is_logged_and_does_not_break_others(self, caplog):
        signal = AbortSignal()
        fired: list[str] = []

        def bad(_signal: AbortSignal) -> None:
            raise RuntimeError("boom in on_abort")

        signal.on_abort(bad)
        signal.on_abort(lambda s: fired.append("ok"))
        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            signal._abort("stop")  # must not raise
        assert fired == ["ok"]
        assert signal.aborted is True
        assert signal.reason == "stop"
        assert any(r.levelno == logging.ERROR for r in caplog.records)
        assert "boom in on_abort" in caplog.text

    def test_raising_late_callback_does_not_propagate(self, caplog):
        signal = AbortSignal()
        signal._abort()

        def bad(_signal: AbortSignal) -> None:
            raise RuntimeError("boom in late on_abort")

        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            signal.on_abort(bad)  # fires immediately; must not raise
        assert "boom in late on_abort" in caplog.text

    @pytest.mark.anyio
    async def test_callback_runs_before_waiters_resume(self):
        """``on_abort`` callbacks run synchronously inside ``_abort``, so a
        waiter woken by the same abort already sees their side effects."""
        signal = AbortSignal()
        events: list[str] = []
        signal.on_abort(lambda s: events.append("callback"))

        async def waiter():
            await signal.wait()
            events.append("waiter")

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(waiter)
                await anyio.wait_all_tasks_blocked()
                signal._abort("stop")
                events.append("after_abort")
        assert events == ["callback", "after_abort", "waiter"]
