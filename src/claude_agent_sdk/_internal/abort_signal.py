"""Cooperative cancellation signal for hook and permission callbacks.

Claude Code may abandon a request it made to the SDK — a permission prompt
or hook that was still running when the user interrupted the turn — by
sending ``control_cancel_request``. The SDK cancels the handler task, but a
callback that does blocking or external work (asking a human, calling a
service, holding a lock) may want to notice sooner, or clean up in a
controlled way. Each inbound control request gets an :class:`AbortSignal`
for that purpose: it is handed to ``can_use_tool`` as
``ToolPermissionContext.signal`` and to hook callbacks as
``context["signal"]``, and it is aborted when the CLI cancels the request
or when the query closes.

The module depends only on anyio, so it works under asyncio and trio alike.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import anyio
import anyio.lowlevel

logger = logging.getLogger(__name__)

__all__ = ["AbortSignal"]


class AbortSignal:
    """Cooperative cancellation signal delivered to callbacks.

    Delivered as ``context["signal"]`` to hook callbacks and as
    ``ToolPermissionContext.signal`` to ``can_use_tool``. Set when the CLI
    abandons the request (``control_cancel_request``) or the Query closes.

    A signal is aborted at most once; the first reason sticks. Check
    :attr:`aborted` between steps of a long-running callback, ``await``
    :meth:`wait` to block until the request is abandoned, or register an
    :meth:`on_abort` callback to be told synchronously::

        async def can_use_tool(tool_name, tool_input, context):
            # Stop polling the approval service once Claude Code moved on.
            while not context.signal.aborted:
                decision = await approval_service.poll(tool_name, tool_input)
                if decision is not None:
                    return decision
                await anyio.sleep(1)
            return PermissionResultDeny(message=context.signal.reason or "aborted")

    The SDK also cancels the handler task when a request is abandoned, so a
    callback that awaits normally is interrupted with the backend's
    cancellation exception shortly after the signal is aborted; the signal
    lets it find out *why* (``reason``) and react before that happens.

    Instances are created by the SDK (one per inbound control request) and
    aborted by it; the ``_abort`` method is not part of the public API.
    """

    def __init__(self) -> None:
        self._aborted = False
        self._reason: str | None = None
        # Created on first wait() so a signal can be built outside an event
        # loop and is only ever bound to the loop that waits on it.
        self._event: anyio.Event | None = None
        self._callbacks: list[Callable[[AbortSignal], None]] = []

    @property
    def aborted(self) -> bool:
        """Whether the request this signal belongs to has been abandoned."""
        return self._aborted

    @property
    def reason(self) -> str | None:
        """Why the signal was aborted (``"cancelled by Claude Code"``,
        ``"query closed"``), or ``None`` while it is not aborted."""
        return self._reason

    async def wait(self) -> None:
        """Wait until the signal is aborted.

        Returns as soon as :attr:`aborted` is true (immediately if it already
        is). Cancellable like any other await: cancelling the waiter does not
        abort the signal. Works under asyncio and trio.
        """
        if not self._aborted:
            if self._event is None:
                self._event = anyio.Event()
            await self._event.wait()
        # A checkpoint, as anyio's Event.wait() guarantees one: an abort is
        # normally followed at once by the SDK cancelling the handler task,
        # and this lets that cancellation be delivered here on every backend
        # rather than at some later await in the callback.
        await anyio.lowlevel.checkpoint()

    def on_abort(self, callback: Callable[[AbortSignal], None]) -> None:
        """Register ``callback(signal)`` to run when the signal is aborted.

        Runs synchronously inside the abort, in registration order, and at
        most once. If the signal is already aborted the callback runs
        immediately. A callback that raises is logged and does not stop the
        others or propagate to the SDK.
        """
        if self._aborted:
            self._invoke(callback)
        else:
            self._callbacks.append(callback)

    def _abort(self, reason: str | None = None) -> None:
        """Abort the signal (SDK-internal). Idempotent: the first reason wins."""
        if self._aborted:
            return
        self._aborted = True
        self._reason = reason
        if self._event is not None:
            self._event.set()
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            self._invoke(callback)

    def _invoke(self, callback: Callable[[AbortSignal], None]) -> None:
        try:
            callback(self)
        except Exception:
            logger.exception("AbortSignal on_abort callback %r raised", callback)

    def __repr__(self) -> str:
        return f"AbortSignal(aborted={self._aborted}, reason={self._reason!r})"
