"""Internal client implementation."""

import json
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from typing import Any

from ..types import ClaudeAgentOptions, Message, _configure_can_use_tool
from .message_parser import parse_message
from .options_validation import validate_options
from .query import Query, stamp_user_message
from .query_setup import attach_session_store, query_kwargs_for
from .session_resume import (
    MaterializedResume,
    apply_materialized_options,
    materialize_resume_session,
)
from .session_store_validation import validate_session_store_options
from .transport import Transport
from .transport.subprocess_cli import SubprocessCLITransport


class InternalClient:
    """Internal client implementation."""

    def __init__(self) -> None:
        """Initialize the internal client."""

    async def process_query(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        options: ClaudeAgentOptions,
        transport: Transport | None = None,
    ) -> AsyncIterator[Message]:
        """Process a query through transport and Query.

        Raises:
            ValueError: If ``options`` combine settings the SDK cannot honor
                (see ``validate_options``). Raised before the subprocess is
                spawned and before any session store is read.
        """

        # Fail fast on invalid option combinations before spawning the
        # subprocess or touching the session store.
        validate_options(options)
        validate_session_store_options(options)

        # resume/continue + session_store: load the session from the store
        # into a temp CLAUDE_CONFIG_DIR for the subprocess to resume from.
        # Skipped when a custom transport was supplied — the materialized
        # options never reach a pre-constructed transport, so loading the
        # store and writing .credentials.json to a temp dir would be wasted.
        materialized = (
            await materialize_resume_session(options) if transport is None else None
        )
        inner = self._process_query_inner(prompt, options, transport, materialized)
        try:
            async for msg in inner:
                yield msg
        finally:
            # ``async for`` does NOT close its iterator when the loop body
            # raises (PEP 533 was deferred). Explicitly aclose the inner
            # generator first so its ``finally: await query.close()`` runs —
            # i.e. the subprocess is terminated — *before* we remove the temp
            # CLAUDE_CONFIG_DIR it is reading/writing.
            try:
                await inner.aclose()
            finally:
                # The temp dir holds a .credentials.json copy — remove it on
                # every exit path, including transport spawn failure before
                # the inner try/finally is reached.
                if materialized is not None:
                    await materialized.cleanup()

    async def _process_query_inner(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        options: ClaudeAgentOptions,
        transport: Transport | None,
        materialized: MaterializedResume | None,
    ) -> AsyncGenerator[Message, None]:
        # Validate and configure permission settings (matching TypeScript SDK logic)
        configured_options = _configure_can_use_tool(options)

        if materialized is not None:
            configured_options = apply_materialized_options(
                configured_options, materialized
            )

        # Use provided transport or create subprocess transport
        if transport is not None:
            chosen_transport = transport
        else:
            chosen_transport = SubprocessCLITransport(
                prompt=prompt,
                options=configured_options,
            )

        # Connect transport
        await chosen_transport.connect()

        # Create Query to handle control protocol. Always use streaming mode
        # internally (matching TypeScript SDK) so agents and the other
        # initialize-request fields are sent via the initialize request. The
        # kwargs come from the setup helpers shared with ClaudeSDKClient.
        query = Query(
            transport=chosen_transport,
            is_streaming_mode=True,  # Always streaming internally
            **query_kwargs_for(configured_options),
        )
        attach_session_store(query, configured_options, materialized)

        try:
            # Start reading messages
            await query.start()

            # Always initialize to send agents via stdin (matching TypeScript SDK)
            await query.initialize()

            # Handle prompt input
            if isinstance(prompt, str):
                # For string prompts, write user message to stdin after initialize
                # (matching TypeScript SDK behavior)
                user_message = {
                    "type": "user",
                    "session_id": "",
                    "message": {"role": "user", "content": prompt},
                    "parent_tool_use_id": None,
                }
                await chosen_transport.write(
                    json.dumps(
                        stamp_user_message(
                            user_message, configured_options.verbatim_prompts
                        )
                    )
                    + "\n"
                )
                query.spawn_task(query.wait_for_result_and_end_input())
            elif isinstance(prompt, AsyncIterable):
                # Stream input in background for async iterables
                query.spawn_task(query.stream_input(prompt))

            # Yield parsed messages, skipping unknown message types
            async for data in query.receive_messages():
                message = parse_message(data)
                if message is not None:
                    yield message

        finally:
            await query.close()
            query.close_receive_stream()
