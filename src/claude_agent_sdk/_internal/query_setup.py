"""Helpers shared by ``query()`` and ``ClaudeSDKClient.connect()`` to set up a ``Query``.

Both entry points construct a ``SubprocessCLITransport`` and a ``Query`` in
their own modules (tests patch those names there); everything in between —
extracting the in-process MCP servers, the initialize-request fields derived
from ``system_prompt`` and ``agents``, the initialize timeout, and the
session-store mirror wiring — lives here so the two paths cannot drift apart.

None of these helpers constructs a transport or a ``Query``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from ..types import ClaudeAgentOptions, SessionKey, _hooks_to_internal_format
from .query import Query, run_end_ceiling_ms
from .session_resume import MaterializedResume, build_mirror_batcher

INITIALIZE_TIMEOUT_ENV = "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT"
DEFAULT_INITIALIZE_TIMEOUT_MS = 60_000
MIN_INITIALIZE_TIMEOUT_SECONDS = 60.0


def sdk_mcp_servers_from_options(options: ClaudeAgentOptions) -> dict[str, Any]:
    """Return ``{name: instance}`` for the in-process (``type == "sdk"``) servers.

    Only a ``dict``-valued ``mcp_servers`` can hold SDK servers; a path or an
    inline-JSON config yields ``{}``. An SDK entry without an ``instance`` is
    rejected earlier by ``validate_options``.
    """
    servers = options.mcp_servers
    if not servers or not isinstance(servers, dict):
        return {}
    return {
        name: config["instance"]  # type: ignore[typeddict-item]
        for name, config in servers.items()
        if isinstance(config, dict) and config.get("type") == "sdk"
    }


def system_prompt_init_fields(
    options: ClaudeAgentOptions,
) -> tuple[bool | None, bool | None]:
    """Return ``(exclude_dynamic_sections, system_prompt_snapshot)`` for initialize.

    ``exclude_dynamic_sections`` is read from the ``preset`` form only,
    ``snapshot`` from the ``preset`` and ``custom`` forms. Each is forwarded
    only when it is a real ``bool``; anything else leaves the field unset, so
    it is simply absent from the initialize request (which older CLIs, that
    ignore unknown initialize fields, treat the same way).
    """
    exclude_dynamic_sections: bool | None = None
    system_prompt_snapshot: bool | None = None
    sp = options.system_prompt
    if isinstance(sp, dict) and sp.get("type") == "preset":
        eds = sp.get("exclude_dynamic_sections")
        if isinstance(eds, bool):
            exclude_dynamic_sections = eds
    if isinstance(sp, dict) and sp.get("type") in ("preset", "custom"):
        snapshot = sp.get("snapshot")
        if isinstance(snapshot, bool):
            system_prompt_snapshot = snapshot
    return exclude_dynamic_sections, system_prompt_snapshot


def agents_to_wire(options: ClaudeAgentOptions) -> dict[str, dict[str, Any]] | None:
    """Convert ``options.agents`` to the shape sent in the initialize request.

    Each ``AgentDefinition`` becomes a dict with its ``None`` fields dropped
    (so the CLI applies its own defaults; falsy values such as ``[]``,
    ``0`` and ``False`` are kept). ``None`` when no agents are configured.
    """
    if not options.agents:
        return None
    return {
        name: {k: v for k, v in asdict(agent_def).items() if v is not None}
        for name, agent_def in options.agents.items()
    }


def initialize_timeout_seconds(environ: Mapping[str, str] | None = None) -> float:
    """Timeout for the initialize handshake, in seconds.

    Read from ``CLAUDE_CODE_STREAM_CLOSE_TIMEOUT`` (milliseconds, default
    60000) in ``environ`` — the process environment when not given — and
    never less than 60 seconds, so a short stream-close timeout cannot make
    the handshake itself fail early.
    """
    if environ is None:
        environ = os.environ
    timeout_ms = int(environ.get(INITIALIZE_TIMEOUT_ENV, DEFAULT_INITIALIZE_TIMEOUT_MS))
    return max(timeout_ms / 1000.0, MIN_INITIALIZE_TIMEOUT_SECONDS)


def query_kwargs_for(options: ClaudeAgentOptions) -> dict[str, Any]:
    """Every ``Query`` keyword argument that is derived from ``options``.

    Excludes ``transport`` and ``is_streaming_mode``, which the entry points
    supply themselves::

        Query(transport=transport, is_streaming_mode=True, **query_kwargs_for(options))
    """
    exclude_dynamic_sections, system_prompt_snapshot = system_prompt_init_fields(
        options
    )
    return {
        "can_use_tool": options.can_use_tool,
        "hooks": _hooks_to_internal_format(options.hooks) if options.hooks else None,
        "sdk_mcp_servers": sdk_mcp_servers_from_options(options),
        "initialize_timeout": initialize_timeout_seconds(),
        "agents": agents_to_wire(options),
        "exclude_dynamic_sections": exclude_dynamic_sections,
        "system_prompt_snapshot": system_prompt_snapshot,
        "skills": options.skills,
        "forward_subagent_text": options.forward_subagent_text,
        "verbatim_prompts": options.verbatim_prompts,
        "run_end_ceiling_ms": run_end_ceiling_ms(options.env),
    }


def attach_session_store(
    query: Query,
    options: ClaudeAgentOptions,
    materialized: MaterializedResume | None,
) -> None:
    """Wire ``options.session_store`` into ``query`` as a transcript-mirror batcher.

    No-op without a store. Otherwise builds the batcher with
    :func:`build_mirror_batcher` — resolving ``projects_dir`` to the
    materialized temp config dir when resuming from the store and honoring
    ``session_store_flush`` — and routes dropped batches to
    :meth:`Query.report_mirror_error`, which surfaces them to consumers as
    ``MirrorErrorMessage`` frames.
    """
    store = options.session_store
    if store is None:
        return

    async def _on_mirror_error(key: SessionKey | None, error: str) -> None:
        query.report_mirror_error(key, error)

    query.set_transcript_mirror_batcher(
        build_mirror_batcher(
            store=store,
            materialized=materialized,
            env=options.env,
            on_error=_on_mirror_error,
            flush_mode=options.session_store_flush,
        )
    )
