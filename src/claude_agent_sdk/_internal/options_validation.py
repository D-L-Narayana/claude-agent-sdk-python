"""Pre-flight validation of ``ClaudeAgentOptions`` combinations.

:func:`validate_options` turns the mutual exclusions and value constraints the
``ClaudeAgentOptions`` docstrings already declare into an early
:class:`ValueError`, raised by ``query()`` and ``ClaudeSDKClient.connect()``
before any subprocess is spawned or any ``SessionStore`` is read. Without it
these combinations surface late and indirectly: as a CLI exit reported as a
``ProcessError``, as a ``KeyError`` inside the SDK, or as a flag the CLI
silently ignores.

It deliberately does *not* run when the dataclass is constructed or when the
transport builds its command line, so options objects stay plain data that
can be assembled incrementally and inspected without side effects.
"""

from __future__ import annotations

import warnings

from ..types import ClaudeAgentOptions


def validate_options(options: ClaudeAgentOptions) -> None:
    """Raise :class:`ValueError` for option combinations the SDK cannot honor.

    Each rule restates a constraint from the ``ClaudeAgentOptions`` docstrings:

    - ``resume`` and ``continue_conversation`` are mutually exclusive.
    - ``session_id`` cannot be combined with ``resume`` or
      ``continue_conversation`` unless ``fork_session`` is set.
    - ``resume_session_at`` requires ``resume``.
    - ``resume_drops_turn`` (any value, including ``""``) requires
      ``resume_session_at``.
    - ``max_turns`` and ``max_budget_usd`` must not be negative.
    - every ``mcp_servers`` entry with ``type == "sdk"`` needs an ``instance``
      that ``resolve_server`` accepts (a lowlevel ``mcp.server.Server`` or a
      FastMCP server); anything else is reported here, before the CLI is
      spawned, instead of when the ``Query`` is built around it.
    - every ``plugins`` entry must have ``type == "local"``.

    Also emits a :class:`DeprecationWarning` when both ``thinking`` and the
    deprecated ``max_thinking_tokens`` are set (``thinking`` wins). Like
    ``_warn_if_can_use_tool_shadowed``, the warning uses ``stacklevel=2`` so
    it is attributed to the SDK entry point that called this function.

    Only ``query()`` and ``ClaudeSDKClient.connect()`` call this, before any
    subprocess or session-store work. Constructing ``ClaudeAgentOptions`` or
    building a CLI command from it never validates.

    Raises:
        ValueError: With a message naming the offending option(s) and how to
            fix the combination.
    """
    if options.resume is not None and options.continue_conversation:
        raise ValueError(
            "ClaudeAgentOptions.resume and continue_conversation are mutually "
            "exclusive: resume loads the given session, continue_conversation "
            "resumes the most recent one in cwd. Set only one of them."
        )

    if options.session_id is not None and not options.fork_session:
        if options.resume is not None:
            raise ValueError(
                "ClaudeAgentOptions.session_id cannot be combined with resume "
                "unless fork_session=True: a resumed session keeps its own ID. "
                "Set fork_session=True to resume into a new session with this "
                "session_id, or drop session_id."
            )
        if options.continue_conversation:
            raise ValueError(
                "ClaudeAgentOptions.session_id cannot be combined with "
                "continue_conversation unless fork_session=True: the continued "
                "session keeps its own ID. Set fork_session=True to continue "
                "into a new session with this session_id, or drop session_id."
            )

    if options.resume_session_at is not None and options.resume is None:
        raise ValueError(
            "ClaudeAgentOptions.resume_session_at requires resume: it truncates "
            "the resumed transcript at that message UUID. Set resume to the "
            "session to resume from, or drop resume_session_at."
        )

    if options.resume_drops_turn is not None and options.resume_session_at is None:
        raise ValueError(
            "ClaudeAgentOptions.resume_drops_turn requires resume_session_at: it "
            "names the turn a truncating resume discards. Set resume_session_at "
            "to the fork point, or drop resume_drops_turn."
        )

    if options.max_turns is not None and options.max_turns < 0:
        raise ValueError(
            f"ClaudeAgentOptions.max_turns must be a non-negative integer, got "
            f"{options.max_turns}. Use None for no limit."
        )

    if options.max_budget_usd is not None and options.max_budget_usd < 0:
        raise ValueError(
            f"ClaudeAgentOptions.max_budget_usd must be a non-negative number, "
            f"got {options.max_budget_usd}. Use None for no limit."
        )

    servers = options.mcp_servers
    if servers and isinstance(servers, dict):
        # Lazy: the bridge pulls in the ``mcp`` package, which this module
        # does not otherwise need. Called through the module so a patched
        # ``resolve_server`` is honored.
        from . import sdk_mcp_bridge

        for name, config in servers.items():
            if not (isinstance(config, dict) and config.get("type") == "sdk"):
                continue
            instance = config.get("instance")
            if instance is None:
                raise ValueError(
                    f"ClaudeAgentOptions.mcp_servers[{name!r}] has type 'sdk' but "
                    "no 'instance'. Pass the config returned by "
                    "create_sdk_mcp_server(), or set 'instance' to an "
                    "mcp.server.Server."
                )
            try:
                sdk_mcp_bridge.resolve_server(instance)
            except TypeError as exc:
                raise ValueError(
                    f"ClaudeAgentOptions.mcp_servers[{name!r}]: {exc}"
                ) from exc

    for index, plugin in enumerate(options.plugins):
        plugin_type = plugin.get("type")
        if plugin_type != "local":
            raise ValueError(
                f"Unsupported plugin type: {plugin_type!r} in "
                f"ClaudeAgentOptions.plugins[{index}]. Only local plugins are "
                "supported: {'type': 'local', 'path': '/path/to/plugin'}."
            )

    if options.thinking is not None and options.max_thinking_tokens is not None:
        warnings.warn(
            "ClaudeAgentOptions.max_thinking_tokens is deprecated and ignored "
            "when thinking is also set; thinking takes precedence. Remove "
            "max_thinking_tokens, or use thinking={'type': 'enabled', "
            "'budget_tokens': N} for a fixed thinking budget.",
            DeprecationWarning,
            stacklevel=2,
        )
