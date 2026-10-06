"""Message parser for Claude Code SDK responses."""

import logging
from typing import Any, cast

from .._errors import MessageParseError
from ..types import (
    AssistantMessage,
    AuthStatusMessage,
    CompactBoundaryMessage,
    ContentBlock,
    ConversationResetMessage,
    DeferredToolUse,
    HookEventMessage,
    InitMessage,
    Message,
    MessageOrigin,
    MirrorErrorMessage,
    RateLimitEvent,
    RateLimitInfo,
    RedactedThinkingBlock,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StatusMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ThinkingBlock,
    ToolProgressMessage,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseSummaryMessage,
    UnknownBlock,
    UserMessage,
)

logger = logging.getLogger(__name__)

# Keys of a ``*_tool_result`` block that identify it rather than carry its
# payload; everything else is the result when the block has no ``content``.
_SERVER_TOOL_RESULT_IDENTITY_KEYS = frozenset({"type", "tool_use_id"})


def _parse_origin(data: dict[str, Any]) -> MessageOrigin | None:
    """Return ``data["origin"]`` if it is a well-formed origin object.

    Passed through as-is (including keys this SDK version doesn't model) so
    newer CLI origin kinds/fields stay visible to callers. Anything that is not
    an object with a string ``kind`` is treated as absent.
    """
    origin = data.get("origin")
    if isinstance(origin, dict) and isinstance(origin.get("kind"), str):
        return cast(MessageOrigin, origin)
    return None


def _opt_str(data: dict[str, Any], *keys: str) -> str | None:
    """Return the first string value found under ``keys``, else ``None``.

    Used for fields that are optional on the wire (or vary between CLI
    versions) and for keys that have both a camelCase and a snake_case
    spelling: list the wire spelling first so it wins when both are present.
    """
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


def _opt_list(data: dict[str, Any], key: str) -> list[Any]:
    """Return ``data[key]`` when it is a list, else an empty list.

    Element types are not validated; the raw value stays available on the
    message's ``data`` either way.
    """
    value = data.get(key)
    return value if isinstance(value, list) else []


def _server_tool_result_content(block: dict[str, Any]) -> dict[str, Any] | list[Any]:
    """Payload of a ``*_tool_result`` block.

    ``content`` is passed through when it is an object or a list (web search
    returns a list of hits, most other server tools a single object).
    Otherwise every field except the identifying ``type``/``tool_use_id`` is
    returned, so nothing the API sent is lost.
    """
    content = block.get("content")
    if isinstance(content, dict | list):
        return content
    return {
        key: value
        for key, value in block.items()
        if key not in _SERVER_TOOL_RESULT_IDENTITY_KEYS
    }


def _parse_content_block(block: dict[str, Any]) -> ContentBlock:
    """Parse one content block of a user or assistant message.

    Known block types parse to their typed block; a known type that lacks a
    required key raises ``KeyError`` (the caller turns it into a
    ``MessageParseError`` with the message's role in the text). Every
    ``*_tool_result`` type is a server-side tool result. Any other type is
    preserved as :class:`UnknownBlock` instead of being dropped.
    """
    block_type = block["type"]
    match block_type:
        case "text":
            return TextBlock(text=block["text"])
        case "thinking":
            # The signature is optional on the wire (summarized or omitted
            # thinking can arrive without one).
            signature = block.get("signature")
            return ThinkingBlock(
                thinking=block["thinking"],
                signature=signature if isinstance(signature, str) else "",
            )
        case "redacted_thinking":
            return RedactedThinkingBlock(data=block["data"])
        case "tool_use":
            return ToolUseBlock(
                id=block["id"],
                name=block["name"],
                input=block["input"],
            )
        case "tool_result":
            return ToolResultBlock(
                tool_use_id=block["tool_use_id"],
                content=block.get("content"),
                is_error=block.get("is_error"),
            )
        case "server_tool_use":
            return ServerToolUseBlock(
                id=block["id"],
                name=block["name"],
                input=block["input"],
            )
        case _:
            tool_use_id = block.get("tool_use_id")
            if (
                isinstance(block_type, str)
                and block_type.endswith("_tool_result")
                and isinstance(tool_use_id, str)
            ):
                return ServerToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=_server_tool_result_content(block),
                )
            logger.debug("Preserving unmodeled content block type: %s", block_type)
            return UnknownBlock(type=str(block_type), data=block)


def _parse_content_blocks(
    raw_content: list[Any], data: dict[str, Any]
) -> list[ContentBlock]:
    """Parse a message's content list, rejecting entries that are not objects."""
    blocks: list[ContentBlock] = []
    for block in raw_content:
        if not isinstance(block, dict):
            raise MessageParseError(
                f"Invalid content block (expected dict, got {type(block).__name__})",
                data,
            )
        blocks.append(_parse_content_block(block))
    return blocks


def parse_message(data: dict[str, Any]) -> Message | None:
    """
    Parse message from CLI output into typed Message objects.

    Args:
        data: Raw message dictionary from CLI output

    Returns:
        Parsed Message object, or ``None`` for message types this SDK version
        does not model (so newer CLI output never breaks older SDKs) and for
        ``tool_progress`` / ``tool_use_summary`` frames that lack the fields
        identifying them.

    Raises:
        MessageParseError: If ``data`` is not a dict, has no ``type``, or a
            recognized message type is missing a required field.
    """
    if not isinstance(data, dict):
        raise MessageParseError(
            f"Invalid message data type (expected dict, got {type(data).__name__})",
            data,
        )

    # Hook events (emitted when ``include_hook_events`` is enabled) arrive as
    # ``system`` messages with ``subtype`` of ``hook_started`` or
    # ``hook_response``. Route them to ``HookEventMessage`` before the generic
    # ``SystemMessage`` handling below.
    if data.get("type") == "system" and data.get("subtype") in (
        "hook_started",
        "hook_response",
    ):
        hook_event_name = (
            data.get("hook_event")
            or data.get("hook_name")
            or data.get("hook_event_name")
            or ""
        )
        return HookEventMessage(
            subtype=data["subtype"],
            hook_event_name=hook_event_name,
            data=data,
            session_id=data.get("session_id"),
            uuid=data.get("uuid"),
        )

    message_type = data.get("type")
    if not message_type:
        raise MessageParseError("Message missing 'type' field", data)

    match message_type:
        case "user":
            try:
                message = data["message"]
                raw_content = message["content"]
                content: str | list[ContentBlock]
                if isinstance(raw_content, list):
                    content = _parse_content_blocks(raw_content, data)
                else:
                    content = raw_content
                return UserMessage(
                    content=content,
                    uuid=data.get("uuid"),
                    parent_tool_use_id=data.get("parent_tool_use_id"),
                    tool_use_result=data.get("tool_use_result"),
                    origin=_parse_origin(data),
                )
            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in user message: {e}", data
                ) from e

        case "assistant":
            try:
                message = data["message"]
                raw_content = message["content"]
                if not isinstance(raw_content, list):
                    raise MessageParseError(
                        f"Invalid assistant content (expected list, got "
                        f"{type(raw_content).__name__})",
                        data,
                    )
                return AssistantMessage(
                    content=_parse_content_blocks(raw_content, data),
                    model=message["model"],
                    parent_tool_use_id=data.get("parent_tool_use_id"),
                    error=data.get("error"),
                    usage=message.get("usage"),
                    message_id=message.get("id"),
                    stop_reason=message.get("stop_reason"),
                    session_id=data.get("session_id"),
                    uuid=data.get("uuid"),
                )
            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in assistant message: {e}", data
                ) from e

        case "system":
            try:
                subtype = data["subtype"]
                match subtype:
                    case "init":
                        # Session preamble. Older or minimal CLIs omit most
                        # keys, so every field is optional and the complete
                        # raw frame stays on ``data``.
                        return InitMessage(
                            subtype=subtype,
                            data=data,
                            session_id=_opt_str(data, "session_id"),
                            model=_opt_str(data, "model"),
                            cwd=_opt_str(data, "cwd"),
                            tools=_opt_list(data, "tools"),
                            mcp_servers=_opt_list(data, "mcp_servers"),
                            permission_mode=_opt_str(
                                data, "permissionMode", "permission_mode"
                            ),
                            api_key_source=_opt_str(
                                data, "apiKeySource", "api_key_source"
                            ),
                            slash_commands=_opt_list(data, "slash_commands"),
                            agents=_opt_list(data, "agents"),
                            skills=_opt_list(data, "skills"),
                            plugins=_opt_list(data, "plugins"),
                            output_style=_opt_str(data, "output_style"),
                            claude_code_version=_opt_str(data, "claude_code_version"),
                            betas=_opt_list(data, "betas"),
                            uuid=_opt_str(data, "uuid"),
                        )
                    case "compact_boundary":
                        metadata = data.get("compact_metadata")
                        if not isinstance(metadata, dict):
                            metadata = {}
                        trigger = metadata.get("trigger")
                        pre_tokens = metadata.get("pre_tokens")
                        return CompactBoundaryMessage(
                            subtype=subtype,
                            data=data,
                            trigger=trigger if isinstance(trigger, str) else None,
                            pre_tokens=(
                                pre_tokens
                                if isinstance(pre_tokens, int)
                                and not isinstance(pre_tokens, bool)
                                else None
                            ),
                            session_id=_opt_str(data, "session_id"),
                            uuid=_opt_str(data, "uuid"),
                        )
                    case "status":
                        return StatusMessage(
                            subtype=subtype,
                            data=data,
                            status=_opt_str(data, "status"),
                            permission_mode=_opt_str(
                                data, "permissionMode", "permission_mode"
                            ),
                            session_id=_opt_str(data, "session_id"),
                            uuid=_opt_str(data, "uuid"),
                        )
                    case "task_started":
                        return TaskStartedMessage(
                            subtype=subtype,
                            data=data,
                            task_id=data["task_id"],
                            description=data["description"],
                            uuid=data["uuid"],
                            session_id=data["session_id"],
                            tool_use_id=data.get("tool_use_id"),
                            task_type=data.get("task_type"),
                        )
                    case "task_progress":
                        return TaskProgressMessage(
                            subtype=subtype,
                            data=data,
                            task_id=data["task_id"],
                            description=data["description"],
                            usage=data["usage"],
                            uuid=data["uuid"],
                            session_id=data["session_id"],
                            tool_use_id=data.get("tool_use_id"),
                            last_tool_name=data.get("last_tool_name"),
                        )
                    case "task_notification":
                        return TaskNotificationMessage(
                            subtype=subtype,
                            data=data,
                            task_id=data["task_id"],
                            status=data["status"],
                            output_file=data["output_file"],
                            summary=data["summary"],
                            uuid=data["uuid"],
                            session_id=data["session_id"],
                            tool_use_id=data.get("tool_use_id"),
                            usage=data.get("usage"),
                        )
                    case "task_updated":
                        # Terminal task completion sometimes arrives only as a
                        # task_updated patch (no separate task_notification), so
                        # expose it as a typed lifecycle message rather than a
                        # generic SystemMessage. Parsed defensively: the patch
                        # may omit uuid/session_id and parsing must never raise
                        # on a lifecycle event.
                        patch = data.get("patch")
                        if not isinstance(patch, dict):
                            patch = {}
                        # Terminal-ness is derived from patch.status; the CLI is
                        # assumed to set it on terminal transitions. A patch that
                        # carries only end_time/result/error (no status) is left
                        # non-terminal (status=None) — the full patch is still
                        # preserved on .patch for callers that need more.
                        return TaskUpdatedMessage(
                            subtype=subtype,
                            data=data,
                            task_id=data.get("task_id", ""),
                            patch=patch,
                            status=patch.get("status"),
                            session_id=data.get("session_id"),
                            uuid=data.get("uuid"),
                        )
                    case "mirror_error":
                        # SDK-synthesized via report_mirror_error — never emitted by the CLI subprocess.
                        return MirrorErrorMessage(
                            subtype=subtype,
                            data=data,
                            key=data.get("key"),
                            error=data.get("error", ""),
                        )
                    case _:
                        return SystemMessage(
                            subtype=subtype,
                            data=data,
                        )
            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in system message: {e}", data
                ) from e

        case "result":
            try:
                deferred = data.get("deferred_tool_use")
                return ResultMessage(
                    subtype=data["subtype"],
                    duration_ms=data["duration_ms"],
                    duration_api_ms=data["duration_api_ms"],
                    is_error=data["is_error"],
                    num_turns=data["num_turns"],
                    session_id=data["session_id"],
                    stop_reason=data.get("stop_reason"),
                    total_cost_usd=data.get("total_cost_usd"),
                    usage=data.get("usage"),
                    result=data.get("result"),
                    structured_output=data.get("structured_output"),
                    model_usage=data.get("modelUsage"),
                    permission_denials=data.get("permission_denials"),
                    deferred_tool_use=DeferredToolUse(
                        id=deferred["id"],
                        name=deferred["name"],
                        input=deferred["input"],
                    )
                    if deferred
                    else None,
                    errors=data.get("errors"),
                    api_error_status=data.get("api_error_status"),
                    uuid=data.get("uuid"),
                    terminal_reason=data.get("terminal_reason"),
                    origin=_parse_origin(data),
                )
            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in result message: {e}", data
                ) from e

        case "stream_event":
            try:
                return StreamEvent(
                    uuid=data["uuid"],
                    session_id=data["session_id"],
                    event=data["event"],
                    parent_tool_use_id=data.get("parent_tool_use_id"),
                )
            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in stream_event message: {e}", data
                ) from e

        case "rate_limit_event":
            try:
                info = data["rate_limit_info"]
                return RateLimitEvent(
                    rate_limit_info=RateLimitInfo(
                        status=info["status"],
                        resets_at=info.get("resetsAt"),
                        rate_limit_type=info.get("rateLimitType"),
                        utilization=info.get("utilization"),
                        overage_status=info.get("overageStatus"),
                        overage_resets_at=info.get("overageResetsAt"),
                        overage_disabled_reason=info.get("overageDisabledReason"),
                        raw=info,
                    ),
                    uuid=data["uuid"],
                    session_id=data["session_id"],
                )
            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in rate_limit_event message: {e}", data
                ) from e

        case "conversation_reset":
            try:
                return ConversationResetMessage(
                    new_conversation_id=data["new_conversation_id"],
                    uuid=data["uuid"],
                    session_id=data["session_id"],
                )
            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in conversation_reset message: {e}",
                    data,
                ) from e

        case "tool_progress":
            tool_use_id = data.get("tool_use_id")
            tool_name = data.get("tool_name")
            elapsed = data.get("elapsed_time_seconds")
            if (
                not isinstance(tool_use_id, str)
                or not isinstance(tool_name, str)
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, int | float)
            ):
                # Without the identifying fields there is nothing to
                # correlate the heartbeat to; skip it like an unknown type.
                logger.debug(
                    "Skipping tool_progress message without tool_use_id, "
                    "tool_name or a numeric elapsed_time_seconds"
                )
                return None
            return ToolProgressMessage(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                elapsed_time_seconds=float(elapsed),
                parent_tool_use_id=_opt_str(data, "parent_tool_use_id"),
                uuid=_opt_str(data, "uuid"),
                session_id=_opt_str(data, "session_id"),
                data=data,
            )

        case "tool_use_summary":
            summary = data.get("summary")
            if not isinstance(summary, str):
                logger.debug("Skipping tool_use_summary message without a summary")
                return None
            return ToolUseSummaryMessage(
                summary=summary,
                preceding_tool_use_ids=_opt_list(data, "preceding_tool_use_ids"),
                uuid=_opt_str(data, "uuid"),
                session_id=_opt_str(data, "session_id"),
                data=data,
            )

        case "auth_status":
            # Wire spelling is camelCase; accept snake_case as well. Anything
            # but JSON ``true`` (including an absent key) means "not
            # authenticating".
            is_authenticating = data.get("isAuthenticating")
            if is_authenticating is None:
                is_authenticating = data.get("is_authenticating")
            return AuthStatusMessage(
                is_authenticating=is_authenticating is True,
                output=_opt_list(data, "output"),
                error=_opt_str(data, "error"),
                uuid=_opt_str(data, "uuid"),
                session_id=_opt_str(data, "session_id"),
                data=data,
            )

        case _:
            # Forward-compatible: skip unrecognized message types so newer
            # CLI versions don't crash older SDK versions.
            logger.debug("Skipping unknown message type: %s", message_type)
            return None
