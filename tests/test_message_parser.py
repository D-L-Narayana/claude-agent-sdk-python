"""Tests for message parser error handling."""

from typing import Any, get_args

import pytest

from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk._internal.message_parser import parse_message
from claude_agent_sdk.types import (
    TERMINAL_TASK_STATUSES,
    AssistantMessage,
    AuthStatusMessage,
    CompactBoundaryMessage,
    ConversationResetMessage,
    DeferredToolUse,
    HookEventMessage,
    InitMessage,
    MirrorErrorMessage,
    RateLimitEvent,
    RedactedThinkingBlock,
    ResultMessage,
    ServerToolName,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StatusMessage,
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


class TestMessageParser:
    """Test message parsing with the new exception behavior."""

    def test_parse_valid_user_message(self):
        """Test parsing a valid user message."""
        data = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "Hello"}]},
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Hello"

    def test_parse_user_message_with_uuid(self):
        """Test parsing a user message with uuid field (issue #414).

        The uuid field is needed for file checkpointing with rewind_files().
        """
        data = {
            "type": "user",
            "uuid": "msg-abc123-def456",
            "message": {"content": [{"type": "text", "text": "Hello"}]},
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert message.uuid == "msg-abc123-def456"
        assert len(message.content) == 1

    def test_parse_user_message_origin(self):
        """origin is surfaced on user messages, for both content shapes, and
        passed through with keys this SDK version doesn't model."""
        peer = {
            "kind": "peer",
            "from": "peer-addr",
            "name": "other-session",
            "verifiedPeerPid": 4242,
            "someFutureField": True,
        }
        for content in ("hi", [{"type": "text", "text": "hi"}]):
            message = parse_message(
                {"type": "user", "message": {"content": content}, "origin": peer}
            )
            assert isinstance(message, UserMessage)
            assert message.origin == peer
            assert message.origin is not None and message.origin["kind"] == "peer"
            assert message.origin["from"] == "peer-addr"

    def test_parse_user_message_origin_absent_or_malformed(self):
        """No origin, or a non-object / kind-less origin, parses to None."""
        for extra in ({}, {"origin": None}, {"origin": "human"}, {"origin": {}}):
            message = parse_message(
                {"type": "user", "message": {"content": "hi"}, **extra}
            )
            assert isinstance(message, UserMessage)
            assert message.origin is None, extra

    def test_parse_user_message_with_tool_use(self):
        """Test parsing a user message with tool_use block."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "Let me read this file"},
                    {
                        "type": "tool_use",
                        "id": "tool_456",
                        "name": "Read",
                        "input": {"file_path": "/example.txt"},
                    },
                ]
            },
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ToolUseBlock)
        assert message.content[1].id == "tool_456"
        assert message.content[1].name == "Read"
        assert message.content[1].input == {"file_path": "/example.txt"}

    def test_parse_user_message_with_tool_result(self):
        """Test parsing a user message with tool_result block."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_789",
                        "content": "File contents here",
                    }
                ]
            },
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert len(message.content) == 1
        assert isinstance(message.content[0], ToolResultBlock)
        assert message.content[0].tool_use_id == "tool_789"
        assert message.content[0].content == "File contents here"

    def test_parse_user_message_with_tool_result_error(self):
        """Test parsing a user message with error tool_result block."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_error",
                        "content": "File not found",
                        "is_error": True,
                    }
                ]
            },
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert len(message.content) == 1
        assert isinstance(message.content[0], ToolResultBlock)
        assert message.content[0].tool_use_id == "tool_error"
        assert message.content[0].content == "File not found"
        assert message.content[0].is_error is True

    def test_parse_user_message_with_mixed_content(self):
        """Test parsing a user message with mixed content blocks."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "Here's what I found:"},
                    {
                        "type": "tool_use",
                        "id": "use_1",
                        "name": "Search",
                        "input": {"query": "test"},
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "use_1",
                        "content": "Search results",
                    },
                    {"type": "text", "text": "What do you think?"},
                ]
            },
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert len(message.content) == 4
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ToolUseBlock)
        assert isinstance(message.content[2], ToolResultBlock)
        assert isinstance(message.content[3], TextBlock)

    def test_parse_user_message_inside_subagent(self):
        """Test parsing a valid user message."""
        data = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "Hello"}]},
            "parent_tool_use_id": "toolu_01Xrwd5Y13sEHtzScxR77So8",
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert message.parent_tool_use_id == "toolu_01Xrwd5Y13sEHtzScxR77So8"

    def test_parse_user_message_with_tool_use_result(self):
        """Test parsing a user message with tool_use_result field.

        The tool_use_result field contains metadata about tool execution results,
        including file edit details like oldString, newString, and structuredPatch.
        """
        tool_result_data = {
            "filePath": "/path/to/file.py",
            "oldString": "old code",
            "newString": "new code",
            "originalFile": "full file contents",
            "structuredPatch": [
                {
                    "oldStart": 33,
                    "oldLines": 7,
                    "newStart": 33,
                    "newLines": 7,
                    "lines": [
                        "   # comment",
                        "-      old line",
                        "+      new line",
                    ],
                }
            ],
            "userModified": False,
            "replaceAll": False,
        }
        data = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "toolu_vrtx_01KXWexk3NJdwkjWzPMGQ2F1",
                        "type": "tool_result",
                        "content": "The file has been updated.",
                    }
                ],
            },
            "parent_tool_use_id": None,
            "session_id": "84afb479-17ae-49af-8f2b-666ac2530c3a",
            "uuid": "2ace3375-1879-48a0-a421-6bce25a9295a",
            "tool_use_result": tool_result_data,
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert message.tool_use_result == tool_result_data
        assert message.tool_use_result["filePath"] == "/path/to/file.py"
        assert message.tool_use_result["oldString"] == "old code"
        assert message.tool_use_result["newString"] == "new code"
        assert message.tool_use_result["structuredPatch"][0]["oldStart"] == 33
        assert message.uuid == "2ace3375-1879-48a0-a421-6bce25a9295a"

    def test_parse_user_message_with_string_content_and_tool_use_result(self):
        """Test parsing a user message with string content and tool_use_result."""
        tool_result_data = {"filePath": "/path/to/file.py", "userModified": True}
        data = {
            "type": "user",
            "message": {"content": "Simple string content"},
            "tool_use_result": tool_result_data,
        }
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert message.content == "Simple string content"
        assert message.tool_use_result == tool_result_data

    def test_parse_valid_assistant_message(self):
        """Test parsing a valid assistant message."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "Read",
                        "input": {"file_path": "/test.txt"},
                    },
                ],
                "model": "claude-opus-4-1-20250805",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ToolUseBlock)

    def test_parse_assistant_message_with_thinking(self):
        """Test parsing an assistant message with thinking block."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I'm thinking about the answer...",
                        "signature": "sig-123",
                    },
                    {"type": "text", "text": "Here's my response"},
                ],
                "model": "claude-opus-4-1-20250805",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 2
        assert isinstance(message.content[0], ThinkingBlock)
        assert message.content[0].thinking == "I'm thinking about the answer..."
        assert message.content[0].signature == "sig-123"
        assert isinstance(message.content[1], TextBlock)
        assert message.content[1].text == "Here's my response"

    def test_parse_assistant_message_with_server_tool_use(self):
        """server_tool_use blocks (e.g. advisor, web_search) are preserved.

        Previously these were dropped, leaving an empty content list on
        messages that only contained a server tool call.
        """
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_01ABC",
                        "name": "advisor",
                        "input": {},
                    },
                ],
                "model": "claude-sonnet-4-5",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        assert isinstance(message.content[0], ServerToolUseBlock)
        assert message.content[0].id == "srvtoolu_01ABC"
        assert message.content[0].name == "advisor"
        assert message.content[0].input == {}

    def test_parse_assistant_message_with_server_tool_result(self):
        """Server-side tool result blocks (e.g. advisor) surface with their raw content dict.

        `content` is passed through as a dict since its shape is tool-specific
        (advisor emits advisor_result / advisor_redacted_result /
        advisor_tool_result_error; other server tools use different shapes).
        """
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "advisor_tool_result",
                        "tool_use_id": "srvtoolu_01ABC",
                        "content": {
                            "type": "advisor_result",
                            "text": "Consider edge cases around empty input.",
                        },
                    },
                ],
                "model": "claude-sonnet-4-5",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        result_block = message.content[0]
        assert isinstance(result_block, ServerToolResultBlock)
        assert result_block.tool_use_id == "srvtoolu_01ABC"
        assert result_block.content == {
            "type": "advisor_result",
            "text": "Consider edge cases around empty input.",
        }

    def test_parse_assistant_message_with_redacted_advisor_result(self):
        """External API users get advisor output as an encrypted blob in the content dict."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "advisor_tool_result",
                        "tool_use_id": "srvtoolu_01ABC",
                        "content": {
                            "type": "advisor_redacted_result",
                            "encrypted_content": "EuYDCioIDhgC...",
                        },
                    },
                ],
                "model": "claude-sonnet-4-5",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        result_block = message.content[0]
        assert isinstance(result_block, ServerToolResultBlock)
        assert result_block.content["type"] == "advisor_redacted_result"
        assert result_block.content["encrypted_content"] == "EuYDCioIDhgC..."

    def test_parse_assistant_message_with_usage(self):
        """Per-turn usage is preserved on AssistantMessage.

        The CLI emits the API's full usage dict (including cache token
        breakdown) on every assistant message. Previously this was dropped
        by the parser, forcing consumers to wait for the aggregate in
        ResultMessage. See issue #673.
        """
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "hi"}],
                "model": "claude-opus-4-5",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 2000,
                    "cache_creation_input_tokens": 500,
                },
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.usage == {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 2000,
            "cache_creation_input_tokens": 500,
        }

    def test_parse_assistant_message_without_usage(self):
        """usage defaults to None when absent (e.g. synthetic messages)."""
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "hi"}],
                "model": "claude-opus-4-5",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.usage is None

    def test_parse_valid_system_message(self):
        """Test parsing a valid system message."""
        data = {"type": "system", "subtype": "start"}
        message = parse_message(data)
        assert isinstance(message, SystemMessage)
        assert message.subtype == "start"

    def test_parse_task_started_message(self):
        """Test parsing a task_started system message yields a TaskStartedMessage."""
        data = {
            "type": "system",
            "subtype": "task_started",
            "task_id": "task-abc",
            "tool_use_id": "toolu_01",
            "description": "Reticulating splines",
            "task_type": "background",
            "uuid": "uuid-1",
            "session_id": "session-1",
        }
        message = parse_message(data)
        assert isinstance(message, TaskStartedMessage)
        assert message.task_id == "task-abc"
        assert message.description == "Reticulating splines"
        assert message.uuid == "uuid-1"
        assert message.session_id == "session-1"
        assert message.tool_use_id == "toolu_01"
        assert message.task_type == "background"

    def test_parse_task_started_message_optional_fields_absent(self):
        """task_started with no optional fields should still parse, optionals set to None."""
        data = {
            "type": "system",
            "subtype": "task_started",
            "task_id": "task-abc",
            "description": "Working",
            "uuid": "uuid-1",
            "session_id": "session-1",
        }
        message = parse_message(data)
        assert isinstance(message, TaskStartedMessage)
        assert message.tool_use_id is None
        assert message.task_type is None

    def test_parse_task_progress_message(self):
        """Test parsing a task_progress system message yields a TaskProgressMessage."""
        data = {
            "type": "system",
            "subtype": "task_progress",
            "task_id": "task-abc",
            "tool_use_id": "toolu_01",
            "description": "Halfway there",
            "usage": {
                "total_tokens": 1234,
                "tool_uses": 5,
                "duration_ms": 9876,
            },
            "last_tool_name": "Read",
            "uuid": "uuid-2",
            "session_id": "session-1",
        }
        message = parse_message(data)
        assert isinstance(message, TaskProgressMessage)
        assert message.task_id == "task-abc"
        assert message.description == "Halfway there"
        assert message.usage == {
            "total_tokens": 1234,
            "tool_uses": 5,
            "duration_ms": 9876,
        }
        assert message.last_tool_name == "Read"
        assert message.tool_use_id == "toolu_01"
        assert message.uuid == "uuid-2"
        assert message.session_id == "session-1"

    def test_parse_task_notification_message(self):
        """Test parsing a task_notification system message yields a TaskNotificationMessage."""
        data = {
            "type": "system",
            "subtype": "task_notification",
            "task_id": "task-abc",
            "tool_use_id": "toolu_01",
            "status": "completed",
            "output_file": "/tmp/out.md",
            "summary": "All done",
            "usage": {
                "total_tokens": 2000,
                "tool_uses": 7,
                "duration_ms": 12345,
            },
            "uuid": "uuid-3",
            "session_id": "session-1",
        }
        message = parse_message(data)
        assert isinstance(message, TaskNotificationMessage)
        assert message.task_id == "task-abc"
        assert message.status == "completed"
        assert message.output_file == "/tmp/out.md"
        assert message.summary == "All done"
        assert message.usage == {
            "total_tokens": 2000,
            "tool_uses": 7,
            "duration_ms": 12345,
        }
        assert message.tool_use_id == "toolu_01"
        assert message.uuid == "uuid-3"
        assert message.session_id == "session-1"

    def test_parse_task_notification_message_optional_fields_absent(self):
        """task_notification with no optional fields (usage, tool_use_id) still parses."""
        data = {
            "type": "system",
            "subtype": "task_notification",
            "task_id": "task-abc",
            "status": "failed",
            "output_file": "/tmp/out.md",
            "summary": "Boom",
            "uuid": "uuid-3",
            "session_id": "session-1",
        }
        message = parse_message(data)
        assert isinstance(message, TaskNotificationMessage)
        assert message.status == "failed"
        assert message.usage is None
        assert message.tool_use_id is None

    def test_parse_task_updated_message_terminal(self):
        """task_updated with a terminal patch.status yields a TaskUpdatedMessage."""
        data = {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "task-abc",
            "patch": {"status": "completed", "end_time": 1780405729183},
            "uuid": "uuid-4",
            "session_id": "session-1",
        }
        message = parse_message(data)
        assert isinstance(message, TaskUpdatedMessage)
        assert message.task_id == "task-abc"
        assert message.patch == {"status": "completed", "end_time": 1780405729183}
        assert message.status == "completed"
        assert message.uuid == "uuid-4"
        assert message.session_id == "session-1"
        assert message.status in TERMINAL_TASK_STATUSES

    def test_parse_task_updated_message_minimal(self):
        """task_updated with only task_id and patch (no uuid/session_id) still parses.

        Mirrors the observed CLI shape where terminal completion arrives as a
        bare task_updated patch — parsing must never raise on a lifecycle event.
        """
        data = {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "b1m21w89v",
            "patch": {"status": "completed", "end_time": 1780405729183},
        }
        message = parse_message(data)
        assert isinstance(message, TaskUpdatedMessage)
        assert message.task_id == "b1m21w89v"
        assert message.status == "completed"
        assert message.uuid is None
        assert message.session_id is None

    @pytest.mark.parametrize("status", ["pending", "running", "paused"])
    def test_parse_task_updated_message_non_terminal_statuses(self, status):
        """Non-terminal task_updated statuses parse and are not treated as done."""
        data = {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "task-abc",
            "patch": {"status": status},
        }
        message = parse_message(data)
        assert isinstance(message, TaskUpdatedMessage)
        assert message.status == status
        assert message.status not in TERMINAL_TASK_STATUSES

    def test_parse_task_updated_message_no_patch(self):
        """task_updated with no patch parses with an empty patch and status None."""
        data = {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "task-abc",
        }
        message = parse_message(data)
        assert isinstance(message, TaskUpdatedMessage)
        assert message.patch == {}
        assert message.status is None

    def test_parse_task_updated_message_patch_without_status(self):
        """A patch lacking 'status' is preserved verbatim; status is None."""
        data = {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "task-abc",
            "patch": {"end_time": 1780405729183},
        }
        message = parse_message(data)
        assert isinstance(message, TaskUpdatedMessage)
        assert message.patch == {"end_time": 1780405729183}
        assert message.status is None

    @pytest.mark.parametrize("patch", ["completed", ["completed"], 42, None])
    def test_parse_task_updated_message_non_dict_patch(self, patch):
        """A non-dict (or missing) patch never raises; patch falls back to {}."""
        data = {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "task-abc",
            "patch": patch,
        }
        message = parse_message(data)
        assert isinstance(message, TaskUpdatedMessage)
        assert message.patch == {}
        assert message.status is None

    @pytest.mark.parametrize("status", ["completed", "failed", "killed"])
    def test_parse_task_updated_message_terminal_statuses(self, status):
        """Every terminal task_updated patch.status is surfaced as terminal.

        ``task_updated`` reports the raw ``killed`` (not the ``stopped`` form
        the CLI maps to on ``task_notification``).
        """
        data = {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "task-abc",
            "patch": {"status": status},
        }
        message = parse_message(data)
        assert isinstance(message, TaskUpdatedMessage)
        assert message.status == status
        assert message.status in TERMINAL_TASK_STATUSES

    def test_parse_task_updated_killed_is_terminal(self):
        """A task stopped via TaskStop reports status='killed' and is terminal.

        In some kill paths no task_notification is emitted, so this task_updated
        patch is the only terminal signal — it must clear a tracked active id.
        """
        data = {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "bs2r8eew4",
            "patch": {"status": "killed", "end_time": 1780405729183},
        }
        message = parse_message(data)
        assert isinstance(message, TaskUpdatedMessage)
        assert message.status == "killed"
        assert message.status in TERMINAL_TASK_STATUSES

    def test_task_updated_backward_compat_isinstance(self):
        """Backward-compat: TaskUpdatedMessage is still a SystemMessage."""
        data = {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "t1",
            "patch": {"status": "failed"},
            "uuid": "u1",
            "session_id": "s1",
        }
        message = parse_message(data)
        assert isinstance(message, TaskUpdatedMessage)
        assert isinstance(message, SystemMessage)
        # Base class fields still populated for legacy code paths.
        assert message.subtype == "task_updated"
        assert message.data == data
        # match-case against SystemMessage still works.
        matched = False
        match message:
            case SystemMessage():
                matched = True
        assert matched

    def test_task_message_backward_compat_isinstance(self):
        """Backward-compat: typed task messages are still SystemMessage instances."""
        started_data = {
            "type": "system",
            "subtype": "task_started",
            "task_id": "t1",
            "description": "desc",
            "uuid": "u1",
            "session_id": "s1",
        }
        progress_data = {
            "type": "system",
            "subtype": "task_progress",
            "task_id": "t1",
            "description": "desc",
            "usage": {"total_tokens": 1, "tool_uses": 0, "duration_ms": 10},
            "uuid": "u2",
            "session_id": "s1",
        }
        notif_data = {
            "type": "system",
            "subtype": "task_notification",
            "task_id": "t1",
            "status": "stopped",
            "output_file": "/o",
            "summary": "s",
            "uuid": "u3",
            "session_id": "s1",
        }
        started = parse_message(started_data)
        progress = parse_message(progress_data)
        notif = parse_message(notif_data)
        # isinstance checks against the base class still work
        assert isinstance(started, SystemMessage)
        assert isinstance(progress, SystemMessage)
        assert isinstance(notif, SystemMessage)
        # match-case against SystemMessage still works
        matched = False
        match started:
            case SystemMessage():
                matched = True
        assert matched

    def test_task_message_backward_compat_base_fields(self):
        """Backward-compat: subtype and data fields on typed task messages are populated."""
        data = {
            "type": "system",
            "subtype": "task_started",
            "task_id": "t1",
            "description": "desc",
            "uuid": "u1",
            "session_id": "s1",
        }
        message = parse_message(data)
        assert isinstance(message, TaskStartedMessage)
        # Base class fields still populated for legacy code paths
        assert message.subtype == "task_started"
        assert message.data == data
        assert message.data["task_id"] == "t1"

    def test_unknown_system_subtype_yields_generic(self):
        """Unknown system subtypes fall through to generic SystemMessage (not a subclass)."""
        data = {"type": "system", "subtype": "some_future_subtype", "foo": "bar"}
        message = parse_message(data)
        assert isinstance(message, SystemMessage)
        # Ensure it's exactly SystemMessage, not one of the typed subclasses
        assert type(message) is SystemMessage
        assert not isinstance(message, TaskStartedMessage)
        assert not isinstance(message, TaskProgressMessage)
        assert not isinstance(message, TaskNotificationMessage)
        assert not isinstance(message, TaskUpdatedMessage)
        assert message.subtype == "some_future_subtype"
        assert message.data == data

    def test_parse_assistant_message_inside_subagent(self):
        """Test parsing a valid assistant message."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "Read",
                        "input": {"file_path": "/test.txt"},
                    },
                ],
                "model": "claude-opus-4-1-20250805",
            },
            "parent_tool_use_id": "toolu_01Xrwd5Y13sEHtzScxR77So8",
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.parent_tool_use_id == "toolu_01Xrwd5Y13sEHtzScxR77So8"

    def test_parse_valid_result_message(self):
        """Test parsing a valid result message."""
        data = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1000,
            "duration_api_ms": 500,
            "is_error": False,
            "num_turns": 2,
            "session_id": "session_123",
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.subtype == "success"
        assert message.stop_reason is None

    def test_parse_result_message_with_stop_reason(self):
        """Test parsing a result message with stop_reason field.

        The stop_reason field mirrors the Anthropic API's stop_reason on the
        final assistant turn (e.g., "end_turn", "max_tokens", "tool_use").
        """
        data = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1000,
            "duration_api_ms": 500,
            "is_error": False,
            "num_turns": 2,
            "session_id": "session_123",
            "stop_reason": "end_turn",
            "result": "Done",
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.stop_reason == "end_turn"
        assert message.result == "Done"

    def test_parse_result_message_with_null_stop_reason(self):
        """Test parsing a result message with explicit null stop_reason."""
        data = {
            "type": "result",
            "subtype": "error_max_turns",
            "duration_ms": 1000,
            "duration_api_ms": 500,
            "is_error": True,
            "num_turns": 10,
            "session_id": "session_123",
            "stop_reason": None,
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.stop_reason is None

    def test_parse_result_message_with_terminal_reason(self):
        """Test parsing a result message with terminal_reason field."""
        data = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1000,
            "duration_api_ms": 500,
            "is_error": False,
            "num_turns": 2,
            "session_id": "session_123",
            "result": "",
            "terminal_reason": "aborted_tools",
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.terminal_reason == "aborted_tools"

    def test_parse_result_message_origin(self):
        """origin on a result identifies what triggered the turn."""
        base = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1000,
            "duration_api_ms": 500,
            "is_error": False,
            "num_turns": 2,
            "session_id": "session_123",
        }
        message = parse_message(base)
        assert isinstance(message, ResultMessage)
        assert message.origin is None

        message = parse_message({**base, "origin": {"kind": "human"}})
        assert isinstance(message, ResultMessage)
        assert message.origin == {"kind": "human"}

        for subkind in ("scheduled-trigger", "peer-send-message"):
            origin = {"kind": "task-notification", "subkind": subkind}
            message = parse_message({**base, "origin": origin})
            assert isinstance(message, ResultMessage)
            assert message.origin == origin

        message = parse_message({**base, "origin": {"kind": "unclassified"}})
        assert isinstance(message, ResultMessage)
        assert message.origin is not None
        assert message.origin["kind"] == "unclassified"

    def test_parse_result_message_missing_terminal_reason_is_none(self):
        """A result message without terminal_reason parses to None."""
        data = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1000,
            "duration_api_ms": 500,
            "is_error": False,
            "num_turns": 2,
            "session_id": "session_123",
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.terminal_reason is None

    def test_parse_rate_limit_event(self):
        """Test parsing a rate_limit_event into a typed RateLimitEvent."""
        data = {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "allowed_warning",
                "resetsAt": 1700000000,
                "rateLimitType": "five_hour",
                "utilization": 0.91,
            },
            "uuid": "abc-123",
            "session_id": "session_xyz",
        }
        message = parse_message(data)
        assert isinstance(message, RateLimitEvent)
        assert message.uuid == "abc-123"
        assert message.session_id == "session_xyz"
        assert message.rate_limit_info.status == "allowed_warning"
        assert message.rate_limit_info.resets_at == 1700000000
        assert message.rate_limit_info.rate_limit_type == "five_hour"
        assert message.rate_limit_info.utilization == 0.91

    def test_parse_conversation_reset(self):
        """conversation_reset parses into a typed ConversationResetMessage."""
        data = {
            "type": "conversation_reset",
            "new_conversation_id": "d2f4a573-ca99-42a2-bb7a-905b40c908e8",
            "uuid": "msg-1",
            "session_id": "66694129-ce74-4ee1-9b0f-994155ac97ba",
        }
        message = parse_message(data)
        assert isinstance(message, ConversationResetMessage)
        assert message.new_conversation_id == "d2f4a573-ca99-42a2-bb7a-905b40c908e8"
        assert message.uuid == "msg-1"
        assert message.session_id == "66694129-ce74-4ee1-9b0f-994155ac97ba"

    def test_parse_conversation_reset_missing_field(self):
        """conversation_reset without new_conversation_id raises."""
        with pytest.raises(MessageParseError) as exc_info:
            parse_message(
                {"type": "conversation_reset", "uuid": "u", "session_id": "s"}
            )
        assert "new_conversation_id" in str(exc_info.value)

    def test_parse_invalid_data_type(self):
        """Test that non-dict data raises MessageParseError."""
        with pytest.raises(MessageParseError) as exc_info:
            parse_message("not a dict")  # type: ignore
        assert "Invalid message data type" in str(exc_info.value)
        assert "expected dict, got str" in str(exc_info.value)

    def test_parse_missing_type_field(self):
        """Test that missing 'type' field raises MessageParseError."""
        with pytest.raises(MessageParseError) as exc_info:
            parse_message({"message": {"content": []}})
        assert "Message missing 'type' field" in str(exc_info.value)

    def test_parse_unknown_message_type(self):
        """Test that unknown message type returns None for forward compatibility."""
        result = parse_message({"type": "unknown_type"})
        assert result is None

    def test_parse_user_message_missing_fields(self):
        """Test that user message with missing fields raises MessageParseError."""
        with pytest.raises(MessageParseError) as exc_info:
            parse_message({"type": "user"})
        assert "Missing required field in user message" in str(exc_info.value)

    def test_parse_assistant_message_missing_fields(self):
        """Test that assistant message with missing fields raises MessageParseError."""
        with pytest.raises(MessageParseError) as exc_info:
            parse_message({"type": "assistant"})
        assert "Missing required field in assistant message" in str(exc_info.value)

    def test_parse_assistant_string_content_raises(self):
        """Assistant content as a bare string raises MessageParseError, not a raw TypeError."""
        with pytest.raises(MessageParseError):
            parse_message(
                {"type": "assistant", "message": {"model": "m", "content": "hi"}}
            )

    @pytest.mark.parametrize("role", ["assistant", "user"])
    def test_non_dict_content_block_raises_documented_error(self, role: str) -> None:
        """A non-dict block raises MessageParseError, never a raw TypeError."""
        message: dict[str, object] = {"content": ["oops"]}
        if role == "assistant":
            message["model"] = "m"
        with pytest.raises(MessageParseError):
            parse_message({"type": role, "message": message})

    def test_parse_system_message_missing_fields(self):
        """Test that system message with missing fields raises MessageParseError."""
        with pytest.raises(MessageParseError) as exc_info:
            parse_message({"type": "system"})
        assert "Missing required field in system message" in str(exc_info.value)

    def test_parse_result_message_missing_fields(self):
        """Test that result message with missing fields raises MessageParseError."""
        with pytest.raises(MessageParseError) as exc_info:
            parse_message({"type": "result", "subtype": "success"})
        assert "Missing required field in result message" in str(exc_info.value)

    def test_message_parse_error_contains_data(self):
        """Test that MessageParseError contains the original data."""
        # Use a malformed known type (missing required fields) to trigger error
        data = {"type": "assistant"}
        with pytest.raises(MessageParseError) as exc_info:
            parse_message(data)
        assert exc_info.value.data == data

    def test_parse_assistant_message_without_error(self):
        """Test that assistant message without error has error=None."""
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello"}],
                "model": "claude-opus-4-5-20251101",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.error is None

    def test_parse_assistant_message_with_authentication_error(self):
        """Test parsing assistant message with authentication_failed error.

        The error field is at the top level of the data, not inside message.
        This matches the actual CLI output format.
        """
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Invalid API key · Fix external API key"}
                ],
                "model": "<synthetic>",
            },
            "session_id": "test-session",
            "error": "authentication_failed",
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.error == "authentication_failed"
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextBlock)

    def test_parse_assistant_message_with_unknown_error(self):
        """Test parsing assistant message with unknown error (e.g., 404, 500).

        When the CLI encounters API errors like model not found or server errors,
        it sets error to 'unknown' and includes the error details in the text content.
        """
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": 'API Error: 500 {"type":"error","error":{"type":"api_error","message":"Internal server error"}}',
                    }
                ],
                "model": "<synthetic>",
            },
            "session_id": "test-session",
            "error": "unknown",
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.error == "unknown"

    def test_parse_assistant_message_with_rate_limit_error(self):
        """Test parsing assistant message with rate_limit error."""
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Rate limit exceeded"}],
                "model": "<synthetic>",
            },
            "error": "rate_limit",
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.error == "rate_limit"

    def test_parse_assistant_message_with_all_fields(self):
        """Test that AssistantMessage preserves id, stop_reason, session_id, uuid."""
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello"}],
                "model": "claude-sonnet-4-5-20250929",
                "id": "msg_01HRq7YZE3apPqSHydvG77Ve",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            "session_id": "fdf2d90a-fd9e-4736-ae35-806edd13643f",
            "uuid": "0dbd2453-1209-4fe9-bd51-4102f64e33df",
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.message_id == "msg_01HRq7YZE3apPqSHydvG77Ve"
        assert message.stop_reason == "end_turn"
        assert message.session_id == "fdf2d90a-fd9e-4736-ae35-806edd13643f"
        assert message.uuid == "0dbd2453-1209-4fe9-bd51-4102f64e33df"
        assert message.usage == {"input_tokens": 10, "output_tokens": 5}

    def test_parse_assistant_message_optional_fields_absent(self):
        """New optional fields default to None when absent."""
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "hi"}],
                "model": "claude-opus-4-5",
            },
        }
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert message.message_id is None
        assert message.stop_reason is None
        assert message.session_id is None
        assert message.uuid is None

    def test_parse_result_message_with_model_usage(self):
        """Test that ResultMessage preserves modelUsage, permission_denials, uuid."""
        data = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 3000,
            "duration_api_ms": 2000,
            "is_error": False,
            "num_turns": 1,
            "session_id": "fdf2d90a-fd9e-4736-ae35-806edd13643f",
            "stop_reason": "end_turn",
            "total_cost_usd": 0.0106,
            "usage": {"input_tokens": 3, "output_tokens": 24},
            "result": "Hello",
            "modelUsage": {
                "claude-sonnet-4-5-20250929": {
                    "inputTokens": 3,
                    "outputTokens": 24,
                    "cacheReadInputTokens": 20012,
                    "costUSD": 0.0106,
                    "contextWindow": 200000,
                    "maxOutputTokens": 64000,
                }
            },
            "permission_denials": [],
            "uuid": "d379c496-f33a-4ea4-b920-3c5483baa6f7",
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.model_usage is not None
        assert "claude-sonnet-4-5-20250929" in message.model_usage
        assert message.model_usage["claude-sonnet-4-5-20250929"]["costUSD"] == 0.0106
        assert message.permission_denials == []
        assert message.uuid == "d379c496-f33a-4ea4-b920-3c5483baa6f7"

    def test_parse_result_message_optional_fields_absent(self):
        """New optional fields default to None when absent."""
        data = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1000,
            "duration_api_ms": 500,
            "is_error": False,
            "num_turns": 1,
            "session_id": "session_123",
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.model_usage is None
        assert message.permission_denials is None
        assert message.deferred_tool_use is None
        assert message.errors is None
        assert message.api_error_status is None
        assert message.uuid is None

    def test_parse_result_message_with_deferred_tool_use(self):
        """ResultMessage parses deferred_tool_use into a DeferredToolUse."""
        data = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1200,
            "duration_api_ms": 900,
            "is_error": False,
            "num_turns": 1,
            "session_id": "session_123",
            "deferred_tool_use": {
                "id": "toolu_01abc",
                "name": "Bash",
                "input": {"command": "rm -rf /tmp/scratch"},
            },
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert isinstance(message.deferred_tool_use, DeferredToolUse)
        assert message.deferred_tool_use.id == "toolu_01abc"
        assert message.deferred_tool_use.name == "Bash"
        assert message.deferred_tool_use.input == {"command": "rm -rf /tmp/scratch"}

    def test_parse_result_message_with_errors(self):
        """Test that ResultMessage preserves the errors field from error results.

        The CLI emits errors: string[] on error result messages (subtypes like
        error_during_execution, error_max_turns, etc.). Without this field,
        SDK users cannot diagnose why a non-zero exit occurred.
        """
        data = {
            "type": "result",
            "subtype": "error_during_execution",
            "duration_ms": 5000,
            "duration_api_ms": 3000,
            "is_error": True,
            "num_turns": 3,
            "session_id": "session_456",
            "errors": [
                "Tool execution failed: permission denied",
                "Unable to write to /etc/hosts",
            ],
            "uuid": "err-uuid-789",
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.errors == [
            "Tool execution failed: permission denied",
            "Unable to write to /etc/hosts",
        ]
        assert message.is_error is True
        assert message.subtype == "error_during_execution"
        assert message.uuid == "err-uuid-789"

    def test_parse_result_message_with_api_error_status(self):
        """ResultMessage surfaces api_error_status for failed API calls.

        The CLI (v2.1.110+) emits api_error_status: number | null on the final
        result message — the HTTP status of the failing API call when
        is_error=True and subtype="success". This is the only safe-to-log
        signal for classifying API failures (e.g. 429 vs 529).
        """
        data = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 2000,
            "duration_api_ms": 1500,
            "is_error": True,
            "num_turns": 1,
            "session_id": "session_overload",
            "api_error_status": 529,
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.api_error_status == 529
        assert message.is_error is True
        assert message.subtype == "success"

    def test_parse_result_message_success_no_errors(self):
        """Test that a successful result message has no errors field."""
        data = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1000,
            "duration_api_ms": 500,
            "is_error": False,
            "num_turns": 1,
            "session_id": "session_789",
            "result": "Task completed successfully",
        }
        message = parse_message(data)
        assert isinstance(message, ResultMessage)
        assert message.errors is None
        assert message.result == "Task completed successfully"

    def test_parse_hook_event_message(self):
        """Hook started events (system/hook_started) parse into HookEventMessage."""
        data = {
            "type": "system",
            "subtype": "hook_started",
            "hook_event": "PreToolUse",
            "hook_name": "PreToolUse",
            "session_id": "sess-123",
            "uuid": "uuid-456",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
        }
        message = parse_message(data)
        assert isinstance(message, HookEventMessage)
        assert message.subtype == "hook_started"
        assert message.hook_event_name == "PreToolUse"
        assert message.session_id == "sess-123"
        assert message.uuid == "uuid-456"
        assert message.data == data

    def test_parse_hook_event_message_response(self):
        """Hook response events (system/hook_response) parse into HookEventMessage."""
        data = {
            "type": "system",
            "subtype": "hook_response",
            "hook_event": "PostToolUse",
            "hook_name": "PostToolUse",
            "session_id": "sess-123",
            "uuid": "uuid-789",
            "output": "",
            "exit_code": 0,
            "outcome": "success",
        }
        message = parse_message(data)
        assert isinstance(message, HookEventMessage)
        assert message.subtype == "hook_response"
        assert message.hook_event_name == "PostToolUse"
        assert message.session_id == "sess-123"
        assert message.uuid == "uuid-789"
        assert message.data["output"] == ""
        assert message.data["exit_code"] == 0
        assert message.data["outcome"] == "success"

    def test_parse_hook_event_message_isinstance_system(self):
        """HookEventMessage is a SystemMessage subclass for backward compat."""
        data = {"type": "system", "subtype": "hook_started", "hook_event": "PreToolUse"}
        message = parse_message(data)
        assert isinstance(message, HookEventMessage)
        assert isinstance(message, SystemMessage)

    def test_parse_hook_event_message_minimal(self):
        """Hook events without session_id/uuid/hook_event still parse."""
        data = {"type": "system", "subtype": "hook_started", "hook_name": "Stop"}
        message = parse_message(data)
        assert isinstance(message, HookEventMessage)
        assert message.subtype == "hook_started"
        assert message.hook_event_name == "Stop"
        assert message.session_id is None
        assert message.uuid is None


# ---------------------------------------------------------------------------
# Message model fidelity: typed system subtypes, new top-level message types,
# and content blocks that are never dropped.
# ---------------------------------------------------------------------------

# The ``init`` frame as emitted by current Claude Code CLIs (wire spellings).
REALISTIC_INIT_FRAME: dict[str, Any] = {
    "type": "system",
    "subtype": "init",
    "cwd": "/home/dev/project",
    "session_id": "8f1c2a4e-5b6d-4c7e-9f0a-1b2c3d4e5f60",
    "tools": ["Task", "Bash", "Glob", "Grep", "Read", "Edit", "Write", "WebFetch"],
    "mcp_servers": [{"name": "filesystem", "status": "connected"}],
    "model": "claude-sonnet-4-5-20250929",
    "permissionMode": "acceptEdits",
    "slash_commands": ["compact", "context", "cost", "review"],
    "apiKeySource": "none",
    "claude_code_version": "2.1.283",
    "output_style": "default",
    "agents": ["general-purpose", "Explore", "Plan"],
    "skills": ["commit", "pr-review"],
    "plugins": [{"name": "my-plugin", "path": "/home/dev/.claude/plugins/my-plugin"}],
    "betas": ["context-1m-2025-08-07"],
    "uuid": "0d1e2f3a-4b5c-6d7e-8f90-a1b2c3d4e5f6",
}

# The minimal ``init`` frame the fake CLIs in the test-suite emit; they later
# read ``init.data[...]`` so the raw frame must stay attached.
MINIMAL_INIT_FRAME: dict[str, Any] = {
    "type": "system",
    "subtype": "init",
    "session_id": "s",
    "model": "m",
    "cwd": ".",
    "tools": [],
    "mcp_servers": [],
    "permissionMode": "default",
    "apiKeySource": "none",
    "fake_env": {"CLAUDE_CODE_SDK_READS_SESSION_STATE": None},
}

TOOL_PROGRESS_FRAME: dict[str, Any] = {
    "type": "tool_progress",
    "tool_use_id": "toolu_01ABC",
    "tool_name": "Bash",
    "parent_tool_use_id": None,
    "elapsed_time_seconds": 12.5,
    "uuid": "tp-uuid-1",
    "session_id": "sess-1",
}


def _assistant_frame(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {"content": list(blocks), "model": "claude-sonnet-4-5"},
    }


def _user_frame(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"type": "user", "message": {"content": list(blocks)}}


class TestInitMessage:
    """``system``/``init`` frames parse into a typed InitMessage."""

    def test_parse_realistic_init_frame(self):
        message = parse_message(dict(REALISTIC_INIT_FRAME))
        assert isinstance(message, InitMessage)
        assert message.subtype == "init"
        assert message.data == REALISTIC_INIT_FRAME
        assert message.session_id == "8f1c2a4e-5b6d-4c7e-9f0a-1b2c3d4e5f60"
        assert message.model == "claude-sonnet-4-5-20250929"
        assert message.cwd == "/home/dev/project"
        assert message.tools == [
            "Task",
            "Bash",
            "Glob",
            "Grep",
            "Read",
            "Edit",
            "Write",
            "WebFetch",
        ]
        assert message.mcp_servers == [{"name": "filesystem", "status": "connected"}]
        assert message.permission_mode == "acceptEdits"
        assert message.api_key_source == "none"
        assert message.slash_commands == ["compact", "context", "cost", "review"]
        assert message.agents == ["general-purpose", "Explore", "Plan"]
        assert message.skills == ["commit", "pr-review"]
        assert message.plugins == [
            {"name": "my-plugin", "path": "/home/dev/.claude/plugins/my-plugin"}
        ]
        assert message.output_style == "default"
        assert message.claude_code_version == "2.1.283"
        assert message.betas == ["context-1m-2025-08-07"]
        assert message.uuid == "0d1e2f3a-4b5c-6d7e-8f90-a1b2c3d4e5f6"

    def test_parse_init_frame_snake_case_spellings(self):
        """permission_mode / api_key_source are accepted alongside the camelCase wire keys."""
        frame = {
            key: value
            for key, value in REALISTIC_INIT_FRAME.items()
            if key not in ("permissionMode", "apiKeySource")
        }
        frame["permission_mode"] = "plan"
        frame["api_key_source"] = "user"
        message = parse_message(frame)
        assert isinstance(message, InitMessage)
        assert message.permission_mode == "plan"
        assert message.api_key_source == "user"

    def test_camel_case_spelling_wins_when_both_present(self):
        frame = {**REALISTIC_INIT_FRAME, "permission_mode": "plan"}
        message = parse_message(frame)
        assert isinstance(message, InitMessage)
        assert message.permission_mode == "acceptEdits"

    def test_parse_minimal_init_frame_from_fake_cli(self):
        """The few-key frame the fake CLIs emit parses, keeping the raw frame on .data."""
        message = parse_message(dict(MINIMAL_INIT_FRAME))
        assert isinstance(message, InitMessage)
        assert message.session_id == "s"
        assert message.model == "m"
        assert message.cwd == "."
        assert message.tools == []
        assert message.mcp_servers == []
        assert message.permission_mode == "default"
        assert message.api_key_source == "none"
        assert message.slash_commands == []
        assert message.agents == []
        assert message.skills == []
        assert message.plugins == []
        assert message.betas == []
        assert message.output_style is None
        assert message.claude_code_version is None
        assert message.uuid is None
        assert message.data["fake_env"] == {"CLAUDE_CODE_SDK_READS_SESSION_STATE": None}

    def test_parse_bare_init_frame_uses_defaults(self):
        """An init frame with no keys beyond type/subtype never raises."""
        data = {"type": "system", "subtype": "init"}
        message = parse_message(data)
        assert isinstance(message, InitMessage)
        assert message.data == data
        assert message.session_id is None
        assert message.model is None
        assert message.cwd is None
        assert message.permission_mode is None
        assert message.api_key_source is None
        assert message.tools == []
        assert message.mcp_servers == []
        assert message.slash_commands == []
        assert message.agents == []
        assert message.skills == []
        assert message.plugins == []
        assert message.betas == []

    @pytest.mark.parametrize("bad", [None, "Bash", 3, {"name": "Bash"}])
    def test_non_list_collections_fall_back_to_empty(self, bad):
        frame = {
            "type": "system",
            "subtype": "init",
            "tools": bad,
            "mcp_servers": bad,
            "slash_commands": bad,
            "agents": bad,
            "skills": bad,
            "plugins": bad,
            "betas": bad,
        }
        message = parse_message(frame)
        assert isinstance(message, InitMessage)
        assert message.tools == []
        assert message.mcp_servers == []
        assert message.slash_commands == []
        assert message.agents == []
        assert message.skills == []
        assert message.plugins == []
        assert message.betas == []
        # The raw value is still visible on the frame.
        assert message.data["tools"] == bad

    @pytest.mark.parametrize("bad", [None, 7, ["default"]])
    def test_non_string_scalars_fall_back_to_none(self, bad):
        frame = {
            "type": "system",
            "subtype": "init",
            "session_id": bad,
            "model": bad,
            "cwd": bad,
            "permissionMode": bad,
            "apiKeySource": bad,
            "output_style": bad,
            "claude_code_version": bad,
            "uuid": bad,
        }
        message = parse_message(frame)
        assert isinstance(message, InitMessage)
        assert message.session_id is None
        assert message.model is None
        assert message.cwd is None
        assert message.permission_mode is None
        assert message.api_key_source is None
        assert message.output_style is None
        assert message.claude_code_version is None
        assert message.uuid is None

    def test_agents_descriptor_dicts_pass_through(self):
        """Some CLI versions send agent descriptor objects instead of names."""
        agents = [{"name": "reviewer", "description": "Reviews PRs"}]
        frame = {"type": "system", "subtype": "init", "agents": agents}
        message = parse_message(frame)
        assert isinstance(message, InitMessage)
        assert message.agents == agents

    def test_init_message_is_system_message(self):
        message = parse_message(dict(REALISTIC_INIT_FRAME))
        assert isinstance(message, SystemMessage)
        matched = False
        match message:
            case SystemMessage():
                matched = True
        assert matched


class TestCompactBoundaryMessage:
    """``system``/``compact_boundary`` frames parse into CompactBoundaryMessage."""

    def test_parse_compact_boundary_with_metadata(self):
        data = {
            "type": "system",
            "subtype": "compact_boundary",
            "compact_metadata": {"trigger": "auto", "pre_tokens": 154_321},
            "uuid": "cb-uuid-1",
            "session_id": "sess-1",
        }
        message = parse_message(data)
        assert isinstance(message, CompactBoundaryMessage)
        assert message.subtype == "compact_boundary"
        assert message.trigger == "auto"
        assert message.pre_tokens == 154_321
        assert message.session_id == "sess-1"
        assert message.uuid == "cb-uuid-1"
        assert message.data == data

    def test_parse_compact_boundary_manual_trigger(self):
        data = {
            "type": "system",
            "subtype": "compact_boundary",
            "compact_metadata": {"trigger": "manual", "pre_tokens": 42},
            "session_id": "sess-1",
        }
        message = parse_message(data)
        assert isinstance(message, CompactBoundaryMessage)
        assert message.trigger == "manual"
        assert message.pre_tokens == 42
        assert message.uuid is None

    def test_parse_compact_boundary_without_metadata(self):
        data = {"type": "system", "subtype": "compact_boundary", "session_id": "s"}
        message = parse_message(data)
        assert isinstance(message, CompactBoundaryMessage)
        assert message.trigger is None
        assert message.pre_tokens is None
        assert message.session_id == "s"
        assert message.uuid is None
        assert message.data == data

    @pytest.mark.parametrize(
        "metadata",
        [
            None,
            "auto",
            ["auto"],
            {"trigger": 1, "pre_tokens": "many"},
            {"pre_tokens": True},
        ],
    )
    def test_malformed_metadata_never_raises(self, metadata):
        data = {
            "type": "system",
            "subtype": "compact_boundary",
            "compact_metadata": metadata,
        }
        message = parse_message(data)
        assert isinstance(message, CompactBoundaryMessage)
        assert message.trigger is None
        assert message.pre_tokens is None
        assert message.data["compact_metadata"] == metadata

    def test_compact_boundary_is_system_message(self):
        message = parse_message({"type": "system", "subtype": "compact_boundary"})
        assert isinstance(message, SystemMessage)
        matched = False
        match message:
            case SystemMessage():
                matched = True
        assert matched


class TestStatusMessage:
    """``system``/``status`` frames parse into StatusMessage."""

    def test_parse_status_compacting(self):
        data = {
            "type": "system",
            "subtype": "status",
            "status": "compacting",
            "uuid": "st-uuid-1",
            "session_id": "sess-1",
        }
        message = parse_message(data)
        assert isinstance(message, StatusMessage)
        assert message.subtype == "status"
        assert message.status == "compacting"
        assert message.permission_mode is None
        assert message.uuid == "st-uuid-1"
        assert message.session_id == "sess-1"
        assert message.data == data

    def test_parse_status_cleared(self):
        """The CLI sends status=null once compaction finishes."""
        data = {
            "type": "system",
            "subtype": "status",
            "status": None,
            "uuid": "st-uuid-2",
            "session_id": "sess-1",
        }
        message = parse_message(data)
        assert isinstance(message, StatusMessage)
        assert message.status is None
        assert message.data == data

    @pytest.mark.parametrize("key", ["permissionMode", "permission_mode"])
    def test_parse_status_permission_mode_spellings(self, key):
        data = {
            "type": "system",
            "subtype": "status",
            "status": None,
            key: "plan",
            "session_id": "sess-1",
        }
        message = parse_message(data)
        assert isinstance(message, StatusMessage)
        assert message.permission_mode == "plan"

    def test_parse_status_minimal(self):
        message = parse_message({"type": "system", "subtype": "status"})
        assert isinstance(message, StatusMessage)
        assert message.status is None
        assert message.permission_mode is None
        assert message.session_id is None
        assert message.uuid is None

    def test_status_is_system_message(self):
        message = parse_message({"type": "system", "subtype": "status"})
        assert isinstance(message, SystemMessage)
        matched = False
        match message:
            case SystemMessage():
                matched = True
        assert matched


class TestToolProgressMessage:
    """Top-level ``tool_progress`` frames parse into ToolProgressMessage."""

    def test_parse_tool_progress(self):
        message = parse_message(dict(TOOL_PROGRESS_FRAME))
        assert isinstance(message, ToolProgressMessage)
        assert message.tool_use_id == "toolu_01ABC"
        assert message.tool_name == "Bash"
        assert message.elapsed_time_seconds == 12.5
        assert message.parent_tool_use_id is None
        assert message.uuid == "tp-uuid-1"
        assert message.session_id == "sess-1"
        assert message.data == TOOL_PROGRESS_FRAME

    def test_parse_tool_progress_inside_subagent(self):
        frame = {**TOOL_PROGRESS_FRAME, "parent_tool_use_id": "toolu_parent"}
        message = parse_message(frame)
        assert isinstance(message, ToolProgressMessage)
        assert message.parent_tool_use_id == "toolu_parent"

    def test_integer_elapsed_time_becomes_float(self):
        frame = {**TOOL_PROGRESS_FRAME, "elapsed_time_seconds": 3}
        message = parse_message(frame)
        assert isinstance(message, ToolProgressMessage)
        assert message.elapsed_time_seconds == 3.0
        assert isinstance(message.elapsed_time_seconds, float)

    def test_tool_progress_without_optional_ids(self):
        frame = {
            "type": "tool_progress",
            "tool_use_id": "toolu_01ABC",
            "tool_name": "Read",
            "elapsed_time_seconds": 0.25,
        }
        message = parse_message(frame)
        assert isinstance(message, ToolProgressMessage)
        assert message.parent_tool_use_id is None
        assert message.uuid is None
        assert message.session_id is None

    def test_tool_progress_without_tool_use_id_is_skipped(self):
        frame = {k: v for k, v in TOOL_PROGRESS_FRAME.items() if k != "tool_use_id"}
        assert parse_message(frame) is None

    def test_tool_progress_without_tool_name_is_skipped(self):
        frame = {k: v for k, v in TOOL_PROGRESS_FRAME.items() if k != "tool_name"}
        assert parse_message(frame) is None

    @pytest.mark.parametrize("elapsed", ["12.5", None, True])
    def test_tool_progress_with_non_numeric_elapsed_is_skipped(self, elapsed):
        frame = {**TOOL_PROGRESS_FRAME, "elapsed_time_seconds": elapsed}
        assert parse_message(frame) is None

    def test_tool_progress_missing_elapsed_is_skipped(self):
        frame = {
            k: v for k, v in TOOL_PROGRESS_FRAME.items() if k != "elapsed_time_seconds"
        }
        assert parse_message(frame) is None


class TestToolUseSummaryMessage:
    """Top-level ``tool_use_summary`` frames parse into ToolUseSummaryMessage."""

    def test_parse_tool_use_summary(self):
        data = {
            "type": "tool_use_summary",
            "summary": "Read 3 files and ran the test suite",
            "preceding_tool_use_ids": ["toolu_01", "toolu_02", "toolu_03"],
            "uuid": "ts-uuid-1",
            "session_id": "sess-1",
        }
        message = parse_message(data)
        assert isinstance(message, ToolUseSummaryMessage)
        assert message.summary == "Read 3 files and ran the test suite"
        assert message.preceding_tool_use_ids == ["toolu_01", "toolu_02", "toolu_03"]
        assert message.uuid == "ts-uuid-1"
        assert message.session_id == "sess-1"
        assert message.data == data

    def test_tool_use_summary_without_preceding_ids_defaults_empty(self):
        data = {"type": "tool_use_summary", "summary": "Edited one file"}
        message = parse_message(data)
        assert isinstance(message, ToolUseSummaryMessage)
        assert message.preceding_tool_use_ids == []
        assert message.uuid is None
        assert message.session_id is None

    def test_tool_use_summary_non_list_preceding_ids_defaults_empty(self):
        data = {
            "type": "tool_use_summary",
            "summary": "Edited one file",
            "preceding_tool_use_ids": "toolu_01",
        }
        message = parse_message(data)
        assert isinstance(message, ToolUseSummaryMessage)
        assert message.preceding_tool_use_ids == []
        assert message.data["preceding_tool_use_ids"] == "toolu_01"

    def test_tool_use_summary_without_summary_is_skipped(self):
        data = {"type": "tool_use_summary", "preceding_tool_use_ids": ["toolu_01"]}
        assert parse_message(data) is None

    @pytest.mark.parametrize("summary", [None, 42, ["Edited one file"]])
    def test_tool_use_summary_non_string_summary_is_skipped(self, summary):
        data = {"type": "tool_use_summary", "summary": summary}
        assert parse_message(data) is None


class TestAuthStatusMessage:
    """Top-level ``auth_status`` frames parse into AuthStatusMessage."""

    def test_parse_auth_status_in_progress(self):
        data = {
            "type": "auth_status",
            "isAuthenticating": True,
            "output": ["Opening browser for login...", "Waiting for callback"],
            "uuid": "as-uuid-1",
            "session_id": "sess-1",
        }
        message = parse_message(data)
        assert isinstance(message, AuthStatusMessage)
        assert message.is_authenticating is True
        assert message.output == [
            "Opening browser for login...",
            "Waiting for callback",
        ]
        assert message.error is None
        assert message.uuid == "as-uuid-1"
        assert message.session_id == "sess-1"
        assert message.data == data

    def test_parse_auth_status_failed(self):
        data = {
            "type": "auth_status",
            "isAuthenticating": False,
            "output": [],
            "error": "Login timed out",
            "uuid": "as-uuid-2",
            "session_id": "sess-1",
        }
        message = parse_message(data)
        assert isinstance(message, AuthStatusMessage)
        assert message.is_authenticating is False
        assert message.output == []
        assert message.error == "Login timed out"

    def test_parse_auth_status_snake_case(self):
        data = {"type": "auth_status", "is_authenticating": True, "output": []}
        message = parse_message(data)
        assert isinstance(message, AuthStatusMessage)
        assert message.is_authenticating is True

    def test_auth_status_missing_flag_defaults_false(self):
        data = {"type": "auth_status", "uuid": "as-uuid-3", "session_id": "sess-1"}
        message = parse_message(data)
        assert isinstance(message, AuthStatusMessage)
        assert message.is_authenticating is False
        assert message.output == []
        assert message.error is None
        assert message.data == data

    def test_auth_status_null_flag_defaults_false(self):
        data = {"type": "auth_status", "isAuthenticating": None}
        message = parse_message(data)
        assert isinstance(message, AuthStatusMessage)
        assert message.is_authenticating is False

    def test_auth_status_non_list_output_defaults_empty(self):
        data = {"type": "auth_status", "isAuthenticating": True, "output": "line"}
        message = parse_message(data)
        assert isinstance(message, AuthStatusMessage)
        assert message.output == []
        assert message.data["output"] == "line"


class TestNewContentBlocks:
    """redacted_thinking, optional thinking signatures, server tool results
    of every kind, and UnknownBlock for anything the SDK does not model."""

    def test_redacted_thinking_block(self):
        data = _assistant_frame(
            {"type": "redacted_thinking", "data": "EqQBCgIYAhIMopaque=="},
            {"type": "text", "text": "Done."},
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 2
        block = message.content[0]
        assert isinstance(block, RedactedThinkingBlock)
        assert block.data == "EqQBCgIYAhIMopaque=="
        assert isinstance(message.content[1], TextBlock)

    def test_thinking_without_signature_parses_with_empty_signature(self):
        data = _assistant_frame({"type": "thinking", "thinking": "Summarized plan"})
        try:
            message = parse_message(data)
        except MessageParseError as exc:
            pytest.fail(f"thinking without signature must parse, got {exc!r}")
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, ThinkingBlock)
        assert block.thinking == "Summarized plan"
        assert block.signature == ""

    def test_thinking_with_null_signature_parses_with_empty_signature(self):
        data = _assistant_frame(
            {"type": "thinking", "thinking": "Summarized plan", "signature": None}
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, ThinkingBlock)
        assert block.signature == ""

    def test_unknown_block_image_in_user_content(self):
        image = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw0KGgo=",
            },
        }
        data = _user_frame({"type": "text", "text": "What is in this picture?"}, image)
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        block = message.content[1]
        assert isinstance(block, UnknownBlock)
        assert block.type == "image"
        assert block.data == image

    def test_unknown_block_image_in_assistant_content(self):
        image = {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/chart.png"},
        }
        data = _assistant_frame(image, {"type": "text", "text": "Here is the chart"})
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 2
        block = message.content[0]
        assert isinstance(block, UnknownBlock)
        assert block.type == "image"
        assert block.data == image
        assert isinstance(message.content[1], TextBlock)

    @pytest.mark.parametrize("role", ["user", "assistant"])
    def test_unknown_block_future_type_preserved(self, role):
        future = {
            "type": "holographic_projection",
            "frames": 3,
            "payload": {"depth": [1, 2, 3]},
        }
        data = _user_frame(future) if role == "user" else _assistant_frame(future)
        message = parse_message(data)
        assert isinstance(message, UserMessage | AssistantMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, UnknownBlock)
        assert block.type == "holographic_projection"
        assert block.data == future
        assert block.data["payload"]["depth"] == [1, 2, 3]

    def test_unknown_blocks_keep_their_position_among_known_user_blocks(self):
        """text / tool_use / tool_result in user content stay typed; the rest is UnknownBlock."""
        data = _user_frame(
            {"type": "text", "text": "Look at this"},
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
            {"type": "tool_use", "id": "use_1", "name": "Read", "input": {"p": 1}},
            {"type": "tool_result", "tool_use_id": "use_1", "content": "ok"},
            {"type": "document", "title": "spec.pdf"},
        )
        message = parse_message(data)
        assert isinstance(message, UserMessage)
        assert isinstance(message.content, list)
        assert [type(block) for block in message.content] == [
            TextBlock,
            UnknownBlock,
            ToolUseBlock,
            ToolResultBlock,
            UnknownBlock,
        ]
        assert message.content[2].id == "use_1"
        assert message.content[3].tool_use_id == "use_1"
        assert message.content[4].type == "document"
        assert message.content[4].data == {"type": "document", "title": "spec.pdf"}

    @pytest.mark.parametrize("name", get_args(ServerToolName))
    def test_every_server_tool_result_type_maps_to_server_tool_result_block(self, name):
        block_type = f"{name}_tool_result"
        data = _assistant_frame(
            {
                "type": block_type,
                "tool_use_id": f"srvtoolu_{name}",
                "content": {"type": f"{name}_result", "value": 1},
            }
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, ServerToolResultBlock)
        assert block.tool_use_id == f"srvtoolu_{name}"
        assert block.content == {"type": f"{name}_result", "value": 1}

    @pytest.mark.parametrize(
        "block_type",
        [
            "advisor_tool_result",
            "web_search_tool_result",
            "web_fetch_tool_result",
            "code_execution_tool_result",
            "bash_code_execution_tool_result",
            "text_editor_code_execution_tool_result",
            "tool_search_tool_result",
        ],
    )
    def test_wire_server_tool_result_types(self, block_type):
        """The block type names the API actually emits all map to ServerToolResultBlock."""
        data = _assistant_frame(
            {
                "type": block_type,
                "tool_use_id": "srvtoolu_wire",
                "content": {"type": "result", "ok": True},
            }
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, ServerToolResultBlock)
        assert block.tool_use_id == "srvtoolu_wire"
        assert block.content == {"type": "result", "ok": True}

    def test_web_search_result_list_content(self):
        results = [
            {
                "type": "web_search_result",
                "title": "Anthropic",
                "url": "https://www.anthropic.com",
                "encrypted_content": "EpQBCg...",
                "page_age": None,
            },
            {
                "type": "web_search_result",
                "title": "Claude",
                "url": "https://claude.ai",
                "encrypted_content": "EqMBCg...",
                "page_age": "2 days ago",
            },
        ]
        data = _assistant_frame(
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_ws",
                "content": results,
            }
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, ServerToolResultBlock)
        assert isinstance(block.content, list)
        assert block.content == results

    def test_server_tool_result_error_object_content(self):
        data = _assistant_frame(
            {
                "type": "web_fetch_tool_result",
                "tool_use_id": "srvtoolu_wf",
                "content": {
                    "type": "web_fetch_tool_result_error",
                    "error_code": "url_not_accessible",
                },
            }
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, ServerToolResultBlock)
        assert block.content == {
            "type": "web_fetch_tool_result_error",
            "error_code": "url_not_accessible",
        }

    def test_server_tool_result_without_content_keeps_remaining_fields(self):
        data = _assistant_frame(
            {
                "type": "code_execution_tool_result",
                "tool_use_id": "srvtoolu_ce",
                "stdout": "42\n",
                "stderr": "",
                "return_code": 0,
            }
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, ServerToolResultBlock)
        assert block.tool_use_id == "srvtoolu_ce"
        assert block.content == {"stdout": "42\n", "stderr": "", "return_code": 0}

    def test_server_tool_result_string_content_is_kept(self):
        """A non-object content value is folded into the remaining-fields dict."""
        data = _assistant_frame(
            {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_bash",
                "content": "exit 1",
            }
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        block = message.content[0]
        assert isinstance(block, ServerToolResultBlock)
        assert block.content == {"content": "exit 1"}

    def test_future_server_tool_result_type_maps_to_server_tool_result_block(self):
        data = _assistant_frame(
            {
                "type": "memory_vault_tool_result",
                "tool_use_id": "srvtoolu_future",
                "content": {"type": "memory_vault_result", "hits": 2},
            }
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, ServerToolResultBlock)
        assert block.tool_use_id == "srvtoolu_future"
        assert block.content == {"type": "memory_vault_result", "hits": 2}

    def test_server_tool_result_without_tool_use_id_is_unknown_block(self):
        raw = {"type": "web_search_tool_result", "content": []}
        message = parse_message(_assistant_frame(raw))
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, UnknownBlock)
        assert block.type == "web_search_tool_result"
        assert block.data == raw

    def test_tool_result_in_assistant_content_stays_tool_result_block(self):
        data = _assistant_frame(
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, ToolResultBlock)
        assert not isinstance(block, ServerToolResultBlock)
        assert block.tool_use_id == "t1"

    def test_server_tool_use_block_still_parses(self):
        data = _assistant_frame(
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {"query": "claude agent sdk"},
            }
        )
        message = parse_message(data)
        assert isinstance(message, AssistantMessage)
        block = message.content[0]
        assert isinstance(block, ServerToolUseBlock)
        assert block.name == "web_search"


SYSTEM_SUBCLASS_FRAMES: list[tuple[dict[str, Any], type[SystemMessage]]] = [
    (dict(MINIMAL_INIT_FRAME), InitMessage),
    (
        {
            "type": "system",
            "subtype": "compact_boundary",
            "compact_metadata": {"trigger": "auto", "pre_tokens": 10},
        },
        CompactBoundaryMessage,
    ),
    ({"type": "system", "subtype": "status", "status": "compacting"}, StatusMessage),
    (
        {
            "type": "system",
            "subtype": "task_started",
            "task_id": "t1",
            "description": "d",
            "uuid": "u",
            "session_id": "s",
        },
        TaskStartedMessage,
    ),
    (
        {
            "type": "system",
            "subtype": "task_progress",
            "task_id": "t1",
            "description": "d",
            "usage": {"total_tokens": 1, "tool_uses": 0, "duration_ms": 1},
            "uuid": "u",
            "session_id": "s",
        },
        TaskProgressMessage,
    ),
    (
        {
            "type": "system",
            "subtype": "task_notification",
            "task_id": "t1",
            "status": "completed",
            "output_file": "/o",
            "summary": "s",
            "uuid": "u",
            "session_id": "s",
        },
        TaskNotificationMessage,
    ),
    (
        {
            "type": "system",
            "subtype": "task_updated",
            "task_id": "t1",
            "patch": {"status": "completed"},
        },
        TaskUpdatedMessage,
    ),
    (
        {"type": "system", "subtype": "hook_started", "hook_event": "PreToolUse"},
        HookEventMessage,
    ),
    (
        {"type": "system", "subtype": "hook_response", "hook_event": "PostToolUse"},
        HookEventMessage,
    ),
    (
        {
            "type": "system",
            "subtype": "mirror_error",
            "key": {"project_key": "p", "session_id": "s"},
            "error": "boom",
        },
        MirrorErrorMessage,
    ),
]


class TestParserInvariants:
    """Behaviour pinned by existing callers that the richer model must keep."""

    def test_unknown_system_subtype_is_exactly_system_message(self):
        data = {"type": "system", "subtype": "some_future_subtype", "session_id": "s"}
        message = parse_message(data)
        assert type(message) is SystemMessage
        assert not isinstance(message, InitMessage)
        assert not isinstance(message, CompactBoundaryMessage)
        assert not isinstance(message, StatusMessage)
        assert message.subtype == "some_future_subtype"
        assert message.data == data

    def test_system_frame_without_subtype_still_raises(self):
        with pytest.raises(MessageParseError) as exc_info:
            parse_message({"type": "system"})
        assert "Missing required field in system message" in str(exc_info.value)

    def test_unknown_top_level_type_still_returns_none(self):
        assert parse_message({"type": "telemetry_blob", "uuid": "u"}) is None

    @pytest.mark.parametrize(
        ("frame", "expected"),
        SYSTEM_SUBCLASS_FRAMES,
        ids=[frame["subtype"] for frame, _ in SYSTEM_SUBCLASS_FRAMES],
    )
    def test_every_system_subclass_is_a_system_message_with_raw_data(
        self, frame, expected
    ):
        message = parse_message(dict(frame))
        assert isinstance(message, expected)
        assert isinstance(message, SystemMessage)
        assert message.subtype == frame["subtype"]
        assert message.data == frame

    @pytest.mark.parametrize(
        ("frame", "prefix"),
        [
            ({"type": "user"}, "Missing required field in user message"),
            (
                {"type": "user", "message": {"content": [{"type": "text"}]}},
                "Missing required field in user message",
            ),
            (
                {"type": "user", "message": {"content": [{"text": "no type"}]}},
                "Missing required field in user message",
            ),
            ({"type": "assistant"}, "Missing required field in assistant message"),
            (
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "id": "x"}]},
                },
                "Missing required field in assistant message",
            ),
            (
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "redacted_thinking"}],
                        "model": "m",
                    },
                },
                "Missing required field in assistant message",
            ),
            (
                {"type": "result", "subtype": "success"},
                "Missing required field in result message",
            ),
        ],
    )
    def test_missing_field_frames_still_raise_with_role_prefix(self, frame, prefix):
        with pytest.raises(MessageParseError) as exc_info:
            parse_message(frame)
        assert prefix in str(exc_info.value)
        assert exc_info.value.data == frame

    def test_hook_events_still_route_before_generic_system_handling(self):
        data = {
            "type": "system",
            "subtype": "hook_started",
            "hook_event": "PreToolUse",
            "session_id": "s",
        }
        message = parse_message(data)
        assert isinstance(message, HookEventMessage)
        assert message.hook_event_name == "PreToolUse"
