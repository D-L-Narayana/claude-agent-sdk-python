"""Tests for the ``Query`` setup helpers shared by ``query()`` and
``ClaudeSDKClient`` (``claude_agent_sdk._internal.query_setup``)."""

import inspect
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    HookMatcher,
    InMemorySessionStore,
    MirrorErrorMessage,
    PermissionResultAllow,
    create_sdk_mcp_server,
    tool,
)
from claude_agent_sdk._internal.message_parser import parse_message
from claude_agent_sdk._internal.query import DEFAULT_RUN_END_CEILING_MS, Query
from claude_agent_sdk._internal.query_setup import (
    agents_to_wire,
    attach_session_store,
    initialize_timeout_seconds,
    query_kwargs_for,
    sdk_mcp_servers_from_options,
    system_prompt_init_fields,
)
from claude_agent_sdk._internal.session_resume import MaterializedResume
from claude_agent_sdk._internal.transcript_mirror_batcher import (
    MAX_PENDING_BYTES,
    MAX_PENDING_ENTRIES,
    TranscriptMirrorBatcher,
)
from claude_agent_sdk.types import _hooks_to_internal_format

SESSION = "550e8400-e29b-41d4-a716-446655440000"

_TIMEOUT_ENV = "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT"
_CEILING_ENV = "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"

# Every Query.__init__ keyword except ``transport`` / ``is_streaming_mode``,
# which the entry points supply themselves.
EXPECTED_QUERY_KWARGS = frozenset(
    {
        "can_use_tool",
        "hooks",
        "sdk_mcp_servers",
        "initialize_timeout",
        "agents",
        "exclude_dynamic_sections",
        "system_prompt_snapshot",
        "skills",
        "forward_subagent_text",
        "verbatim_prompts",
        "run_end_ceiling_ms",
    }
)


async def _hook(input_data, tool_use_id, context):
    return {}


async def _can_use_tool(tool_name, input_data, context):
    return PermissionResultAllow()


def _query_for_test() -> Query:
    transport = AsyncMock()
    transport.is_ready = Mock(return_value=True)
    return Query(transport=transport, is_streaming_mode=True)


class TestSdkMcpServersFromOptions:
    def test_no_servers(self):
        assert sdk_mcp_servers_from_options(ClaudeAgentOptions()) == {}

    @pytest.mark.parametrize("config", ["/path/to/mcp.json", Path("/p/mcp.json")])
    def test_path_configs_have_no_sdk_servers(self, config):
        assert (
            sdk_mcp_servers_from_options(ClaudeAgentOptions(mcp_servers=config)) == {}
        )

    def test_only_sdk_entries_are_extracted(self):
        instance = object()
        options = ClaudeAgentOptions(
            mcp_servers={
                "calc": {"type": "sdk", "name": "calc", "instance": instance},
                "ext": {"type": "stdio", "command": "echo"},
                "legacy": {"command": "echo"},
                "http": {"type": "http", "url": "https://example.com/mcp"},
            }
        )
        assert sdk_mcp_servers_from_options(options) == {"calc": instance}

    def test_create_sdk_mcp_server_instance_is_extracted(self):
        @tool("noop", "Does nothing", {})
        async def noop(args):
            return {"content": []}

        config = create_sdk_mcp_server("calc", tools=[noop])
        options = ClaudeAgentOptions(mcp_servers={"calc": config})
        servers = sdk_mcp_servers_from_options(options)
        assert list(servers) == ["calc"]
        assert servers["calc"] is config["instance"]


class TestSystemPromptInitFields:
    @pytest.mark.parametrize(
        ("system_prompt", "expected"),
        [
            (None, (None, None)),
            ("Be helpful", (None, None)),
            ({"type": "preset", "preset": "claude_code"}, (None, None)),
            (
                {
                    "type": "preset",
                    "preset": "claude_code",
                    "exclude_dynamic_sections": True,
                },
                (True, None),
            ),
            (
                {
                    "type": "preset",
                    "preset": "claude_code",
                    "exclude_dynamic_sections": False,
                    "snapshot": True,
                },
                (False, True),
            ),
            (
                {"type": "preset", "preset": "claude_code", "snapshot": False},
                (None, False),
            ),
            (
                {"type": "custom", "prompt": "Be helpful", "snapshot": False},
                (None, False),
            ),
            (
                {"type": "custom", "prompt": "Be helpful", "snapshot": True},
                (None, True),
            ),
            # exclude_dynamic_sections is a preset-only field.
            (
                {"type": "custom", "prompt": "x", "exclude_dynamic_sections": True},
                (None, None),
            ),
            # The file form carries neither field on the wire.
            ({"type": "file", "path": "/p.md", "snapshot": False}, (None, None)),
            # Non-bool values are ignored rather than forwarded.
            (
                {"type": "preset", "preset": "claude_code", "snapshot": "yes"},
                (None, None),
            ),
            (
                {
                    "type": "preset",
                    "preset": "claude_code",
                    "exclude_dynamic_sections": 1,
                },
                (None, None),
            ),
        ],
    )
    def test_fields(self, system_prompt, expected):
        options = ClaudeAgentOptions(system_prompt=system_prompt)
        assert system_prompt_init_fields(options) == expected


class TestAgentsToWire:
    def test_none_when_unset(self):
        assert agents_to_wire(ClaudeAgentOptions()) is None

    def test_none_when_empty(self):
        assert agents_to_wire(ClaudeAgentOptions(agents={})) is None

    def test_drops_only_none_fields(self):
        options = ClaudeAgentOptions(
            agents={
                "reviewer": AgentDefinition(
                    description="Reviews code",
                    prompt="Review carefully",
                    tools=["Read"],
                    model="sonnet",
                ),
                "minimal": AgentDefinition(description="d", prompt="p"),
            }
        )
        assert agents_to_wire(options) == {
            "reviewer": {
                "description": "Reviews code",
                "prompt": "Review carefully",
                "tools": ["Read"],
                "model": "sonnet",
            },
            "minimal": {"description": "d", "prompt": "p"},
        }

    def test_falsy_non_none_values_are_kept(self):
        options = ClaudeAgentOptions(
            agents={
                "bg": AgentDefinition(
                    description="d",
                    prompt="p",
                    tools=[],
                    maxTurns=0,
                    background=False,
                )
            }
        )
        assert agents_to_wire(options) == {
            "bg": {
                "description": "d",
                "prompt": "p",
                "tools": [],
                "maxTurns": 0,
                "background": False,
            }
        }


class TestInitializeTimeoutSeconds:
    def test_env_value_is_milliseconds(self):
        assert initialize_timeout_seconds({_TIMEOUT_ENV: "120000"}) == 120.0

    def test_default_is_sixty_seconds(self):
        assert initialize_timeout_seconds({}) == 60.0

    def test_floored_at_sixty_seconds(self):
        assert initialize_timeout_seconds({_TIMEOUT_ENV: "1000"}) == 60.0

    def test_reads_os_environ_by_default(self):
        with patch.dict(os.environ, {_TIMEOUT_ENV: "90000"}):
            assert initialize_timeout_seconds() == 90.0

    def test_os_environ_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_TIMEOUT_ENV, None)
            assert initialize_timeout_seconds() == 60.0


class TestQueryKwargsFor:
    def test_exact_key_set(self):
        assert set(query_kwargs_for(ClaudeAgentOptions())) == EXPECTED_QUERY_KWARGS

    def test_every_key_is_a_query_parameter(self):
        params = set(inspect.signature(Query.__init__).parameters)
        assert set(query_kwargs_for(ClaudeAgentOptions())) <= params - {
            "self",
            "transport",
            "is_streaming_mode",
        }

    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_TIMEOUT_ENV, None)
            os.environ.pop(_CEILING_ENV, None)
            kwargs = query_kwargs_for(ClaudeAgentOptions())
        assert kwargs == {
            "can_use_tool": None,
            "hooks": None,
            "sdk_mcp_servers": {},
            "initialize_timeout": 60.0,
            "agents": None,
            "exclude_dynamic_sections": None,
            "system_prompt_snapshot": None,
            "skills": None,
            "forward_subagent_text": False,
            "verbatim_prompts": False,
            "run_end_ceiling_ms": DEFAULT_RUN_END_CEILING_MS,
        }

    def test_values_follow_options(self):
        instance = object()
        hooks = {"PreToolUse": [HookMatcher(matcher="Bash", hooks=[_hook], timeout=5)]}
        agents = {"reviewer": AgentDefinition(description="d", prompt="p")}
        options = ClaudeAgentOptions(
            can_use_tool=_can_use_tool,
            hooks=hooks,
            mcp_servers={"calc": {"type": "sdk", "name": "calc", "instance": instance}},
            agents=agents,
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "exclude_dynamic_sections": True,
                "snapshot": False,
            },
            skills=["reviewer"],
            forward_subagent_text=True,
            verbatim_prompts=True,
            env={_CEILING_ENV: "4321"},
        )
        with patch.dict(os.environ, {_TIMEOUT_ENV: "120000"}):
            kwargs = query_kwargs_for(options)
        assert kwargs == {
            "can_use_tool": _can_use_tool,
            "hooks": _hooks_to_internal_format(hooks),
            "sdk_mcp_servers": {"calc": instance},
            "initialize_timeout": 120.0,
            "agents": {"reviewer": {"description": "d", "prompt": "p"}},
            "exclude_dynamic_sections": True,
            "system_prompt_snapshot": False,
            "skills": ["reviewer"],
            "forward_subagent_text": True,
            "verbatim_prompts": True,
            "run_end_ceiling_ms": 4321,
        }
        assert kwargs["hooks"] == {
            "PreToolUse": [{"matcher": "Bash", "hooks": [_hook], "timeout": 5}]
        }

    def test_skills_all_passthrough(self):
        assert query_kwargs_for(ClaudeAgentOptions(skills="all"))["skills"] == "all"

    def test_run_end_ceiling_prefers_options_env_over_process_env(self):
        with patch.dict(os.environ, {_CEILING_ENV: "1000"}):
            assert query_kwargs_for(ClaudeAgentOptions())["run_end_ceiling_ms"] == 1000
            kwargs = query_kwargs_for(ClaudeAgentOptions(env={_CEILING_ENV: "0"}))
        assert kwargs["run_end_ceiling_ms"] == 0

    @pytest.mark.anyio
    async def test_kwargs_construct_a_query(self):
        hooks = {"PreToolUse": [HookMatcher(matcher="Bash", hooks=[_hook])]}
        options = ClaudeAgentOptions(
            can_use_tool=_can_use_tool,
            hooks=hooks,
            agents={"reviewer": AgentDefinition(description="d", prompt="p")},
            system_prompt={"type": "custom", "prompt": "x", "snapshot": True},
            skills="all",
            forward_subagent_text=True,
            verbatim_prompts=True,
            env={_CEILING_ENV: "99"},
        )
        transport = AsyncMock()
        transport.is_ready = Mock(return_value=True)
        with patch.dict(os.environ, {_TIMEOUT_ENV: "75000"}):
            q = Query(
                transport=transport, is_streaming_mode=True, **query_kwargs_for(options)
            )
        assert q.transport is transport
        assert q.is_streaming_mode is True
        assert q.can_use_tool is _can_use_tool
        assert q.hooks == _hooks_to_internal_format(hooks)
        assert q.sdk_mcp_servers == {}
        assert q._initialize_timeout == 75.0
        assert q._agents == {"reviewer": {"description": "d", "prompt": "p"}}
        assert q._exclude_dynamic_sections is None
        assert q._system_prompt_snapshot is True
        assert q._skills == "all"
        assert q._forward_subagent_text is True
        assert q._verbatim_prompts is True
        assert q._run_end_ceiling_ms == 99
        q.close_receive_stream()


class TestAttachSessionStore:
    @pytest.mark.anyio
    async def test_no_store_is_a_no_op(self):
        q = _query_for_test()
        attach_session_store(q, ClaudeAgentOptions(), None)
        assert q._transcript_mirror_batcher is None
        q.close_receive_stream()

    @pytest.mark.anyio
    async def test_sets_a_batched_mirror_batcher(self, tmp_path):
        store = InMemorySessionStore()
        q = _query_for_test()
        options = ClaudeAgentOptions(
            session_store=store, env={"CLAUDE_CONFIG_DIR": str(tmp_path)}
        )
        attach_session_store(q, options, None)
        batcher = q._transcript_mirror_batcher
        assert isinstance(batcher, TranscriptMirrorBatcher)
        assert batcher.store is store
        assert batcher.projects_dir == str(tmp_path / "projects")
        assert batcher.max_pending_entries == MAX_PENDING_ENTRIES
        assert batcher.max_pending_bytes == MAX_PENDING_BYTES
        q.close_receive_stream()

    @pytest.mark.anyio
    async def test_eager_flush_mode_zeroes_thresholds(self, tmp_path):
        q = _query_for_test()
        options = ClaudeAgentOptions(
            session_store=InMemorySessionStore(),
            session_store_flush="eager",
            env={"CLAUDE_CONFIG_DIR": str(tmp_path)},
        )
        attach_session_store(q, options, None)
        batcher = q._transcript_mirror_batcher
        assert batcher is not None
        assert batcher.max_pending_entries == 0
        assert batcher.max_pending_bytes == 0
        q.close_receive_stream()

    @pytest.mark.anyio
    async def test_materialized_resume_points_at_the_temp_projects_dir(self, tmp_path):
        q = _query_for_test()
        materialized = MaterializedResume(
            config_dir=tmp_path / "resume",
            resume_session_id=SESSION,
            cleanup=AsyncMock(),
        )
        options = ClaudeAgentOptions(
            session_store=InMemorySessionStore(),
            resume=SESSION,
            env={"CLAUDE_CONFIG_DIR": str(tmp_path / "ignored")},
        )
        attach_session_store(q, options, materialized)
        batcher = q._transcript_mirror_batcher
        assert batcher is not None
        assert batcher.projects_dir == str(tmp_path / "resume" / "projects")
        q.close_receive_stream()

    @pytest.mark.anyio
    async def test_on_error_routes_to_report_mirror_error(self, tmp_path):
        q = _query_for_test()
        options = ClaudeAgentOptions(
            session_store=InMemorySessionStore(),
            env={"CLAUDE_CONFIG_DIR": str(tmp_path)},
        )
        attach_session_store(q, options, None)
        batcher = q._transcript_mirror_batcher
        assert batcher is not None

        key = {"project_key": "proj", "session_id": "sess"}
        await batcher.on_error(key, "disk full")

        frame = q._message_receive.receive_nowait()
        assert frame["type"] == "system"
        assert frame["subtype"] == "mirror_error"
        assert frame["error"] == "disk full"
        assert frame["key"] == key
        assert frame["session_id"] == "sess"
        message = parse_message(frame)
        assert isinstance(message, MirrorErrorMessage)
        assert message.key == key
        q.close_receive_stream()

    @pytest.mark.anyio
    async def test_dropped_batch_surfaces_as_mirror_error_frame(self, tmp_path):
        class FailingStore(InMemorySessionStore):
            async def append(self, key, entries):  # type: ignore[override]
                raise RuntimeError("disk full")

        q = _query_for_test()
        options = ClaudeAgentOptions(
            session_store=FailingStore(), env={"CLAUDE_CONFIG_DIR": str(tmp_path)}
        )
        attach_session_store(q, options, None)
        batcher = q._transcript_mirror_batcher
        assert batcher is not None

        file_path = str(tmp_path / "projects" / "proj" / "sess.jsonl")
        with patch(
            "claude_agent_sdk._internal.transcript_mirror_batcher.anyio.sleep",
            new=AsyncMock(),
        ):
            batcher.enqueue(file_path, [{"type": "user", "uuid": "u1"}])
            await batcher.flush()

        frame = q._message_receive.receive_nowait()
        assert frame["subtype"] == "mirror_error"
        assert "disk full" in frame["error"]
        assert frame["key"] == {"project_key": "proj", "session_id": "sess"}
        q.close_receive_stream()
