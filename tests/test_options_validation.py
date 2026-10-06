"""Tests for fail-fast ``ClaudeAgentOptions`` validation.

``validate_options`` rejects option combinations the SDK cannot honor with a
``ValueError`` before any subprocess is spawned or any session store is
touched. It is called from both entry points (``query()`` and
``ClaudeSDKClient.connect()``) and nowhere else: constructing
``ClaudeAgentOptions`` directly, or building a CLI command from it, stays
permissive.
"""

import warnings
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    InMemorySessionStore,
    create_sdk_mcp_server,
    query,
)
from claude_agent_sdk._internal.options_validation import validate_options

SESSION = "550e8400-e29b-41d4-a716-446655440000"
OTHER_SESSION = "8f8b1c0e-2b1e-4a3f-9c2d-5e6f7a8b9c0d"
AT = "0d78eb23-2d48-4741-b970-4ed0a3356cce"
DROPS = "ce0a8011-2c8d-40f2-86e5-d6e1b0c041c0"


def _assert_valid(options: ClaudeAgentOptions) -> None:
    """validate_options accepts ``options`` without raising or warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validate_options(options) is None


class TestDefaults:
    def test_default_options_are_a_no_op(self):
        _assert_valid(ClaudeAgentOptions())

    def test_construction_never_validates(self):
        """The dataclass stays permissive; only the entry points validate."""
        options = ClaudeAgentOptions(continue_conversation=True, resume=SESSION)
        assert options.continue_conversation is True
        assert options.resume == SESSION


class TestResumeAndContinue:
    def test_resume_with_continue_conversation_rejected(self):
        with pytest.raises(ValueError, match="continue_conversation") as exc_info:
            validate_options(
                ClaudeAgentOptions(resume=SESSION, continue_conversation=True)
            )
        assert "resume" in str(exc_info.value)

    def test_resume_alone_ok(self):
        _assert_valid(ClaudeAgentOptions(resume=SESSION))

    def test_continue_conversation_alone_ok(self):
        _assert_valid(ClaudeAgentOptions(continue_conversation=True))


class TestSessionIdWithResume:
    def test_session_id_with_resume_rejected(self):
        with pytest.raises(ValueError, match="fork_session") as exc_info:
            validate_options(
                ClaudeAgentOptions(session_id=SESSION, resume=OTHER_SESSION)
            )
        assert "session_id" in str(exc_info.value)

    def test_session_id_with_continue_conversation_rejected(self):
        with pytest.raises(ValueError, match="fork_session"):
            validate_options(
                ClaudeAgentOptions(session_id=SESSION, continue_conversation=True)
            )

    def test_session_id_with_resume_and_fork_session_ok(self):
        _assert_valid(
            ClaudeAgentOptions(
                session_id=SESSION, resume=OTHER_SESSION, fork_session=True
            )
        )

    def test_session_id_with_continue_conversation_and_fork_session_ok(self):
        _assert_valid(
            ClaudeAgentOptions(
                session_id=SESSION, continue_conversation=True, fork_session=True
            )
        )

    def test_session_id_alone_ok(self):
        _assert_valid(ClaudeAgentOptions(session_id=SESSION))

    def test_fork_session_with_resume_ok(self):
        _assert_valid(ClaudeAgentOptions(resume=SESSION, fork_session=True))


class TestTruncatingResume:
    def test_resume_session_at_without_resume_rejected(self):
        with pytest.raises(ValueError, match="resume_session_at") as exc_info:
            validate_options(ClaudeAgentOptions(resume_session_at=AT))
        assert "resume" in str(exc_info.value)

    def test_resume_session_at_with_continue_but_no_resume_rejected(self):
        with pytest.raises(ValueError, match="resume_session_at"):
            validate_options(
                ClaudeAgentOptions(continue_conversation=True, resume_session_at=AT)
            )

    def test_resume_session_at_with_resume_ok(self):
        _assert_valid(ClaudeAgentOptions(resume=SESSION, resume_session_at=AT))

    def test_resume_drops_turn_without_resume_session_at_rejected(self):
        with pytest.raises(ValueError, match="resume_drops_turn") as exc_info:
            validate_options(
                ClaudeAgentOptions(resume=SESSION, resume_drops_turn=DROPS)
            )
        assert "resume_session_at" in str(exc_info.value)

    def test_empty_resume_drops_turn_without_resume_session_at_rejected(self):
        """``""`` is a value (``is not None``), so the guard applies to it too."""
        with pytest.raises(ValueError, match="resume_drops_turn"):
            validate_options(ClaudeAgentOptions(resume=SESSION, resume_drops_turn=""))

    def test_resume_drops_turn_with_resume_session_at_ok(self):
        _assert_valid(
            ClaudeAgentOptions(
                resume=SESSION, resume_session_at=AT, resume_drops_turn=DROPS
            )
        )

    def test_empty_resume_drops_turn_allowed_with_resume_session_at(self):
        """An empty declaration is forwarded to the CLI (which rejects it); the
        SDK must not reject it here and silently disarm the guard."""
        _assert_valid(
            ClaudeAgentOptions(
                resume=SESSION, resume_session_at=AT, resume_drops_turn=""
            )
        )


class TestLimits:
    @pytest.mark.parametrize("value", [-1, -100])
    def test_negative_max_turns_rejected(self, value):
        with pytest.raises(ValueError, match="max_turns") as exc_info:
            validate_options(ClaudeAgentOptions(max_turns=value))
        assert str(value) in str(exc_info.value)

    @pytest.mark.parametrize("value", [None, 0, 1, 50])
    def test_non_negative_max_turns_ok(self, value):
        _assert_valid(ClaudeAgentOptions(max_turns=value))

    @pytest.mark.parametrize("value", [-0.01, -5.0])
    def test_negative_max_budget_usd_rejected(self, value):
        with pytest.raises(ValueError, match="max_budget_usd") as exc_info:
            validate_options(ClaudeAgentOptions(max_budget_usd=value))
        assert str(value) in str(exc_info.value)

    @pytest.mark.parametrize("value", [None, 0.0, 0.0001, 2.5])
    def test_non_negative_max_budget_usd_ok(self, value):
        _assert_valid(ClaudeAgentOptions(max_budget_usd=value))


class TestMcpServers:
    def test_sdk_server_without_instance_rejected(self):
        with pytest.raises(ValueError, match="instance") as exc_info:
            validate_options(
                ClaudeAgentOptions(
                    mcp_servers={"calc": {"type": "sdk", "name": "calc"}}
                )
            )
        message = str(exc_info.value)
        assert "calc" in message
        assert "create_sdk_mcp_server" in message

    def test_sdk_server_with_none_instance_rejected(self):
        with pytest.raises(ValueError, match="instance"):
            validate_options(
                ClaudeAgentOptions(
                    mcp_servers={
                        "calc": {"type": "sdk", "name": "calc", "instance": None}
                    }
                )
            )

    def test_sdk_server_with_lowlevel_server_instance_ok(self):
        from mcp.server import Server

        _assert_valid(
            ClaudeAgentOptions(
                mcp_servers={"x": {"type": "sdk", "name": "x", "instance": Server("x")}}
            )
        )

    def test_create_sdk_mcp_server_config_ok(self):
        _assert_valid(
            ClaudeAgentOptions(mcp_servers={"calc": create_sdk_mcp_server("calc")})
        )

    def test_unusable_sdk_instance_rejected_before_spawn(self, monkeypatch):
        """An instance ``resolve_server`` refuses fails here, as a ValueError
        naming the entry, instead of at Query construction after the CLI
        subprocess has already been spawned."""
        from claude_agent_sdk._internal import sdk_mcp_bridge

        def _reject(instance):
            raise TypeError("boom")

        monkeypatch.setattr(sdk_mcp_bridge, "resolve_server", _reject)
        with pytest.raises(ValueError, match="boom") as exc_info:
            validate_options(
                ClaudeAgentOptions(
                    mcp_servers={
                        "calc": {"type": "sdk", "name": "calc", "instance": object()}
                    }
                )
            )
        assert "mcp_servers['calc']" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, TypeError)

    def test_each_sdk_instance_is_resolved_once(self, monkeypatch):
        from claude_agent_sdk._internal import sdk_mcp_bridge

        seen: list[object] = []

        def _spy(instance):
            seen.append(instance)
            return instance

        monkeypatch.setattr(sdk_mcp_bridge, "resolve_server", _spy)
        first, second = object(), object()
        _assert_valid(
            ClaudeAgentOptions(
                mcp_servers={
                    "a": {"type": "sdk", "name": "a", "instance": first},
                    "ext": {"type": "stdio", "command": "echo"},
                    "b": {"type": "sdk", "name": "b", "instance": second},
                }
            )
        )
        assert seen == [first, second]

    def test_non_sdk_entries_are_not_resolved(self, monkeypatch):
        from claude_agent_sdk._internal import sdk_mcp_bridge

        def _never(instance):
            raise AssertionError("resolve_server must only see sdk instances")

        monkeypatch.setattr(sdk_mcp_bridge, "resolve_server", _never)
        _assert_valid(
            ClaudeAgentOptions(
                mcp_servers={
                    "stdio": {"type": "stdio", "command": "echo", "args": ["hi"]},
                    "legacy": {"command": "echo"},
                    "sse": {"type": "sse", "url": "https://example.com/sse"},
                    "http": {"type": "http", "url": "https://example.com/mcp"},
                }
            )
        )
        _assert_valid(ClaudeAgentOptions(mcp_servers="/path/to/mcp.json"))
        _assert_valid(ClaudeAgentOptions(mcp_servers=Path("/path/to/mcp.json")))

    def test_external_servers_ok(self):
        _assert_valid(
            ClaudeAgentOptions(
                mcp_servers={
                    "stdio": {"type": "stdio", "command": "echo", "args": ["hi"]},
                    "legacy": {"command": "echo"},
                    "sse": {"type": "sse", "url": "https://example.com/sse"},
                    "http": {"type": "http", "url": "https://example.com/mcp"},
                }
            )
        )

    @pytest.mark.parametrize(
        "config", ["/path/to/mcp.json", Path("/path/to/mcp.json"), '{"mcpServers": {}}']
    )
    def test_path_and_json_string_configs_ok(self, config):
        _assert_valid(ClaudeAgentOptions(mcp_servers=config))


class TestPlugins:
    def test_non_local_plugin_rejected(self):
        with pytest.raises(ValueError, match="Unsupported plugin type") as exc_info:
            validate_options(
                ClaudeAgentOptions(plugins=[{"type": "remote", "path": "/p"}])
            )
        assert "remote" in str(exc_info.value)
        assert "local" in str(exc_info.value)

    def test_plugin_without_type_rejected(self):
        with pytest.raises(ValueError, match="Unsupported plugin type"):
            validate_options(ClaudeAgentOptions(plugins=[{"path": "/p"}]))

    def test_local_plugins_ok(self):
        _assert_valid(
            ClaudeAgentOptions(
                plugins=[
                    {"type": "local", "path": "/plugins/a"},
                    {"type": "local", "path": "/plugins/b"},
                ]
            )
        )


class TestThinkingDeprecation:
    def test_thinking_with_max_thinking_tokens_warns(self):
        with pytest.warns(DeprecationWarning, match="max_thinking_tokens") as record:
            validate_options(
                ClaudeAgentOptions(
                    thinking={"type": "adaptive"}, max_thinking_tokens=4096
                )
            )
        assert len(record) == 1
        assert "thinking" in str(record[0].message)

    def test_only_thinking_is_silent(self):
        _assert_valid(ClaudeAgentOptions(thinking={"type": "adaptive"}))

    def test_only_max_thinking_tokens_is_silent(self):
        _assert_valid(ClaudeAgentOptions(max_thinking_tokens=4096))

    def test_warning_is_attributed_to_the_caller_of_validate_options(self):
        """stacklevel=2: the warning registry key is the SDK entry point that
        calls validate_options (here: this test), not the validator itself."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            validate_options(
                ClaudeAgentOptions(thinking={"type": "disabled"}, max_thinking_tokens=0)
            )
        (warning,) = record
        assert warning.filename == __file__

    def test_warning_is_ignorable_via_filterwarnings(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            validate_options(
                ClaudeAgentOptions(
                    thinking={"type": "enabled", "budget_tokens": 1024},
                    max_thinking_tokens=1024,
                )
            )
        assert record == []


class _UntouchableStore(InMemorySessionStore):
    """A store whose read methods fail the test if the SDK reaches them."""

    async def load(self, key):  # type: ignore[override]
        raise AssertionError("store.load() must not run before validation")

    async def list_sessions(self, project_key):  # type: ignore[override]
        raise AssertionError("store.list_sessions() must not run before validation")


class _StopBeforeConnectError(Exception):
    """Sentinel raised from a transport to end the entry point early."""


def _failing_transport() -> AsyncMock:
    """Transport that aborts at connect(), after validation has run."""
    transport = AsyncMock()
    transport.connect = AsyncMock(side_effect=_StopBeforeConnectError)
    return transport


class TestQueryEntryPoint:
    """query() validates before constructing the transport or touching a store."""

    @pytest.mark.anyio
    async def test_query_raises_before_creating_transport(self):
        options = ClaudeAgentOptions(resume=SESSION, continue_conversation=True)
        with (
            patch(
                "claude_agent_sdk._internal.client.SubprocessCLITransport"
            ) as transport_class,
            pytest.raises(ValueError, match="continue_conversation"),
        ):
            async for _ in query(prompt="hi", options=options):
                pass  # pragma: no cover
        transport_class.assert_not_called()

    @pytest.mark.anyio
    async def test_query_raises_before_touching_the_store(self):
        options = ClaudeAgentOptions(
            resume=SESSION,
            continue_conversation=True,
            session_store=_UntouchableStore(),
        )
        with (
            patch(
                "claude_agent_sdk._internal.client.SubprocessCLITransport"
            ) as transport_class,
            pytest.raises(ValueError, match="continue_conversation"),
        ):
            async for _ in query(prompt="hi", options=options):
                pass  # pragma: no cover
        transport_class.assert_not_called()

    @pytest.mark.anyio
    async def test_query_rejects_each_rule_before_the_transport(self):
        invalid = [
            ClaudeAgentOptions(session_id=SESSION, resume=OTHER_SESSION),
            ClaudeAgentOptions(resume_session_at=AT),
            ClaudeAgentOptions(resume=SESSION, resume_drops_turn=DROPS),
            ClaudeAgentOptions(max_turns=-1),
            ClaudeAgentOptions(max_budget_usd=-1.0),
            ClaudeAgentOptions(mcp_servers={"s": {"type": "sdk", "name": "s"}}),
            ClaudeAgentOptions(plugins=[{"type": "remote", "path": "/p"}]),
        ]
        with patch(
            "claude_agent_sdk._internal.client.SubprocessCLITransport"
        ) as transport_class:
            for options in invalid:
                with pytest.raises(ValueError):
                    async for _ in query(prompt="hi", options=options):
                        pass  # pragma: no cover
        transport_class.assert_not_called()

    @pytest.mark.anyio
    async def test_query_rejects_unusable_sdk_instance_before_the_transport(
        self, monkeypatch
    ):
        from claude_agent_sdk._internal import sdk_mcp_bridge

        def _reject(instance):
            raise TypeError("boom")

        monkeypatch.setattr(sdk_mcp_bridge, "resolve_server", _reject)
        options = ClaudeAgentOptions(
            mcp_servers={"calc": {"type": "sdk", "name": "calc", "instance": object()}}
        )
        with (
            patch(
                "claude_agent_sdk._internal.client.SubprocessCLITransport"
            ) as transport_class,
            pytest.raises(ValueError, match="boom"),
        ):
            async for _ in query(prompt="hi", options=options):
                pass  # pragma: no cover
        transport_class.assert_not_called()

    @pytest.mark.anyio
    async def test_query_emits_thinking_deprecation_before_connect(self):
        options = ClaudeAgentOptions(
            thinking={"type": "adaptive"}, max_thinking_tokens=2048
        )
        with (
            pytest.warns(DeprecationWarning, match="max_thinking_tokens"),
            pytest.raises(_StopBeforeConnectError),
        ):
            async for _ in query(
                prompt="hi", options=options, transport=_failing_transport()
            ):
                pass  # pragma: no cover

    @pytest.mark.anyio
    async def test_query_with_custom_transport_still_validates(self):
        transport = _failing_transport()
        options = ClaudeAgentOptions(resume=SESSION, continue_conversation=True)
        with pytest.raises(ValueError, match="continue_conversation"):
            async for _ in query(prompt="hi", options=options, transport=transport):
                pass  # pragma: no cover
        transport.connect.assert_not_called()


class TestClientEntryPoint:
    """ClaudeSDKClient.connect() validates before constructing the transport."""

    @pytest.mark.anyio
    async def test_connect_raises_before_creating_transport(self):
        options = ClaudeAgentOptions(session_id=SESSION, resume=OTHER_SESSION)
        client = ClaudeSDKClient(options=options)
        with (
            patch(
                "claude_agent_sdk._internal.transport.subprocess_cli.SubprocessCLITransport"
            ) as transport_class,
            pytest.raises(ValueError, match="fork_session"),
        ):
            await client.connect()
        transport_class.assert_not_called()
        assert client._transport is None
        assert client._query is None
        # A failed connect leaves the client safely disconnectable.
        await client.disconnect()

    @pytest.mark.anyio
    async def test_context_manager_raises_before_creating_transport(self):
        options = ClaudeAgentOptions(resume=SESSION, continue_conversation=True)
        with (
            patch(
                "claude_agent_sdk._internal.transport.subprocess_cli.SubprocessCLITransport"
            ) as transport_class,
            pytest.raises(ValueError, match="continue_conversation"),
        ):
            async with ClaudeSDKClient(options=options):
                pass  # pragma: no cover
        transport_class.assert_not_called()

    @pytest.mark.anyio
    async def test_connect_raises_before_touching_the_store(self):
        options = ClaudeAgentOptions(
            resume=SESSION,
            continue_conversation=True,
            session_store=_UntouchableStore(),
        )
        with (
            patch(
                "claude_agent_sdk._internal.transport.subprocess_cli.SubprocessCLITransport"
            ) as transport_class,
            pytest.raises(ValueError, match="continue_conversation"),
        ):
            await ClaudeSDKClient(options=options).connect()
        transport_class.assert_not_called()

    @pytest.mark.anyio
    async def test_connect_emits_thinking_deprecation_before_connect(self):
        options = ClaudeAgentOptions(
            thinking={"type": "adaptive"}, max_thinking_tokens=2048
        )
        client = ClaudeSDKClient(options=options, transport=_failing_transport())
        with (
            pytest.warns(DeprecationWarning, match="max_thinking_tokens"),
            pytest.raises(_StopBeforeConnectError),
        ):
            await client.connect()

    @pytest.mark.anyio
    async def test_connect_with_custom_transport_still_validates(self):
        transport = _failing_transport()
        options = ClaudeAgentOptions(plugins=[{"type": "remote", "path": "/p"}])
        client = ClaudeSDKClient(options=options, transport=transport)
        with pytest.raises(ValueError, match="Unsupported plugin type"):
            await client.connect()
        transport.connect.assert_not_called()
