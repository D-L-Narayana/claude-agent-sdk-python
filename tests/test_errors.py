"""Tests for Claude SDK error handling."""

import copy
import pickle

import pytest

from claude_agent_sdk import (
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    ResultError,
)
from claude_agent_sdk._errors import ControlRequestError, ControlRequestTimeoutError


class _CustomTimeoutError(ControlRequestTimeoutError):
    """Module-level subclass so pickle can import it by qualified name."""


class TestErrorTypes:
    """Test error types and their properties."""

    def test_base_error(self):
        """Test base ClaudeSDKError."""
        error = ClaudeSDKError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert isinstance(error, Exception)

    def test_cli_not_found_error(self):
        """Test CLINotFoundError."""
        error = CLINotFoundError("Claude Code not found")
        assert isinstance(error, ClaudeSDKError)
        assert "Claude Code not found" in str(error)

    def test_connection_error(self):
        """Test CLIConnectionError."""
        error = CLIConnectionError("Failed to connect to CLI")
        assert isinstance(error, ClaudeSDKError)
        assert "Failed to connect to CLI" in str(error)

    def test_process_error(self):
        """Test ProcessError with exit code and stderr."""
        error = ProcessError("Process failed", exit_code=1, stderr="Command not found")
        assert error.exit_code == 1
        assert error.stderr == "Command not found"
        assert "Process failed" in str(error)
        assert "exit code: 1" in str(error)
        assert "Command not found" in str(error)

    def test_result_error_carries_payload(self):
        """ResultError is a ProcessError that exposes the result payload."""
        data = {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "errors": [],
            "result": "API Error: Stream idle timeout - no chunks received",
            "api_error_status": None,
            "terminal_reason": "api_error",
            "session_id": "s-1",
        }
        error = ResultError("Claude Code returned an error result: x", data, 1)
        assert isinstance(error, ProcessError)
        assert isinstance(error, ClaudeSDKError)
        assert error.exit_code == 1
        assert error.data is data
        assert error.subtype == "success"
        assert error.errors == []
        assert error.result == "API Error: Stream idle timeout - no chunks received"
        assert error.api_error_status is None
        assert error.terminal_reason == "api_error"
        assert error.session_id == "s-1"
        assert "exit code: 1" in str(error)

    def test_result_error_tolerates_missing_or_malformed_fields(self):
        error = ResultError("boom", {"errors": 42, "api_error_status": "500"})
        assert error.subtype is None
        assert error.errors == []
        assert error.result is None
        assert error.api_error_status is None
        assert error.terminal_reason is None
        assert error.session_id is None
        assert error.exit_code is None
        assert ResultError("boom").data == {}

    def test_result_error_normalizes_errors_like_the_message_text(self):
        """A bare-string `errors` is kept and blank entries are dropped, so the
        structured field agrees with the text the reader builds from it."""
        assert ResultError("m", {"errors": "boom"}).errors == ["boom"]
        assert ResultError("m", {"errors": [" ", "x ", 3]}).errors == ["x"]

    def test_result_error_survives_pickle_and_copy(self):
        """Exceptions cross process boundaries via pickle (multiprocessing,
        ProcessPoolExecutor); ResultError must round-trip with its payload."""
        import copy
        import pickle

        data = {"subtype": "error_max_turns", "errors": ["too many"], "session_id": "s"}
        error = ResultError("Claude Code returned an error result: too many", data, 1)
        for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error)):
            assert type(clone) is ResultError
            assert str(clone) == str(error)
            assert clone.exit_code == 1
            assert clone.subtype == "error_max_turns"
            assert clone.errors == ["too many"]
            assert clone.session_id == "s"
            assert clone.data == data

    def test_json_decode_error(self):
        """Test CLIJSONDecodeError."""
        import json

        try:
            json.loads("{invalid json}")
        except json.JSONDecodeError as e:
            error = CLIJSONDecodeError("{invalid json}", e)
            assert error.line == "{invalid json}"
            assert error.original_error == e
            assert "Failed to decode JSON" in str(error)


class TestControlRequestErrors:
    """Typed errors for the control protocol (timeouts, CLI error responses)."""

    def test_hierarchy(self):
        error = ControlRequestError("boom")
        assert isinstance(error, ClaudeSDKError)
        assert isinstance(error, Exception)
        # Not a process failure: the CLI is alive, it just refused the request.
        assert not isinstance(error, ProcessError)

        timeout = ControlRequestTimeoutError(
            "Control request timeout: interrupt", timeout=60.0
        )
        assert isinstance(timeout, ControlRequestError)
        assert isinstance(timeout, ClaudeSDKError)

    def test_attributes_default_to_none(self):
        error = ControlRequestError("boom")
        assert str(error) == "boom"
        assert error.args == ("boom",)
        assert error.subtype is None
        assert error.request_id is None

    def test_attributes(self):
        error = ControlRequestError(
            "Model 'nope' is not available",
            subtype="set_model",
            request_id="req_3_0a1b2c3d",
        )
        # The message is the CLI's error text, undecorated.
        assert str(error) == "Model 'nope' is not available"
        assert error.subtype == "set_model"
        assert error.request_id == "req_3_0a1b2c3d"

    def test_timeout_error_attributes(self):
        error = ControlRequestTimeoutError(
            "Control request timeout: initialize",
            timeout=60.0,
            subtype="initialize",
            request_id="req_1_deadbeef",
        )
        assert str(error) == "Control request timeout: initialize"
        assert error.timeout == 60.0
        assert error.subtype == "initialize"
        assert error.request_id == "req_1_deadbeef"

    def test_optional_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            ControlRequestError("boom", "set_model")  # type: ignore[misc]
        with pytest.raises(TypeError):
            ControlRequestTimeoutError("boom", 1.0)  # type: ignore[misc]
        with pytest.raises(TypeError):
            ControlRequestTimeoutError("boom")  # type: ignore[call-arg]

    def test_control_request_error_survives_pickle_and_copy(self):
        """Exceptions cross process boundaries via pickle (multiprocessing,
        ProcessPoolExecutor); the typed fields must round-trip."""
        error = ControlRequestError("boom", subtype="set_model", request_id="req_7")
        for clone in (
            pickle.loads(pickle.dumps(error)),
            copy.copy(error),
            copy.deepcopy(error),
        ):
            assert type(clone) is ControlRequestError
            assert str(clone) == "boom"
            assert clone.subtype == "set_model"
            assert clone.request_id == "req_7"

    def test_timeout_error_survives_pickle_and_copy(self):
        """``timeout`` is a required keyword, so the default exception
        reconstruction (``type(e)(*e.args)``) would not do; it must still
        round-trip with every field."""
        error = ControlRequestTimeoutError(
            "Control request timeout: set_model",
            timeout=12.5,
            subtype="set_model",
            request_id="req_8",
        )
        for clone in (
            pickle.loads(pickle.dumps(error)),
            copy.copy(error),
            copy.deepcopy(error),
        ):
            assert type(clone) is ControlRequestTimeoutError
            assert str(clone) == "Control request timeout: set_model"
            assert clone.timeout == 12.5
            assert clone.subtype == "set_model"
            assert clone.request_id == "req_8"

    def test_timeout_error_subclass_survives_pickle(self):
        error = _CustomTimeoutError("late", timeout=1.0, subtype="interrupt")
        clone = pickle.loads(pickle.dumps(error))
        assert type(clone) is _CustomTimeoutError
        assert str(clone) == "late"
        assert clone.timeout == 1.0
        assert clone.subtype == "interrupt"
        assert clone.request_id is None
