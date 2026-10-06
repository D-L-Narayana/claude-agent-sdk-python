"""The public API surface: every exported name resolves and is documented."""

from __future__ import annotations

import inspect

import pytest

import claude_agent_sdk

# Names added by this upgrade; each must be exported from the package root.
NEW_EXPORTS = [
    "AbortSignal",
    "ControlRequestError",
    "ControlRequestTimeoutError",
    "InitMessage",
    "CompactBoundaryMessage",
    "StatusMessage",
    "ToolProgressMessage",
    "ToolUseSummaryMessage",
    "AuthStatusMessage",
    "RedactedThinkingBlock",
    "UnknownBlock",
    "SQLiteSessionStore",
    "MirrorStats",
    "sync_session_to_store",
    "SessionSyncReport",
    "export_session_from_store",
    "to_sdk_message",
]


def test_all_names_resolve() -> None:
    missing = [
        name for name in claude_agent_sdk.__all__ if not hasattr(claude_agent_sdk, name)
    ]
    assert missing == []


def test_all_has_no_duplicates() -> None:
    assert len(claude_agent_sdk.__all__) == len(set(claude_agent_sdk.__all__))


@pytest.mark.parametrize("name", NEW_EXPORTS)
def test_new_symbols_are_exported(name: str) -> None:
    assert name in claude_agent_sdk.__all__
    assert getattr(claude_agent_sdk, name) is not None


def test_public_classes_and_functions_have_docstrings() -> None:
    undocumented = []
    for name in claude_agent_sdk.__all__:
        obj = getattr(claude_agent_sdk, name)
        is_api_object = inspect.isclass(obj) or inspect.isfunction(obj)
        if is_api_object and not (inspect.getdoc(obj) or "").strip():
            undocumented.append(name)
    assert undocumented == []


def test_stores_subpackage_exports_sqlite_store() -> None:
    from claude_agent_sdk import stores

    assert stores.__all__ == ["SQLiteSessionStore"]
    assert stores.SQLiteSessionStore is claude_agent_sdk.SQLiteSessionStore


def test_schema_helpers_keep_private_aliases() -> None:
    from claude_agent_sdk._internal import _schema

    assert (
        claude_agent_sdk._python_type_to_json_schema
        is _schema.python_type_to_json_schema
    )
    assert (
        claude_agent_sdk._typeddict_to_json_schema is _schema.typeddict_to_json_schema
    )
    assert claude_agent_sdk._python_type_to_json_schema(str) == {"type": "string"}


def test_error_hierarchy() -> None:
    assert issubclass(
        claude_agent_sdk.ControlRequestTimeoutError,
        claude_agent_sdk.ControlRequestError,
    )
    assert issubclass(
        claude_agent_sdk.ControlRequestError, claude_agent_sdk.ClaudeSDKError
    )
    assert issubclass(claude_agent_sdk.ResultError, claude_agent_sdk.ProcessError)
