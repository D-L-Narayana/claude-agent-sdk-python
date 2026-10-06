"""The public API surface: every exported name resolves and is documented.

Classification note: several exports are *typing aliases*, not runtime
classes -- ``CanUseTool`` and ``HookCallback`` are
``collections.abc.Callable[[...], Awaitable[...]]``, ``PermissionMode`` is a
``Literal[...]``, ``Message`` and ``ContentBlock`` are unions. Such an alias
carries no docstring of its own. On Python 3.10, ``inspect.isclass`` returns
True for every ``types.GenericAlias`` -- ``Callable[...]``, ``dict[...]``,
``list[...]`` -- because the alias forwards attribute lookups, ``__class__``
included, to its origin; from Python 3.11 it returns False. Classifying by
``inspect.isclass`` alone therefore reported the two callback aliases as
undocumented classes on the 3.10 CI job while passing everywhere else.
``is_type_alias`` below recognizes aliases through
``typing.get_origin`` (and the ``GenericAlias`` / ``UnionType`` types), so the
documentation check applies to genuine runtime classes and functions on every
supported interpreter. ``TestAliasClassification`` needs only the standard
library and pins that behavior, including on Python 3.10.
"""

from __future__ import annotations

import collections.abc
import inspect
import types
import typing
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, Literal

import pytest

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

# Exported typing aliases: documented where they are defined, never classes.
EXPORTED_CALLABLE_ALIASES = ("CanUseTool", "HookCallback")
EXPORTED_OTHER_ALIASES = ("PermissionMode", "Message", "ContentBlock")
# Exported runtime objects the documentation check must keep examining.
EXPORTED_RUNTIME_OBJECTS = (
    "ClaudeSDKClient",
    "query",
    "ClaudeAgentOptions",
    "AbortSignal",
    "SQLiteSessionStore",
    "export_session_from_store",
)


def is_type_alias(obj: object) -> bool:
    """True for parameterized generics and unions: typing aliases, not classes.

    ``typing.get_origin`` is non-None for ``Callable[[...], ...]``,
    ``list[int]``, ``Literal[...]``, ``Optional[X]`` and ``X | Y`` on every
    supported Python; the isinstance checks cover the builtin alias types
    directly. A bare class, ``collections.abc.Callable`` itself included, has
    no origin and is not an alias.
    """
    return (
        isinstance(obj, (types.GenericAlias, types.UnionType))
        or typing.get_origin(obj) is not None
    )


def is_runtime_api_object(obj: object) -> bool:
    """A genuine class or function -- what a docstring can be required of."""
    return not is_type_alias(obj) and (inspect.isclass(obj) or inspect.isfunction(obj))


def undocumented_names(
    namespace: object, names: list[str] | tuple[str, ...]
) -> list[str]:
    """The ``names`` on ``namespace`` that are runtime API objects without a docstring."""
    undocumented: list[str] = []
    for name in names:
        obj = getattr(namespace, name)
        if is_runtime_api_object(obj) and not (inspect.getdoc(obj) or "").strip():
            undocumented.append(name)
    return undocumented


@pytest.fixture(scope="module")
def sdk() -> Any:
    """The package root, imported lazily so the standard-library-only tests in
    this module can run on an interpreter that lacks the SDK's dependencies
    (an ImportError here fails the SDK-backed tests; it never skips them)."""
    import claude_agent_sdk

    return claude_agent_sdk


class TestAliasClassification:
    """``is_type_alias`` separates typing aliases from runtime classes on every
    supported interpreter, without weakening the check for real classes."""

    @pytest.mark.parametrize(
        "alias",
        [
            pytest.param(
                Callable[[str, dict[str, Any]], Awaitable[str]], id="callable-alias"
            ),
            pytest.param(dict[str, Any], id="dict-alias"),
            pytest.param(list[int], id="list-alias"),
            pytest.param(Literal["a", "b"], id="literal-alias"),
            pytest.param(int | None, id="pep604-union"),
            pytest.param(typing.Optional[int], id="typing-optional"),  # noqa: UP045
        ],
    )
    def test_typing_aliases_are_not_runtime_api_objects(self, alias: object) -> None:
        assert is_type_alias(alias)
        assert not is_runtime_api_object(alias)
        # Whatever inspect.isclass says about the alias on this interpreter
        # (True for the Callable / dict / list aliases on 3.10, False on
        # 3.11+), it is never reported as an undocumented class.
        assert undocumented_names(SimpleNamespace(Alias=alias), ["Alias"]) == []

    def test_callable_alias_has_the_shape_the_3_10_job_tripped_on(self) -> None:
        """Pins the exact facts behind the Python 3.10 CI failure: a
        ``collections.abc.Callable[...]`` alias has no docstring of its own and
        its origin is ``collections.abc.Callable``; it must still not count as
        an undocumented class."""
        alias = Callable[[str, dict[str, Any]], Awaitable[str]]
        assert typing.get_origin(alias) is collections.abc.Callable
        assert not (inspect.getdoc(alias) or "").strip()
        assert not is_runtime_api_object(alias)

    def test_bare_classes_and_functions_are_runtime_api_objects(self) -> None:
        class Documented:
            """Has a docstring."""

        def documented() -> None:
            """Has a docstring."""

        for obj in (Documented, documented, dict, collections.abc.Callable):
            assert not is_type_alias(obj)
            assert is_runtime_api_object(obj)

    def test_undocumented_classes_and_functions_are_still_reported(self) -> None:
        """The repair narrows *what* is classified, not *whether* missing
        docstrings fail: a real class or function without one is reported,
        in ``__all__`` order, next to aliases and documented objects that are not."""

        class Undocumented:
            pass

        def undocumented() -> None:
            pass

        class Documented:
            """Has a docstring."""

        namespace = SimpleNamespace(
            Undocumented=Undocumented,
            Alias=Callable[[str], Awaitable[str]],
            Documented=Documented,
            undocumented=undocumented,
            Mode=Literal["x"],
        )
        names = ["Undocumented", "Alias", "Documented", "undocumented", "Mode"]
        assert undocumented_names(namespace, names) == ["Undocumented", "undocumented"]


def test_all_names_resolve(sdk: Any) -> None:
    missing = [name for name in sdk.__all__ if not hasattr(sdk, name)]
    assert missing == []


def test_all_has_no_duplicates(sdk: Any) -> None:
    assert len(sdk.__all__) == len(set(sdk.__all__))


@pytest.mark.parametrize("name", NEW_EXPORTS)
def test_new_symbols_are_exported(sdk: Any, name: str) -> None:
    assert name in sdk.__all__
    assert getattr(sdk, name) is not None


def test_public_classes_and_functions_have_docstrings(sdk: Any) -> None:
    assert undocumented_names(sdk, sdk.__all__) == []


def test_documentation_check_examines_the_runtime_api(sdk: Any) -> None:
    """The check is not vacuous: the exported classes and functions are
    classified as runtime objects and examined, the exported aliases are not."""
    runtime = {
        name for name in sdk.__all__ if is_runtime_api_object(getattr(sdk, name))
    }
    assert set(EXPORTED_RUNTIME_OBJECTS) <= runtime
    assert runtime.isdisjoint(EXPORTED_CALLABLE_ALIASES + EXPORTED_OTHER_ALIASES)
    # Most of the public surface is dataclasses, TypedDicts, exceptions and
    # functions; a classifier that excluded them would make the check hollow.
    assert len(runtime) >= 100, sorted(runtime)


@pytest.mark.parametrize("name", EXPORTED_CALLABLE_ALIASES)
def test_exported_callback_aliases_are_callable_aliases(sdk: Any, name: str) -> None:
    """``CanUseTool`` / ``HookCallback`` are ``Callable[[...], Awaitable[...]]``
    aliases documented at their definitions in ``types.py``; they are exactly
    the objects Python 3.10's ``inspect.isclass`` misreports as classes."""
    alias = getattr(sdk, name)
    assert typing.get_origin(alias) is collections.abc.Callable
    assert is_type_alias(alias)
    assert not is_runtime_api_object(alias)


def test_stores_subpackage_exports_sqlite_store(sdk: Any) -> None:
    from claude_agent_sdk import stores

    assert stores.__all__ == ["SQLiteSessionStore"]
    assert stores.SQLiteSessionStore is sdk.SQLiteSessionStore


def test_schema_helpers_keep_private_aliases(sdk: Any) -> None:
    from claude_agent_sdk._internal import _schema

    assert sdk._python_type_to_json_schema is _schema.python_type_to_json_schema
    assert sdk._typeddict_to_json_schema is _schema.typeddict_to_json_schema
    assert sdk._python_type_to_json_schema(str) == {"type": "string"}


def test_error_hierarchy(sdk: Any) -> None:
    assert issubclass(sdk.ControlRequestTimeoutError, sdk.ControlRequestError)
    assert issubclass(sdk.ControlRequestError, sdk.ClaudeSDKError)
    assert issubclass(sdk.ResultError, sdk.ProcessError)
