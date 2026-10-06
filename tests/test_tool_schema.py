"""Tests for the JSON Schema an SDK MCP tool's input types are turned into.

The conversion lives in ``claude_agent_sdk._internal._schema`` and is what
``@tool`` and ``create_sdk_mcp_server`` send to Claude Code as a tool's
``inputSchema`` (and validate arguments against). Every case here is a pure
function of a Python type, so the tests are synchronous.
"""

import ast
import datetime
import decimal
import enum
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, NewType, Optional, TypedDict, Union

import pytest
import typing_extensions

from claude_agent_sdk._internal import _schema
from claude_agent_sdk._internal._schema import (
    build_input_schema,
    python_type_to_json_schema,
    typeddict_to_json_schema,
)

STRING = {"type": "string"}
INTEGER = {"type": "integer"}
NUMBER = {"type": "number"}
BOOLEAN = {"type": "boolean"}
NULL = {"type": "null"}


class Color(str, enum.Enum):
    RED = "red"
    GREEN = "green"


class Priority(enum.IntEnum):
    LOW = 1
    HIGH = 2


class Mixed(enum.Enum):
    NAME = "a"
    NUMBER = 1


class Address(TypedDict):
    street: str
    city: Annotated[str, "City name"]


ADDRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "street": STRING,
        "city": {"type": "string", "description": "City name"},
    },
    "required": ["city", "street"],
}


# The common 3.10-compatible spelling: TypedDict and NotRequired both from
# typing_extensions (on 3.10 the stdlib TypedDict does not know NotRequired).
class Person(typing_extensions.TypedDict):
    name: str
    address: Address
    previous: list[Address]
    nickname: typing_extensions.NotRequired[str | None]


class Loose(TypedDict, total=False):
    anything: str


class Settings(typing_extensions.TypedDict):
    key: typing_extensions.ReadOnly[str]
    retries: typing_extensions.ReadOnly[typing_extensions.NotRequired[int]]
    mode: typing_extensions.Required[Annotated[str, "Operating mode"]]


class Opaque:
    pass


UserId = NewType("UserId", int)


# --- python_type_to_json_schema --------------------------------------------------


@pytest.mark.parametrize(
    ("py_type", "expected"),
    [
        pytest.param(str, STRING, id="str"),
        pytest.param(int, INTEGER, id="int"),
        pytest.param(float, NUMBER, id="float"),
        pytest.param(bool, BOOLEAN, id="bool"),
        pytest.param(type(None), NULL, id="NoneType"),
        pytest.param(None, NULL, id="None"),
        pytest.param(Any, {}, id="Any-accepts-anything"),
        pytest.param(object, {}, id="object-accepts-anything"),
        pytest.param(Opaque, STRING, id="unknown-class-falls-back-to-string"),
        pytest.param(bytes, STRING, id="bytes-falls-back-to-string"),
        pytest.param(UserId, INTEGER, id="NewType-uses-its-supertype"),
    ],
)
def test_scalars(py_type: Any, expected: dict[str, Any]) -> None:
    assert python_type_to_json_schema(py_type) == expected


@pytest.mark.parametrize(
    ("py_type", "expected"),
    [
        pytest.param(
            datetime.datetime,
            {"type": "string", "format": "date-time"},
            id="datetime",
        ),
        pytest.param(datetime.date, {"type": "string", "format": "date"}, id="date"),
        pytest.param(datetime.time, {"type": "string", "format": "time"}, id="time"),
        pytest.param(uuid.UUID, {"type": "string", "format": "uuid"}, id="uuid"),
        pytest.param(decimal.Decimal, NUMBER, id="decimal"),
    ],
)
def test_formatted_scalars(py_type: Any, expected: dict[str, Any]) -> None:
    assert python_type_to_json_schema(py_type) == expected


@pytest.mark.parametrize(
    ("py_type", "expected"),
    [
        pytest.param(list, {"type": "array"}, id="bare-list"),
        pytest.param(list[str], {"type": "array", "items": STRING}, id="list[str]"),
        pytest.param(
            list[list[int]],
            {"type": "array", "items": {"type": "array", "items": INTEGER}},
            id="nested-list",
        ),
        pytest.param(
            Sequence[int], {"type": "array", "items": INTEGER}, id="Sequence[int]"
        ),
        pytest.param(tuple, {"type": "array"}, id="bare-tuple"),
        pytest.param(
            tuple[int, ...], {"type": "array", "items": INTEGER}, id="tuple[int,...]"
        ),
        pytest.param(
            tuple[str, int],
            {
                "type": "array",
                "prefixItems": [STRING, INTEGER],
                "minItems": 2,
                "maxItems": 2,
            },
            id="fixed-tuple",
        ),
        pytest.param(tuple[()], {"type": "array", "maxItems": 0}, id="empty-tuple"),
        pytest.param(set, {"type": "array", "uniqueItems": True}, id="bare-set"),
        pytest.param(
            set[str],
            {"type": "array", "items": STRING, "uniqueItems": True},
            id="set[str]",
        ),
        pytest.param(
            frozenset[int],
            {"type": "array", "items": INTEGER, "uniqueItems": True},
            id="frozenset[int]",
        ),
    ],
)
def test_arrays(py_type: Any, expected: dict[str, Any]) -> None:
    assert python_type_to_json_schema(py_type) == expected


@pytest.mark.parametrize(
    ("py_type", "expected"),
    [
        pytest.param(dict, {"type": "object"}, id="bare-dict"),
        pytest.param(dict[str, Any], {"type": "object"}, id="dict[str,Any]"),
        pytest.param(dict[str, object], {"type": "object"}, id="dict[str,object]"),
        pytest.param(
            dict[str, int],
            {"type": "object", "additionalProperties": INTEGER},
            id="dict[str,int]",
        ),
        pytest.param(
            dict[str, list[str]],
            {
                "type": "object",
                "additionalProperties": {"type": "array", "items": STRING},
            },
            id="dict[str,list[str]]",
        ),
        pytest.param(
            Mapping[str, float],
            {"type": "object", "additionalProperties": NUMBER},
            id="Mapping[str,float]",
        ),
        pytest.param(
            dict[str, Address],
            {"type": "object", "additionalProperties": ADDRESS_SCHEMA},
            id="dict[str,TypedDict]",
        ),
    ],
)
def test_objects(py_type: Any, expected: dict[str, Any]) -> None:
    assert python_type_to_json_schema(py_type) == expected


@pytest.mark.parametrize(
    ("py_type", "expected"),
    [
        # A None argument must validate when the hint allows it, so the null
        # alternative is kept instead of being dropped from the union.
        pytest.param(str | None, {"anyOf": [STRING, NULL]}, id="str|None"),
        pytest.param(
            Optional[int],  # noqa: UP045 - the spelling under test
            {"anyOf": [INTEGER, NULL]},
            id="Optional[int]",
        ),
        pytest.param(str | int, {"anyOf": [STRING, INTEGER]}, id="str|int"),
        pytest.param(
            Union[str, int],  # noqa: UP007 - the spelling under test
            {"anyOf": [STRING, INTEGER]},
            id="Union[str,int]",
        ),
        pytest.param(
            str | int | None, {"anyOf": [STRING, INTEGER, NULL]}, id="str|int|None"
        ),
        pytest.param(
            list[str] | None,
            {"anyOf": [{"type": "array", "items": STRING}, NULL]},
            id="list[str]|None",
        ),
        pytest.param(
            Address | None, {"anyOf": [ADDRESS_SCHEMA, NULL]}, id="TypedDict|None"
        ),
        pytest.param(
            Literal["a", "b"] | None,
            {"anyOf": [{"type": "string", "enum": ["a", "b"]}, NULL]},
            id="Literal|None",
        ),
    ],
)
def test_unions_keep_every_alternative(py_type: Any, expected: dict[str, Any]) -> None:
    assert python_type_to_json_schema(py_type) == expected


@pytest.mark.parametrize(
    ("py_type", "expected"),
    [
        pytest.param(
            Literal["a", "b"], {"type": "string", "enum": ["a", "b"]}, id="str-literals"
        ),
        pytest.param(
            Literal[1, 2, 3], {"type": "integer", "enum": [1, 2, 3]}, id="ints"
        ),
        pytest.param(Literal[True], {"type": "boolean", "enum": [True]}, id="bool"),
        pytest.param(Literal[None], {"type": "null", "enum": [None]}, id="None"),
        pytest.param(Literal["a", 1], {"enum": ["a", 1]}, id="mixed-has-no-type"),
        pytest.param(
            Literal[Color.RED], {"type": "string", "enum": ["red"]}, id="enum-member"
        ),
        pytest.param(
            Color, {"type": "string", "enum": ["red", "green"]}, id="str-Enum"
        ),
        pytest.param(Priority, {"type": "integer", "enum": [1, 2]}, id="IntEnum"),
        pytest.param(Mixed, {"enum": ["a", 1]}, id="mixed-Enum-has-no-type"),
    ],
)
def test_literals_and_enums(py_type: Any, expected: dict[str, Any]) -> None:
    assert python_type_to_json_schema(py_type) == expected


@pytest.mark.parametrize(
    ("py_type", "expected"),
    [
        pytest.param(
            Annotated[str, "The query"],
            {"type": "string", "description": "The query"},
            id="top-level",
        ),
        pytest.param(Annotated[int, 42], INTEGER, id="non-string-metadata-ignored"),
        pytest.param(
            Annotated[int, 42, "first", "second"],
            {"type": "integer", "description": "first"},
            id="first-string-wins",
        ),
        pytest.param(
            Annotated[Annotated[int, "inner"], "outer"],
            {"type": "integer", "description": "inner"},
            id="nested-Annotated-flattens-to-first",
        ),
        pytest.param(
            list[Annotated[int, "An id"]],
            {"type": "array", "items": {"type": "integer", "description": "An id"}},
            id="inside-list",
        ),
        pytest.param(
            Annotated[list[Annotated[int, "An id"]], "Ids"],
            {
                "type": "array",
                "items": {"type": "integer", "description": "An id"},
                "description": "Ids",
            },
            id="both-levels",
        ),
        pytest.param(
            dict[str, Annotated[int, "A count"]],
            {
                "type": "object",
                "additionalProperties": {"type": "integer", "description": "A count"},
            },
            id="inside-dict-values",
        ),
        pytest.param(
            Annotated[str | None, "Maybe"],
            {"anyOf": [STRING, NULL], "description": "Maybe"},
            id="around-Optional",
        ),
        pytest.param(
            Annotated[str, "Sure"] | None,
            {"anyOf": [{"type": "string", "description": "Sure"}, NULL]},
            id="inside-Optional",
        ),
        pytest.param(
            tuple[Annotated[float, "lat"], Annotated[float, "lon"]],
            {
                "type": "array",
                "prefixItems": [
                    {"type": "number", "description": "lat"},
                    {"type": "number", "description": "lon"},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
            id="inside-fixed-tuple",
        ),
        pytest.param(
            Annotated[Color, "Pick one"],
            {"type": "string", "enum": ["red", "green"], "description": "Pick one"},
            id="around-Enum",
        ),
    ],
)
def test_annotated_descriptions_survive_at_any_nesting(
    py_type: Any, expected: dict[str, Any]
) -> None:
    assert python_type_to_json_schema(py_type) == expected


def test_schemas_are_fresh_dicts() -> None:
    """Callers may add to a returned schema without affecting later calls."""
    first = python_type_to_json_schema(str)
    first["description"] = "mutated"
    assert python_type_to_json_schema(str) == STRING
    assert python_type_to_json_schema(Annotated[str, "x"]) == {
        "type": "string",
        "description": "x",
    }


# --- typeddict_to_json_schema ------------------------------------------------------


def test_typeddict_simple() -> None:
    assert typeddict_to_json_schema(Address) == ADDRESS_SCHEMA
    assert python_type_to_json_schema(Address) == ADDRESS_SCHEMA


def test_typeddict_nested_and_in_lists_recurse() -> None:
    assert typeddict_to_json_schema(Person) == {
        "type": "object",
        "properties": {
            "name": STRING,
            "address": ADDRESS_SCHEMA,
            "previous": {"type": "array", "items": ADDRESS_SCHEMA},
            "nickname": {"anyOf": [STRING, NULL]},
        },
        "required": ["address", "name", "previous"],
    }


def test_typeddict_from_typing_extensions_is_recognized_everywhere() -> None:
    """``typing_extensions.TypedDict`` is the 3.10-compatible way to use
    NotRequired; it must be treated as a TypedDict on every Python version,
    not fall back to the string default."""
    assert python_type_to_json_schema(Person)["type"] == "object"
    assert build_input_schema(Person)["properties"]["name"] == STRING


def test_typeddict_total_false_has_no_required_keys() -> None:
    assert typeddict_to_json_schema(Loose) == {
        "type": "object",
        "properties": {"anything": STRING},
    }


def test_typeddict_empty() -> None:
    class Empty(TypedDict):
        pass

    assert typeddict_to_json_schema(Empty) == {"type": "object", "properties": {}}


def test_typeddict_readonly_required_and_notrequired_are_unwrapped() -> None:
    assert typeddict_to_json_schema(Settings) == {
        "type": "object",
        "properties": {
            "key": STRING,
            "retries": INTEGER,
            "mode": {"type": "string", "description": "Operating mode"},
        },
        "required": ["key", "mode"],
    }


def test_typeddict_with_every_supported_field_kind() -> None:
    class Everything(TypedDict):
        when: datetime.datetime
        ref: uuid.UUID
        amount: decimal.Decimal
        color: Color
        level: Literal["low", "high"]
        tags: set[str]
        scores: dict[str, float]
        point: tuple[float, float]

    schema = typeddict_to_json_schema(Everything)
    assert schema["properties"] == {
        "when": {"type": "string", "format": "date-time"},
        "ref": {"type": "string", "format": "uuid"},
        "amount": NUMBER,
        "color": {"type": "string", "enum": ["red", "green"]},
        "level": {"type": "string", "enum": ["low", "high"]},
        "tags": {"type": "array", "items": STRING, "uniqueItems": True},
        "scores": {"type": "object", "additionalProperties": NUMBER},
        "point": {
            "type": "array",
            "prefixItems": [NUMBER, NUMBER],
            "minItems": 2,
            "maxItems": 2,
        },
    }
    assert schema["required"] == sorted(schema["properties"])


# --- build_input_schema ------------------------------------------------------------


def test_build_explicit_json_schema_passes_through_untouched() -> None:
    explicit = {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}},
        "required": ["name"],
        "additionalProperties": False,
    }
    assert build_input_schema(explicit) is explicit


def test_build_dict_of_types_requires_every_parameter_in_order() -> None:
    assert build_input_schema(
        {
            "query": Annotated[str, "What to search for"],
            "limit": int | None,
            "tags": list[str],
        }
    ) == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"},
            "limit": {"anyOf": [INTEGER, NULL]},
            "tags": {"type": "array", "items": STRING},
        },
        "required": ["query", "limit", "tags"],
    }


def test_build_empty_dict_is_an_object_with_no_parameters() -> None:
    assert build_input_schema({}) == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_build_parameters_named_type_and_properties_are_still_parameters() -> None:
    """Only a dict whose ``type`` value is a string is an explicit JSON Schema;
    Python types under those keys are parameters like any other."""
    assert build_input_schema({"type": str, "properties": int}) == {
        "type": "object",
        "properties": {"type": STRING, "properties": INTEGER},
        "required": ["type", "properties"],
    }


def test_build_typeddict() -> None:
    assert build_input_schema(Address) == ADDRESS_SCHEMA


def test_build_anything_else_is_an_empty_object() -> None:
    assert build_input_schema(Opaque) == {"type": "object", "properties": {}}
    assert build_input_schema(str) == {"type": "object", "properties": {}}


# --- module hygiene -----------------------------------------------------------------


def test_schema_module_depends_on_the_standard_library_only() -> None:
    """``_schema`` is imported by the package root, so it must not import it
    back (or anything else in the package) and start an import cycle."""
    tree = ast.parse(Path(_schema.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports"
            assert node.module is not None
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "collections",
        "datetime",
        "decimal",
        "enum",
        "sys",
        "types",
        "typing",
        "typing_extensions",
        "uuid",
    }, imported
