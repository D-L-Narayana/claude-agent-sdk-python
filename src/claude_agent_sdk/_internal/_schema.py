"""JSON Schema for the input of SDK MCP tools.

A tool declares its input as a JSON Schema dict (sent as is), a dict of
parameter names to Python types, or a TypedDict class. This module turns
the Python types into the JSON Schema that Claude Code sends to the model
and that the SDK validates arguments against.

Supported annotations and what they become:

- ``str``/``int``/``float``/``bool`` → the matching JSON type; ``None`` →
  ``null``; ``Any``/``object`` → ``{}`` (anything)
- ``datetime``/``date``/``time`` → ``string`` with a ``format``;
  ``uuid.UUID`` → ``string`` with ``format: uuid``; ``Decimal`` → ``number``
- ``list[X]``, ``Sequence[X]``, ``tuple[X, ...]`` → ``array`` of ``X``;
  ``set[X]``/``frozenset[X]`` add ``uniqueItems``; ``tuple[A, B]`` →
  ``prefixItems`` with a fixed length
- ``dict[str, X]``/``Mapping[str, X]`` → ``object`` with
  ``additionalProperties``; bare ``dict``/``dict[str, Any]`` → ``object``
- ``X | None`` / ``Optional[X]`` / ``Union[...]`` → ``anyOf`` of every
  alternative (``None`` included, so a null argument validates)
- ``Literal[...]`` and ``enum.Enum`` subclasses → ``enum`` (plus ``type``
  when every value shares one JSON type)
- TypedDict classes (nested, in lists, behind ``NotRequired``/``Required``/
  ``ReadOnly``) → ``object`` with ``properties`` and ``required``
- ``Annotated[X, "description"]`` → ``X``'s schema with the first string
  metadata as its ``description``, at any nesting
- ``NewType`` → its supertype; anything else → ``string``

It depends on the standard library (and ``typing_extensions``) only and must
not import the package root, which imports it.
"""

from __future__ import annotations

import collections.abc
import datetime
import decimal
import enum
import sys
import types as builtin_types
import typing
import uuid
from collections.abc import Callable
from typing import Annotated, Any, Literal, Union, get_args, get_origin

if sys.version_info >= (3, 11):
    from typing import get_type_hints
else:
    # On 3.10 stdlib get_type_hints doesn't strip NotRequired markers.
    from typing_extensions import get_type_hints

# The stdlib check does not recognize typing_extensions.TypedDict classes (the
# 3.10-compatible way to use NotRequired; re-implemented there through 3.12),
# and on 3.10 the typing_extensions check does not recognize stdlib ones, so
# both are asked. typing_extensions is only a declared dependency below 3.11
# (pydantic and mcp bring it along everywhere else), hence the guard.
_TYPEDDICT_CHECKS: tuple[Callable[[Any], bool], ...]
try:
    from typing_extensions import is_typeddict as _ext_is_typeddict
except ImportError:  # pragma: no cover - a declared dependency on 3.10
    _TYPEDDICT_CHECKS = (typing.is_typeddict,)
else:
    _TYPEDDICT_CHECKS = (typing.is_typeddict, _ext_is_typeddict)

__all__ = [
    "build_input_schema",
    "python_type_to_json_schema",
    "typeddict_to_json_schema",
]

# Looked up by identity first, then by subclass in this order (bool before
# int, datetime before date: each is a subclass of the next).
_SCALARS: tuple[tuple[type, dict[str, Any]], ...] = (
    (str, {"type": "string"}),
    (bytes, {"type": "string"}),
    (bool, {"type": "boolean"}),
    (int, {"type": "integer"}),
    (float, {"type": "number"}),
    (type(None), {"type": "null"}),
    (decimal.Decimal, {"type": "number"}),
    (datetime.datetime, {"type": "string", "format": "date-time"}),
    (datetime.date, {"type": "string", "format": "date"}),
    (datetime.time, {"type": "string", "format": "time"}),
    (uuid.UUID, {"type": "string", "format": "uuid"}),
)

# JSON type of a Literal or Enum value; bool before int, it is a subclass.
_VALUE_TYPES: tuple[tuple[type, str], ...] = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (type(None), "null"),
)

_ARRAY_ORIGINS = (
    list,
    collections.abc.Sequence,
    collections.abc.MutableSequence,
    collections.abc.Iterable,
    collections.abc.Collection,
)
_UNIQUE_ARRAY_ORIGINS = (
    set,
    frozenset,
    collections.abc.Set,
    collections.abc.MutableSet,
)
_OBJECT_ORIGINS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)
_TYPE_QUALIFIERS = ("NotRequired", "Required", "ReadOnly")


def _is_typeddict(candidate: Any) -> bool:
    return any(check(candidate) for check in _TYPEDDICT_CHECKS)


def _is_any(py_type: Any) -> bool:
    return py_type is Any or py_type is object


def _values_schema(values: list[Any]) -> dict[str, Any]:
    """``{"enum": values}`` plus ``"type"`` when every value has the same JSON type."""
    json_types: set[str] = set()
    for value in values:
        for python_type, json_type in _VALUE_TYPES:
            if isinstance(value, python_type):
                json_types.add(json_type)
                break
        else:
            json_types.add("?")
    schema: dict[str, Any] = {}
    if len(json_types) == 1 and "?" not in json_types:
        schema["type"] = json_types.pop()
    schema["enum"] = values
    return schema


def python_type_to_json_schema(py_type: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema dict.

    The result is a fresh dict each time, so callers may add to it. See the
    module docstring for the supported annotations; anything unknown is
    sent as a string.
    """
    if py_type is None:
        return {"type": "null"}
    if _is_any(py_type):
        return {}

    origin = get_origin(py_type)

    # NotRequired/Required/ReadOnly survive include_extras=True; unwrap them
    if getattr(origin, "_name", None) in _TYPE_QUALIFIERS:
        return python_type_to_json_schema(get_args(py_type)[0])

    if origin is Annotated:
        args = get_args(py_type)
        schema = python_type_to_json_schema(args[0])
        for meta in args[1:]:
            if isinstance(meta, str):
                schema["description"] = meta
                break
        return schema

    if origin is Literal:
        values = [
            value.value if isinstance(value, enum.Enum) else value
            for value in get_args(py_type)
        ]
        return _values_schema(values)

    if origin is Union or isinstance(py_type, builtin_types.UnionType):
        return {"anyOf": [python_type_to_json_schema(a) for a in get_args(py_type)]}

    if origin is not None:
        return _generic_to_json_schema(origin, get_args(py_type))

    supertype = getattr(py_type, "__supertype__", None)
    if supertype is not None:  # a NewType
        return python_type_to_json_schema(supertype)

    for scalar, schema in _SCALARS:
        if py_type is scalar:
            return dict(schema)

    if _is_typeddict(py_type):
        return typeddict_to_json_schema(py_type)

    if isinstance(py_type, type):
        if issubclass(py_type, enum.Enum):
            return _values_schema([member.value for member in py_type])
        for scalar, schema in _SCALARS:
            if issubclass(py_type, scalar):
                return dict(schema)
        # Mappings before sequences: a Mapping is a Collection too.
        if issubclass(py_type, _OBJECT_ORIGINS):
            return {"type": "object"}
        if issubclass(py_type, _UNIQUE_ARRAY_ORIGINS):
            return {"type": "array", "uniqueItems": True}
        if py_type is tuple or issubclass(py_type, _ARRAY_ORIGINS):
            return {"type": "array"}

    return {"type": "string"}


def _generic_to_json_schema(origin: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    """Schema for a parameterized generic such as ``list[int]`` or ``dict[str, X]``."""
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": python_type_to_json_schema(args[0])}
        if args == ((),) or not args:  # tuple[()] on older Pythons / bare
            return {"type": "array", "maxItems": 0}
        return {
            "type": "array",
            "prefixItems": [python_type_to_json_schema(a) for a in args],
            "minItems": len(args),
            "maxItems": len(args),
        }
    if origin in _UNIQUE_ARRAY_ORIGINS:
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = python_type_to_json_schema(args[0])
        schema["uniqueItems"] = True
        return schema
    if origin in _ARRAY_ORIGINS:
        if args:
            return {"type": "array", "items": python_type_to_json_schema(args[0])}
        return {"type": "array"}
    if origin in _OBJECT_ORIGINS:
        if len(args) == 2 and not _is_any(args[1]):
            return {
                "type": "object",
                "additionalProperties": python_type_to_json_schema(args[1]),
            }
        return {"type": "object"}
    if isinstance(origin, type) and issubclass(origin, _OBJECT_ORIGINS):
        return {"type": "object"}
    if isinstance(origin, type) and issubclass(origin, _ARRAY_ORIGINS):
        return {"type": "array"}
    return {"type": "string"}


def typeddict_to_json_schema(td_class: type) -> dict[str, Any]:
    """Convert a TypedDict class to a JSON Schema dict."""
    hints = get_type_hints(td_class, include_extras=True)

    properties: dict[str, Any] = {}
    for field_name, field_type in hints.items():
        properties[field_name] = python_type_to_json_schema(field_type)

    required_keys = getattr(td_class, "__required_keys__", set(properties.keys()))
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required_keys:
        schema["required"] = sorted(required_keys)
    return schema


def build_input_schema(input_schema: type | dict[str, Any]) -> dict[str, Any]:
    """Turn a tool's declared input_schema into the JSON Schema sent on the wire.

    A dict with a string ``type`` and ``properties`` is an explicit JSON
    Schema and passes through untouched; any other dict maps parameter names
    to Python types, all of them required; a TypedDict class is converted;
    anything else is an object with no declared parameters.
    """
    if isinstance(input_schema, dict):
        if (
            "type" in input_schema
            and "properties" in input_schema
            and isinstance(input_schema["type"], str)
        ):
            return input_schema
        properties = {
            param_name: python_type_to_json_schema(param_type)
            for param_name, param_type in input_schema.items()
        }
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        }
    if _is_typeddict(input_schema):
        return typeddict_to_json_schema(input_schema)
    return {"type": "object", "properties": {}}
