"""Schema-aware decoding of Qwen XML parameter text."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import Any


def parameter_schemas(tools: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Each offered tool's parameter schemas by lowercase name; a spec that is not an object reads as untyped."""

    result = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool["function"] if isinstance(tool.get("function"), dict) else tool
        parameters = function.get("parameters") or function.get("input_schema") or {}
        properties = parameters.get("properties") or {} if isinstance(parameters, dict) else {}
        result[str(function.get("name", "")).lower()] = {
            name: schema for name, schema in properties.items() if isinstance(schema, dict)
        } if isinstance(properties, dict) else {}
    return result


def typed_parameter(schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    return isinstance(kind, str) and kind in {"array", "object", "boolean", "integer", "number", "null"}


def _closed_json(text: str) -> str | None:
    """Close only unfinished JSON containers, leaving malformed or unfinished strings alone."""

    closers, in_string, escaped = [], False, False
    for char in text:
        if in_string:
            escaped, in_string = (False, True) if escaped else (char == "\\", char != '"')
        elif char == '"':
            in_string = True
        elif char in "[{":
            closers.append("]" if char == "[" else "}")
        elif char in "]}" and (not closers or closers.pop() != char):
            return None
    return text.rstrip() + "".join(reversed(closers)) if closers and not in_string else None


def _python_literal(text: str) -> Any:
    words = {"true": True, "false": False, "none": None, "null": None}
    word = words.get(text.strip().lower(), ...)
    if word is not ...:
        return word

    def lists(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [lists(item) for item in value]
        if isinstance(value, dict):
            return {key: lists(item) for key, item in value.items()}
        return value

    try:
        return lists(ast.literal_eval(text.strip()))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(str(exc)) from None


def decode_parameter(value: str, schema: dict[str, Any]) -> Any:
    if not typed_parameter(schema):
        return value
    kind = schema["type"]
    valid = {
        "array": lambda item: isinstance(item, list),
        "object": lambda item: isinstance(item, dict),
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: type(item) is int,
        "number": lambda item: type(item) in (int, float),
        "null": lambda item: item is None,
    }[kind]
    for text in (value, _closed_json(value) if kind in ("array", "object") else None):
        if text is None:
            continue
        try:
            parsed = json.loads(text)
            json.dumps(parsed, allow_nan=False)
        except (ValueError, TypeError):
            continue
        return parsed if valid(parsed) else value
    try:
        parsed = _python_literal(value)
        json.dumps(parsed, allow_nan=False)
    except (ValueError, TypeError):
        return value
    return parsed if valid(parsed) else value
