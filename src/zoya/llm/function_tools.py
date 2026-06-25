"""Bridges Zoya's tool registry with Gemini's function-calling schema.

The previous version hand-rebuilt ``types.Schema`` property-by-property, which
silently dropped enums, nested objects and number constraints — producing tools
the model then called with wrong arguments. The SDK actually accepts a standard
JSON-Schema dict as a function declaration's ``parameters``, so we now pass the
Pydantic-generated schema through with minimal, surgical clean-up.

Why clean-up at all? Pydantic v2 emits a few ``$defs`` / ``title`` keys that the
Gemini function-declaration validator rejects, so we strip them while keeping
``type``/``description``/``enum``/``items``/``properties``/``required`` intact.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

from zoya.automation.tools.base import ITool

#: Top-level JSON-Schema keys Gemini's function-declaration validator accepts.
_ALLOWED_PARAM_KEYS = {
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "format",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
}


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip keys Gemini rejects (e.g. ``$defs``, ``title``), recursively.

    Nested object/array schemas are filtered the same way so a tool that takes
    a structured argument still reaches the model correctly.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _ALLOWED_PARAM_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {
                prop: _clean_schema(sub) for prop, sub in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            cleaned["items"] = _clean_schema(value)
        else:
            cleaned[key] = value

    # Gemini expects the top-level type to be OBJECT for function parameters.
    if "type" not in cleaned:
        cleaned["type"] = "OBJECT"
    else:
        cleaned["type"] = str(cleaned["type"]).upper()
    return cleaned


def build_gemini_tools(tools: list[ITool]) -> types.Tool | None:
    """Convert Zoya tools into a single Gemini :class:`~google.genai.types.Tool`.

    Args:
        tools: Instantiated :class:`ITool` plugins (usually from the registry).

    Returns:
        A ``types.Tool`` wrapping one ``FunctionDeclaration`` per tool, or
        ``None`` when there are no tools (so callers can omit the ``tools``
        config field entirely, which the SDK prefers over an empty list).
    """
    if not tools:
        return None

    declarations: list[types.FunctionDeclaration] = []
    for tool in tools:
        schema_data = tool.schema()
        params_schema = schema_data.get("parameters", {})
        params = _clean_schema(params_schema)

        declarations.append(
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=params,  # SDK coerces the cleaned dict.
            )
        )

    return types.Tool(function_declarations=declarations)


__all__ = ["build_gemini_tools"]
