"""Bridges Zoya's tool registry with Gemini's function calling schema."""

from typing import Any

from zoya.automation.tools.base import ITool


def build_gemini_tools(tools: list[ITool]) -> list[dict[str, Any]]:
    """Convert a list of Zoya tools into Gemini Function Declarations.

    Args:
        tools: A list of instantiated ITool plugins.

    Returns:
        A list of dictionaries representing Gemini tools. The google-genai
        SDK accepts this directly in the `tools` parameter of `generate_content`.
    """
    declarations = []

    for tool in tools:
        # Get the JSON schema from the tool
        # tool.schema() returns a dict with name, description, and parameters
        schema_data = tool.schema()
        schema_params = schema_data.get("parameters", {})

        # Clean up the schema to match exactly what Gemini expects
        properties = schema_params.get("properties", {})
        required = schema_params.get("required", [])

        # Gemini requires 'type' to be uppercase in its raw REST API, but the
        # google-genai SDK handles standard JSON schemas reasonably well.
        # However, to be safe, we structure it closely to OpenAPI 3.0.
        function_decl = {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "OBJECT",
                "properties": properties,
            },
        }

        if required:
            function_decl["parameters"]["required"] = required

        declarations.append(function_decl)

    # Wrap them in the Tool format expected by Gemini
    if not declarations:
        return []

    return [{"function_declarations": declarations}]
