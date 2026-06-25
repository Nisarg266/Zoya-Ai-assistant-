"""Low-level Gemini Client wrapper.

Handles the raw `google-genai` SDK, manages network calls asynchronously,
and deals with context windows and system instructions.
"""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types

from zoya.core.config import ZoyaSettings
from .schemas import ChatMessage, Role

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper around the Google GenAI SDK for Zoya."""

    def __init__(self, settings: ZoyaSettings) -> None:
        """Initialize the Gemini client using the Zoya configuration.

        Args:
            settings: The validated Zoya configuration.
        """
        self.settings = settings
        self.api_key = settings.app.gemini_api_key
        self.model_name = settings.app.gemini_model

        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. The Brain will not function.")

        # Initialize the Async client
        self._client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1alpha"})

    async def generate_response(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
    ) -> types.GenerateContentResponse:
        """Send a conversation history to the model and get a response.

        Args:
            messages: The list of chat messages (history).
            tools: Formatted function declarations for tool calling.
            system_instruction: The core prompt that dictates the persona.

        Returns:
            The raw response object from the SDK.
        """
        # Convert our agnostic ChatMessage format to the SDK's Content format
        contents = []
        for msg in messages:
            # We skip 'system' role messages here as they are handled via `system_instruction` config
            if msg.role == Role.SYSTEM:
                continue
            
            # Reconstruct content parts to match types.Content
            parts = []
            if isinstance(msg.content, str):
                parts.append(types.Part.from_text(text=msg.content))
            else:
                # Assuming msg.content is a list of dicts (function calls/responses)
                for part_dict in msg.content:
                    if "function_call" in part_dict:
                        parts.append(types.Part.from_function_call(
                            name=part_dict["function_call"]["name"],
                            args=part_dict["function_call"]["args"]
                        ))
                    elif "function_response" in part_dict:
                        parts.append(types.Part.from_function_response(
                            name=part_dict["function_response"]["name"],
                            response=part_dict["function_response"]["response"]
                        ))
                    else:
                        # Fallback for plain text inside dict structure if it ever happens
                        parts.append(types.Part.from_text(text=str(part_dict)))

            # 'user' and 'model' map directly to the SDK roles
            contents.append(types.Content(role=msg.role.value, parts=parts))

        config_dict = {}
        if system_instruction:
            config_dict["system_instruction"] = system_instruction
            
        if tools:
            # Convert raw dict tools to types.Tool
            parsed_tools = []
            for t_group in tools:
                if "function_declarations" in t_group:
                    func_decls = []
                    for fd in t_group["function_declarations"]:
                        schema_dict = fd["parameters"]
                        # We map our raw schema to types.Schema
                        func_decls.append(
                            types.FunctionDeclaration(
                                name=fd["name"],
                                description=fd.get("description", ""),
                                parameters=types.Schema(
                                    type=types.Type.OBJECT,
                                    properties={
                                        k: types.Schema(type=v.get("type", "STRING").upper(), description=v.get("description", ""))
                                        for k, v in schema_dict.get("properties", {}).items()
                                    },
                                    required=schema_dict.get("required")
                                )
                            )
                        )
                    parsed_tools.append(types.Tool(function_declarations=func_decls))
            config_dict["tools"] = parsed_tools

        config = types.GenerateContentConfig(**config_dict)

        logger.debug(f"Sending request to {self.model_name} with {len(contents)} messages.")
        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )
        return response
