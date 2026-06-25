"""The high-level facade for the Brain (LLM).

Coordinates the chat loop: taking a user prompt, passing it to the Gemini client,
and executing any requested tools until a final text response is produced.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from zoya.automation.tools.base import ITool
from zoya.core.config import ZoyaSettings
from .client import GeminiClient
from .function_tools import build_gemini_tools
from .schemas import ChatMessage, Role

logger = logging.getLogger(__name__)


class ZoyaBrain:
    """The central LLM orchestrator.

    Maintains conversational history and handles the "ReAct" loop:
    User Input -> Model -> Tool Call -> Tool Execution -> Model -> Output.
    """

    def __init__(self, settings: ZoyaSettings, tools: list[ITool] | None = None) -> None:
        """Initialize the Brain.

        Args:
            settings: The Zoya configuration.
            tools: An optional list of instantiated ITool plugins that the LLM can use.
        """
        self.settings = settings
        self.client = GeminiClient(settings)
        self.tools = tools or []
        self._tool_map = {tool.name: tool for tool in self.tools}
        self.gemini_tools = build_gemini_tools(self.tools)
        
        # In-memory session history
        self.history: list[ChatMessage] = []

        # Default system instruction
        self.system_instruction = (
            "You are Zoya, a highly advanced desktop AI assistant inspired by JARVIS. "
            "You run on Windows and have access to tools that can automate the system, "
            "manage files, windows, and processes. Be concise, professional, and helpful. "
            "When asked to perform an action, use the tools provided. If a tool fails, "
            "explain the error to the user."
        )

    async def chat(self, user_prompt: str) -> str:
        """Process a single turn of conversation.

        Args:
            user_prompt: The text input from the user.

        Returns:
            The final text response from the model.
        """
        if not self.client.api_key:
            return "Error: GEMINI_API_KEY is not configured in .env."

        # Append user message
        self.history.append(ChatMessage(role=Role.USER, content=user_prompt))

        while True:
            try:
                # Call Gemini
                response = await self.client.generate_response(
                    messages=self.history,
                    tools=self.gemini_tools if self.gemini_tools else None,
                    system_instruction=self.system_instruction
                )
            except Exception as e:
                logger.exception("Failed to generate response from Gemini.")
                return f"I encountered an error communicating with my brain: {e}"

            if not response.candidates:
                return "I'm sorry, I couldn't generate a response."

            candidate = response.candidates[0]
            
            # The model could return text, function calls, or both.
            # We need to process function calls if any.
            part = candidate.content.parts[0] if candidate.content and candidate.content.parts else None
            
            if not part:
                 return "I'm sorry, my response was empty."

            # Case 1: The model returned a function call
            if part.function_call:
                function_call = part.function_call
                tool_name = function_call.name
                
                # Gemini args is usually a dict-like protobuf structure. We need it as a dict.
                # In google-genai, args is a Python dict directly
                tool_args = function_call.args if isinstance(function_call.args, dict) else dict(function_call.args)
                
                logger.info(f"LLM called tool: {tool_name} with args: {tool_args}")
                
                # Record the model's function call in history
                self.history.append(ChatMessage(
                    role=Role.MODEL,
                    content=[{"function_call": {"name": tool_name, "args": tool_args}}]
                ))

                # Execute the tool
                tool_result_payload = await self._execute_tool(tool_name, tool_args)
                
                # Record the tool response in history
                self.history.append(ChatMessage(
                    role=Role.USER, # In the SDK, function responses must be supplied by the user role
                    content=[{"function_response": {"name": tool_name, "response": tool_result_payload}}]
                ))
                
                # Loop back to let the model see the tool result
                continue

            # Case 2: The model returned text (final answer)
            if part.text:
                response_text = part.text
                self.history.append(ChatMessage(role=Role.MODEL, content=response_text))
                return response_text

            return "I generated an unknown response format."

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return its payload in a format Gemini accepts."""
        if name not in self._tool_map:
            error_msg = f"Tool '{name}' is not registered."
            logger.error(error_msg)
            return {"error": error_msg}

        tool = self._tool_map[name]
        try:
            # We assume tools implement an async __call__ or run method.
            # From zoya.automation.tools.base, it might be async tool(params) or tool.run()
            # Let's check how ITool is defined. Usually it's `await tool(args)`.
            # If the tool is synchronous, we wrap it.
            # Let's assume the tool provides a `dispatch` or `execute` or `__call__` method
            # that takes a dict.
            # For now, we will use `await tool(args)` and expect it to return a ToolResult or dict.
            # We'll use a generic approach:
            if hasattr(tool, "execute"):
                # If they have execute, call it
                result = await tool.execute(args)
            elif callable(tool):
                result = await tool(args)
            else:
                return {"error": f"Tool '{name}' is not callable or lacks an execute method."}

            if hasattr(result, "to_payload"):
                return result.to_payload()
            elif hasattr(result, "model_dump"):
                return result.model_dump()
            return result
        except Exception as e:
            logger.exception(f"Error executing tool {name}")
            return {"error": str(e), "success": False}
