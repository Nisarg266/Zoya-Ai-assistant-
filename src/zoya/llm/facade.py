"""The high-level facade for the Brain (LLM).

:class:`ZoyaBrain` orchestrates the full conversational loop using a manual
ReAct pattern:

    User prompt
        └─► Gemini (with tool declarations)
              ├─ returns text        → done
              └─ returns tool call(s) → dispatch each tool via
                  ``await tool.execute(args)`` → feed ``FunctionResponse`` back
                  → loop again

Two entry points share the tool-dispatch core:

* :meth:`chat`          — **non-streaming**. Robust and simple; returns the
  final assembled answer string. This is the recommended path and what the
  demo script uses.
* :meth:`chat_stream`   — **streaming**. An async generator yielding
  :data:`~zoya.llm.schemas.BrainEvent` objects (``TextDelta`` /
  ``ToolCallStarted`` / ``ToolCallFinished`` / ``TurnComplete`` / ``ErrorEvent``)
  so a UI can render live token-by-token progress.

Both walk *every* part of each model response (the old code read only
``parts[0]`` and silently dropped parallel tool calls), enforce a max-iteration
guard against runaway tool ping-pong, and turn every failure into a mapped
:class:`~zoya.core.exceptions.LLMError`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from google.genai import types

from zoya.automation.schemas import ToolResult
from zoya.automation.tools.base import ITool
from zoya.core.config import ZoyaSettings
from zoya.core.exceptions import LLMError, ToolValidationError
from zoya.core.logging import get_logger
from zoya.llm.client import DEFAULT_SYSTEM_INSTRUCTION, GeminiClient
from zoya.llm.function_tools import build_gemini_tools
from zoya.llm.history import ConversationHistory
from zoya.llm.schemas import (
    BrainEvent,
    ErrorEvent,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)

_log = get_logger("llm.brain")


class ZoyaBrain:
    """The central LLM orchestrator (manual ReAct, streaming-capable)."""

    def __init__(
        self,
        settings: ZoyaSettings,
        tools: list[ITool] | None = None,
        *,
        history: ConversationHistory | None = None,
        system_instruction: str | None = None,
    ) -> None:
        """Initialise the Brain.

        Args:
            settings: The validated Zoya configuration.
            tools: Instantiated tool plugins the model may call.
            history: Optional pre-seeded history (e.g. restored from disk).
                A fresh one is created when omitted.
            system_instruction: Override the default persona prompt.
        """
        self.settings = settings
        self.client = GeminiClient(settings)
        self.tools: list[ITool] = list(tools or [])
        self._tool_map: dict[str, ITool] = {t.name: t for t in self.tools}
        self._gemini_tools = build_gemini_tools(self.tools)

        # max_turns=0 means "no turn cap" — rely solely on the token budget.
        turn_cap = settings.llm.max_history_turns or None
        self.history = history or ConversationHistory(max_turns=turn_cap)
        if history is not None and turn_cap is not None:
            # Respect the configured cap even on an injected history.
            history._max_turns = turn_cap  # noqa: SLF001
        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        self._max_iterations = settings.llm.max_react_iterations
        self._max_history_tokens: int = settings.llm.max_history_tokens

    # ------------------------------------------------------------------ utils
    @property
    def is_configured(self) -> bool:
        """``True`` when the client has an API key ready."""
        return self.client.is_configured

    def reset(self) -> None:
        """Clear conversation history (keeps tools + persona)."""
        self.history.clear()

    # ============================================================ public API
    async def chat(self, prompt: str) -> str:
        """Process one user turn (non-streaming) and return the final answer.

        Tools execute transparently between model calls. LLM errors are caught
        and returned as a user-facing message so an interactive loop never
        crashes on a transient outage.
        """
        if not self.is_configured:
            return "Error: GEMINI_API_KEY is not configured in your .env file."

        self.history.add_user_text(prompt)

        try:
            return await self._react_nonstream()
        except LLMError as exc:
            _log.error("Brain turn failed: %s", exc)
            return f"I encountered an error communicating with my brain: {exc}"

    async def chat_stream(self, prompt: str) -> AsyncIterator[BrainEvent]:
        """Process one user turn, yielding :data:`BrainEvent` objects as we go.

        The final answer is streamed token-by-token as ``TextDelta`` chunks and
        terminated by one ``TurnComplete`` (carrying the full text). Tool turns
        emit ``ToolCallStarted`` / ``ToolCallFinished``. Any LLM error becomes an
        ``ErrorEvent`` and ends the stream.
        """
        if not self.is_configured:
            yield ErrorEvent(
                error_type="LLMAuthError",
                message="GEMINI_API_KEY is not configured.",
            )
            return

        self.history.add_user_text(prompt)

        try:
            async for event in self._react_stream():
                yield event
        except LLMError as exc:
            _log.error("Brain turn failed: %s", exc)
            yield ErrorEvent(error_type=type(exc).__name__, message=str(exc))

    # ============================================================ non-stream
    async def _react_nonstream(self) -> str:
        """Manual ReAct loop using :meth:`GeminiClient.generate`.

        Walks every part of each response; dispatches all tool calls in a turn
        together; feeds ``FunctionResponse`` blocks back; returns the first
        tool-free text answer.
        """
        for _ in range(self._max_iterations):
            response = await self.client.generate(
                contents=self.history.as_contents(),
                tools=self._gemini_tools,
                system_instruction=self.system_instruction,
            )
            text, calls = _parse_response(response)

            if calls:
                # Record the model turn (text + calls) then answer each call.
                self.history.add_model_turn(text, calls)
                results: list[tuple[str, dict[str, Any]]] = []
                for name, args in calls:
                    _log.info("LLM called tool %s with %r", name, args)
                    payload = await self._execute_tool(name, args)
                    results.append((name, payload))
                self.history.add_tool_responses(results)
                continue

            # Final answer.
            if text:
                self.history.add_model_text(text)
            return text or "I'm sorry, I couldn't generate a response."

        return self._exhausted_message()

    # ================================================================ stream
    async def _react_stream(self) -> AsyncIterator[BrainEvent]:
        """Manual ReAct loop using :meth:`GeminiClient.generate_stream`.

        Text is streamed live as ``TextDelta``. Function calls are merged
        defensively across chunks (complete ``args`` overwrite; ``partial_args``
        accumulate) before dispatch — so even when the SDK streams a call's
        arguments in pieces we still reconstruct the full call.
        """
        for _ in range(self._max_iterations):
            text_buf: list[str] = []
            call_buckets: dict[str, dict[str, Any]] = {}
            call_order: list[str] = []

            async for chunk in self.client.generate_stream(
                contents=self.history.as_contents(),
                tools=self._gemini_tools,
                system_instruction=self.system_instruction,
            ):
                chunk_text, chunk_calls = _parse_response(chunk)
                if chunk_text:
                    text_buf.append(chunk_text)
                    yield TextDelta(text=chunk_text)
                for fc_name, fc_args, fc_id in chunk_calls:
                    key = fc_id or fc_name
                    if key not in call_buckets:
                        call_buckets[key] = {"name": fc_name, "args": {}}
                        call_order.append(key)
                    # Complete args (the common case) overwrite; partials update.
                    if fc_args:
                        call_buckets[key]["args"] = dict(fc_args)

            calls = [
                (call_buckets[k]["name"], call_buckets[k]["args"])
                for k in call_order
            ]
            final_text = "".join(text_buf)

            if calls:
                self.history.add_model_turn(final_text, calls)
                results: list[tuple[str, dict[str, Any]]] = []
                for name, args in calls:
                    _log.info("LLM called tool %s with %r", name, args)
                    yield ToolCallStarted(name=name, args=dict(args))
                    payload = await self._execute_tool(name, args)
                    yield ToolCallFinished(
                        name=name,
                        success=bool(payload.get("success", True)),
                        summary=_summarise_result(payload),
                    )
                    results.append((name, payload))
                self.history.add_tool_responses(results)
                continue

            if final_text:
                self.history.add_model_text(final_text)
            yield TurnComplete(text=final_text)
            return

        yield ErrorEvent(
            error_type="LLMResponseError", message=self._exhausted_message()
        )

    def _exhausted_message(self) -> str:
        return (
            f"Exceeded max ReAct iterations ({self._max_iterations}) "
            "without a final answer."
        )

    # --------------------------------------------------------- tool dispatch
    async def _execute_tool(
        self, name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one tool and return a Gemini-ready ``FunctionResponse`` payload.

        Failures are *never* raised here: they're turned into an error payload
        so the model can read them and self-correct (Gemini's tool contract).
        A :class:`ToolValidationError` from bad args is likewise surfaced to the
        model rather than crashing the loop.
        """
        tool = self._tool_map.get(name)
        if tool is None:
            _log.error("Model requested unknown tool %r.", name)
            return {"success": False, "error": f"Unknown tool: {name}"}

        try:
            result: ToolResult = await tool.execute(args)
        except ToolValidationError as exc:
            _log.warning("Tool %s rejected args: %s", name, exc)
            return {"success": False, "error": f"Invalid arguments: {exc}"}
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("Tool %s raised unexpectedly.", name)
            return {"success": False, "error": f"Unexpected error: {exc}"}

        payload = result.to_payload()
        # Guarantee a JSON-serialisable dict for the FunctionResponse.
        if not isinstance(payload, dict):
            payload = {"result": payload}
        return payload

    # ------------------------------------------------------------------ close
    async def aclose(self) -> None:
        """Release the underlying client transport."""
        await self.client.aclose()


# ===========================================================================
# Response parsing helpers
# ===========================================================================
def _parse_response(
    response: types.GenerateContentResponse,
) -> tuple[str, list[tuple[str, dict[str, Any], str | None]]]:
    """Extract ``(concatenated_text, [(name, args, id), ...])`` from a response.

    Walks every part of the first candidate (the old code read only
    ``parts[0]`` and silently dropped parallel tool calls / interleaved text).
    Returns empty text + empty list for a blocked/empty response.

    The per-call ``id`` (when Gemini assigns one to parallel calls) is returned
    so the streaming merger can keep distinct same-named calls apart.
    """
    if not response.candidates:
        return "", []
    candidate = response.candidates[0]
    content = getattr(candidate, "content", None)
    if content is None or not content.parts:
        return "", []

    text_chunks: list[str] = []
    calls: list[tuple[str, dict[str, Any], str | None]] = []
    for part in content.parts:
        if part.text:
            text_chunks.append(part.text)
        if part.function_call:
            fc = part.function_call
            args = dict(fc.args) if fc.args else {}
            fc_id = getattr(fc, "id", None)
            calls.append((fc.name, args, fc_id))
    return "".join(text_chunks), calls


def _summarise_result(result: dict[str, Any]) -> str:
    """Build a short human-readable summary of a tool result for the UI."""
    if not result.get("success", True):
        return str(result.get("error") or "tool failed")
    data = result.get("data", result)
    if isinstance(data, dict) and data:
        return str(next(iter(data.values())))[:120]
    return "ok"


__all__ = ["ZoyaBrain"]
