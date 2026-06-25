"""Provider-native conversation history for the Brain.

The previous facade stored history as a mix of ``ChatMessage`` objects and raw
dicts and re-converted them on every call — fragile and easy to corrupt (e.g.
storing a tool *response* under the wrong role). This class stores native
``google.genai.types.Content`` directly, so :meth:`as_contents` is a zero-cost
hand-off to the SDK.

Conversation invariants enforced here
-------------------------------------
* Every user turn is appended verbatim.
* A model turn that issues tool calls is stored once, as a single ``Content``
  with one ``FunctionCall`` part per call.
* Each tool call is answered by exactly one ``FunctionResponse`` part, grouped
  in a single *user-role* ``Content`` (Gemini's required shape).
* :meth:`trim` drops whole user/model turn-pairs only — it never orphans a
  ``FunctionCall`` from its matching ``FunctionResponse``.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

from zoya.core.logging import get_logger
from zoya.llm.schemas import ChatMessage, Role

_log = get_logger("llm.history")


class ConversationHistory:
    """Append-only store of native Gemini ``Content`` blocks.

    Parameters
    ----------
    max_turns:
        Optional cap on the number of *user/model turn-pairs* kept. When set,
        :meth:`trim` is invoked automatically after each user/model append.
        ``None`` means unbounded (fine for short-lived sessions).
    """

    def __init__(self, max_turns: int | None = None) -> None:
        self._contents: list[types.Content] = []
        self._max_turns = max_turns

    # ------------------------------------------------------------------ size
    def __len__(self) -> int:
        return len(self._contents)

    @property
    def contents(self) -> list[types.Content]:
        """Read-only view of the stored Content blocks."""
        return list(self._contents)

    # ------------------------------------------------------------------ add
    def add_user_text(self, text: str) -> None:
        """Append a plain-text user message."""
        self._contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=text)],
            )
        )
        self._maybe_trim()

    def add_model_text(self, text: str) -> None:
        """Append a plain-text model response."""
        self._contents.append(
            types.Content(
                role="model",
                parts=[types.Part(text=text)],
            )
        )
        self._maybe_trim()

    def add_model_turn(
        self,
        text: str | None,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Record one model turn that may carry text *and/or* tool calls.

        Used when a turn interleaves a natural-language aside with one or more
        function calls — both must be stored together so Gemini sees the same
        turn shape we received.
        """
        parts: list[types.Part] = []
        if text:
            parts.append(types.Part(text=text))
        parts.extend(
            types.Part(function_call=types.FunctionCall(name=name, args=args))
            for name, args in calls
        )
        if not parts:
            return
        self._contents.append(types.Content(role="model", parts=parts))
        self._maybe_trim()

    def add_model_tool_calls(
        self, calls: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Record one model turn that requested one or more tool calls.

        Args:
            calls: list of ``(tool_name, args_dict)`` tuples, in the order the
                model emitted them.
        """
        parts = [
            types.Part(function_call=types.FunctionCall(name=name, args=args))
            for name, args in calls
        ]
        self._contents.append(types.Content(role="model", parts=parts))

    def add_tool_responses(
        self, responses: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Answer tool calls with their results.

        Args:
            responses: list of ``(tool_name, response_dict)`` tuples. All
                responses are bundled into a single user-role ``Content``,
                which is the contract Gemini requires.
        """
        parts = [
            types.Part(
                function_response=types.FunctionResponse(name=name, response=resp)
            )
            for name, resp in responses
        ]
        self._contents.append(types.Content(role="user", parts=parts))

    # ------------------------------------------------------------------ out
    def as_contents(self) -> list[types.Content]:
        """Return the history ready to pass straight to ``generate_content``."""
        return list(self._contents)

    # ------------------------------------------------------------------ misc
    def clear(self) -> None:
        """Drop the entire conversation (keeps the instance reusable)."""
        self._contents.clear()

    def to_chat_messages(self) -> list[ChatMessage]:
        """Project the native history back into provider-agnostic messages.

        Handy for persistence / display. Tool calls & responses are represented
        as rich part dicts.
        """
        messages: list[ChatMessage] = []
        for content in self._contents:
            role = Role.MODEL if content.role == "model" else Role.USER
            parts_out: list[dict[str, Any]] = []
            text_bits: list[str] = []
            for part in content.parts or []:
                if part.text:
                    text_bits.append(part.text)
                elif part.function_call:
                    fc = part.function_call
                    parts_out.append(
                        {
                            "function_call": {
                                "name": fc.name,
                                "args": dict(fc.args or {}),
                            }
                        }
                    )
                elif part.function_response:
                    fr = part.function_response
                    parts_out.append(
                        {
                            "function_response": {
                                "name": fr.name,
                                "response": dict(fr.response or {}),
                            }
                        }
                    )
            content_repr: str | list[dict[str, Any]]
            if parts_out and not text_bits:
                content_repr = parts_out
            elif parts_out and text_bits:
                # Mixed turn: keep text + structured parts together.
                content_repr = [{"text": "\n".join(text_bits)}, *parts_out]
            else:
                content_repr = "\n".join(text_bits)
            messages.append(ChatMessage(role=role, content=content_repr))
        return messages

    # ------------------------------------------------------------------ trim
    def drop_oldest_turn_pair(self) -> bool:
        """Drop the oldest logical unit from the front; return whether anything
        was dropped.

        A "logical unit" is one of:
        * a plain user/model text turn, or
        * a model tool-call turn **together with** its matching tool-response
          turn (so we never corrupt the call↔response contract).

        Returns ``False`` when the history is already empty.
        """
        if not self._contents:
            return False

        first = self._contents[0]
        parts = first.parts or []
        has_function_call = any(p.function_call for p in parts)
        has_function_response = any(p.function_response for p in parts)

        # Always pop the front...
        self._contents.pop(0)

        # ...and if it was a model tool-call turn, also drop the tool-response
        # turn that immediately follows it (keeping them paired).
        if first.role == "model" and has_function_call:
            if self._contents:
                next_parts = self._contents[0].parts or []
                if any(p.function_response for p in next_parts):
                    self._contents.pop(0)
        # An orphaned tool-response at the front (call already gone) is dropped
        # alone — `has_function_response` needs no extra handling since we only
        # popped one Content above.
        _log.debug(
            "Dropped oldest history unit (was tool_call=%s, tool_response=%s); %d remain.",
            has_function_call,
            has_function_response,
            len(self._contents),
        )
        return True

    def _maybe_trim(self) -> None:
        """Enforce ``max_turns`` by dropping whole user/model turn-pairs.

        A "turn-pair" is a user ``Content`` followed by its model ``Content``
        (which may itself be tool calls). We never split a model tool-call from
        its matching tool-response: trimming stops at the first tool-response
        we would have to orphan.
        """
        if self._max_turns is None:
            return
        pairs_to_keep = self._max_turns
        # Count user/model turn-pairs (a tool-response user Content is NOT a turn).
        # We trim from the front while over the cap.
        while self._count_turn_pairs() > pairs_to_keep and len(self._contents) >= 2:
            if self._safe_to_drop_front():
                dropped = self._contents.pop(0)
                _log.debug("Trimmed history content (role=%s).", dropped.role)
            else:
                break  # would orphan a tool-response; stop trimming.

    def _count_turn_pairs(self) -> int:
        """Approximate user→model turn-pairs (ignoring tool-response Contents)."""
        count = 0
        for content in self._contents:
            if content.role == "user":
                # Only count as a turn if it's NOT a tool-response bundle.
                parts = content.parts or []
                if not any(p.function_response for p in parts):
                    count += 1
        return count

    def _safe_to_drop_front(self) -> bool:
        """``True`` if popping ``self._contents[0]`` won't orphan a tool call.

        The front is only safe to drop when it is a plain-text user or model
        turn (no function_response, and not a model tool-call whose response is
        still present later). Because responses are appended immediately after
        their call, dropping from the front past any tool-call/response cluster
        is only safe once we've passed it.
        """
        first = self._contents[0]
        parts = first.parts or []
        if any(p.function_response for p in parts):
            return False  # tool-response must stay adjacent to its call
        if first.role == "model" and any(p.function_call for p in parts):
            return False  # tool-call cluster — don't split from its response
        return True


__all__ = ["ConversationHistory"]
