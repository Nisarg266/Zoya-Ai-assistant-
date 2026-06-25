"""Core schemas for the Gemini LLM integration.

Two concerns live here:

1. **Transport shapes** — :class:`Role` and :class:`ChatMessage` describe a
   single conversational turn in a provider-agnostic way. The actual conversion
   to the SDK's ``types.Content`` happens in :mod:`zoya.llm.history` so this
   module stays free of any ``google.genai`` import (useful for the Memory
   subsystem later).

2. **Streaming events** — the :data:`BrainEvent` union is what
   :meth:`ZoyaBrain.chat_stream` yields. A consumer does::

       async for ev in brain.chat_stream(prompt):
           match ev:
               case TextDelta(text):   ...
               case ToolCallStarted(): ...
               case TurnComplete(text): ...
   """

from __future__ import annotations

from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, Field


class Role(str, Enum):
    """The role of a message sender in a conversation."""

    USER = "user"
    MODEL = "model"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A provider-agnostic chat message representing a single turn.

    ``content`` is either plain text or a list of rich part dicts
    (tool calls / tool results) — mirroring how multi-part model turns look.
    """

    role: Role
    content: str | list[dict[str, Any]] = Field(
        ...,
        description="Text content, or a list of rich content parts (tool calls/results).",
    )

    def to_gemini_format(self) -> dict[str, Any]:
        """Convert to the dict shape the ``google-genai`` SDK accepts."""
        return {
            "role": self.role.value,
            "parts": (
                [{"text": self.content}]
                if isinstance(self.content, str)
                else self.content
            ),
        }


# ===========================================================================
# Streaming events
#
# chat_stream() yields one of these per "thing that happened" during a turn.
# They are frozen (immutable) so they're safe to forward to a UI / queue.
# ===========================================================================
class _BaseEvent(BaseModel):
    """Common base: every event is frozen + carries a discriminator ``kind``."""

    model_config = {"frozen": True}


class TextDelta(_BaseEvent):
    """An incremental chunk of the final answer (streamed token(s))."""

    kind: str = "text_delta"
    text: str


class ToolCallStarted(_BaseEvent):
    """The model requested a tool call."""

    kind: str = "tool_call_started"
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolCallFinished(_BaseEvent):
    """A tool finished; carries a concise result summary for the UI."""

    kind: str = "tool_call_finished"
    name: str
    success: bool
    summary: str


class TurnComplete(_BaseEvent):
    """The whole turn finished; ``text`` is the fully-assembled answer."""

    kind: str = "turn_complete"
    text: str


class ErrorEvent(_BaseEvent):
    """A failure surfaced during streaming (mapped LLM error)."""

    kind: str = "error"
    error_type: str
    message: str


#: Discriminated union yielded by :meth:`ZoyaBrain.chat_stream`.
BrainEvent = Union[
    TextDelta,
    ToolCallStarted,
    ToolCallFinished,
    TurnComplete,
    ErrorEvent,
]


__all__ = [
    "Role",
    "ChatMessage",
    # events
    "TextDelta",
    "ToolCallStarted",
    "ToolCallFinished",
    "TurnComplete",
    "ErrorEvent",
    "BrainEvent",
]
