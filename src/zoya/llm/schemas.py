"""Core schemas for the Gemini LLM integration.

Defines the generic chat message structures that will be used across
the application (e.g., in the Memory subsystem later).
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """The role of the message sender."""

    USER = "user"
    MODEL = "model"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A generic chat message representing a single turn in a conversation."""

    role: Role
    content: str | list[dict[str, Any]] = Field(
        ...,
        description="Text content, or a list of rich content parts (like tool calls/results)",
    )

    def to_gemini_format(self) -> dict[str, Any]:
        """Convert to the format expected by the google-genai SDK."""
        # The new SDK takes "user" and "model" roles directly for contents.
        # System instructions are usually set at the client/config level.
        return {
            "role": self.role.value,
            "parts": [{"text": self.content}] if isinstance(self.content, str) else self.content,
        }
