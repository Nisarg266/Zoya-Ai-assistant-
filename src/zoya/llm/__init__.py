"""Gemini Integration module (The Brain).

Handles asynchronous communication with the Google Gemini API (google-genai
2.x), manages conversation history, exposes both a non-streaming and a
streaming chat interface, retries transient failures, and maps provider errors
onto Zoya's :class:`~zoya.core.exceptions.LLMError` hierarchy.
"""

from zoya.llm.client import DEFAULT_SYSTEM_INSTRUCTION, GeminiClient
from zoya.llm.errors import is_retryable, map_sdk_error
from zoya.llm.facade import ZoyaBrain
from zoya.llm.history import ConversationHistory
from zoya.llm.retry import with_retry
from zoya.llm.schemas import (
    BrainEvent,
    ChatMessage,
    ErrorEvent,
    Role,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)

__all__ = [
    # client
    "GeminiClient",
    "DEFAULT_SYSTEM_INSTRUCTION",
    # facade
    "ZoyaBrain",
    # history
    "ConversationHistory",
    # schemas
    "ChatMessage",
    "Role",
    "BrainEvent",
    "TextDelta",
    "ToolCallStarted",
    "ToolCallFinished",
    "TurnComplete",
    "ErrorEvent",
    # resilience
    "with_retry",
    "map_sdk_error",
    "is_retryable",
]
