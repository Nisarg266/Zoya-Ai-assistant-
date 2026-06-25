"""Gemini Integration module (The Brain).

This module handles asynchronous communication with the Google Gemini API,
managing chat sessions, and mapping the Desktop Automation tool registry
into Gemini's function calling schema.
"""

from .client import GeminiClient
from .facade import ZoyaBrain
from .schemas import ChatMessage, Role

__all__ = [
    "GeminiClient",
    "ZoyaBrain",
    "ChatMessage",
    "Role",
]
