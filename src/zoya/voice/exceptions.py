"""Exception hierarchy for the Voice Input Module.

All voice errors descend from :class:`VoiceError`, which itself subclasses the
project-wide :class:`~zoya.core.exceptions.ZoyaError`. That means the rest of
Zoya can catch any voice failure with a single ``except ZoyaError`` (or
``except VoiceError`` for voice-specific handling), and every error carries the
rich ``code`` / ``context`` diagnostics defined by
:class:`~zoya.core.exceptions.ZoyaError`.

These are defined locally (not in ``zoya.core.exceptions``) so that adding the
voice module needs **no changes to any core file** — keeping the module fully
self-contained.
"""

from __future__ import annotations

from zoya.core.exceptions import ZoyaError


class VoiceError(ZoyaError):
    """Base class for every voice-input failure."""

    default_code = "VOICE_ERROR"


class VoiceUnavailableError(VoiceError):
    """A required capability is missing.

    Raised when ``faster-whisper`` / ``sounddevice`` / ``numpy`` is not
    installed, or no microphone device is available. The module imports safely
    regardless; this is only raised when a feature is *used* without its deps.
    """

    default_code = "VOICE_UNAVAILABLE"


class MicrophoneError(VoiceError):
    """Microphone capture failed (device busy, not found, underflow, ...)."""

    default_code = "VOICE_MIC"


class TranscriptionError(VoiceError):
    """Speech-to-text failed after the audio was captured."""

    default_code = "VOICE_STT"


__all__ = [
    "VoiceError",
    "VoiceUnavailableError",
    "MicrophoneError",
    "TranscriptionError",
]
