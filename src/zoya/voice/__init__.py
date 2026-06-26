"""Voice Input Module for the Zoya AI Assistant.

Provides continuous, asynchronous Speech-to-Text (microphone → recognised text)
powered by `Faster-Whisper <https://github.com/SYSTRAN/faster-whisper>`_.

Supported languages: **English (en)**, **Hindi (hi)** and **Gujarati (gu)**.

Architecture (data flow)::

    Microphone ──► AudioCapture ──► VoiceListener ──► Transcriber ──► text
                   (sounddevice)    (VAD segmentation) (Faster-Whisper)
                                                                         │
                                                                         ▼
                                                              Gemini Brain (ZoyaBrain)

The whole stack is ``async``: capture blocks flow through an ``asyncio.Queue``
and each blocking Whisper call is offloaded to a worker thread, so the
microphone never stalls while a segment is being decoded.

Quick start
-----------
    from zoya.voice import VoiceInput, load_voice_settings

    settings = load_voice_settings()
    voice = VoiceInput(settings)
    async for text in voice.utterances():
        print("Heard:", text)

Or wire it straight into the Brain::

    from zoya.llm.facade import ZoyaBrain
    brain = ZoyaBrain(...)
    await VoiceInput(settings).run(brain)   # text in, text out (no TTS)

Dependency safety
-----------------
``faster-whisper``, ``sounddevice`` and ``numpy`` are **optional** dependencies
imported lazily. Importing :mod:`zoya.voice` always succeeds; a missing stack
surfaces as :class:`VoiceUnavailableError` only when a feature is actually used.
"""

from __future__ import annotations

from zoya.voice.capture import AudioCapture
from zoya.voice.config import (
    SUPPORTED_LANGUAGES,
    VoiceSettings,
    load_voice_settings,
)
from zoya.voice.exceptions import (
    MicrophoneError,
    TranscriptionError,
    VoiceError,
    VoiceUnavailableError,
)
from zoya.voice.listener import VoiceListener
from zoya.voice.pipeline import VoiceInput
from zoya.voice.transcriber import Transcriber

__all__ = [
    # config
    "VoiceSettings",
    "SUPPORTED_LANGUAGES",
    "load_voice_settings",
    # components
    "AudioCapture",
    "Transcriber",
    "VoiceListener",
    "VoiceInput",
    # errors
    "VoiceError",
    "VoiceUnavailableError",
    "MicrophoneError",
    "TranscriptionError",
]
