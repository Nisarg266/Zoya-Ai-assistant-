"""Public facade for the Voice Input Module and its Brain integration.

:class:`VoiceInput` is the single class the rest of Zoya should use. It hides
the capture → segment → transcribe pipeline behind two small surfaces:

* :meth:`VoiceInput.utterances` — an async generator of recognised text. Use
  this when you want to handle the text yourself.
* :meth:`VoiceInput.run` — a ready-made **continuous voice session** that pipes
  every recognised utterance straight into the Gemini Brain
  (:class:`~zoya.llm.facade.ZoyaBrain`) and writes the Brain's answer to the
  console/log.

.. note::
    This module deliberately implements **only the input side** (Speech-to-Text).
    Text-to-Speech (the Brain talking back) is intentionally out of scope here
    and will live in a future ``tts`` module.

Decoupling from the Brain
-------------------------
Rather than importing :class:`ZoyaBrain` directly, :meth:`run` accepts any
object exposing ``async def chat(self, prompt: str) -> str`` (duck typing). This
keeps the voice package free of an ``llm`` import, trivially unit-testable with
a fake brain, and still accepts the real ``ZoyaBrain`` at runtime.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator, Protocol

from zoya.core.logging import get_logger
from zoya.voice.capture import AudioCapture
from zoya.voice.config import VoiceSettings, load_voice_settings
from zoya.voice.exceptions import VoiceError
from zoya.voice.listener import VoiceListener
from zoya.voice.transcriber import Transcriber

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zoya.llm.facade import ZoyaBrain

_log = get_logger("voice.pipeline")


class _BrainLike(Protocol):
    """Structural type for anything we can feed recognised text to."""

    async def chat(self, prompt: str) -> str: ...


class VoiceInput:
    """High-level entry point for voice-driven input.

    Args:
        settings: Voice configuration (load via :func:`load_voice_settings`).
        transcriber / capture / listener: optional injection seams, primarily
            for tests. When omitted, sensible default components are built from
            ``settings``.
    """

    def __init__(
        self,
        settings: VoiceSettings | None = None,
        *,
        transcriber: Transcriber | None = None,
        capture: object | None = None,
        listener: VoiceListener | None = None,
    ) -> None:
        self.settings = settings or load_voice_settings()
        self._transcriber = transcriber or Transcriber(self.settings)
        self._listener = listener or VoiceListener(
            self.settings,
            transcriber=self._transcriber,
            capture=capture,  # type: ignore[arg-type]
        )

    # ----------------------------------------------------------- raw utterances
    async def utterances(self) -> AsyncIterator[str]:
        """Yield recognised text, one string per utterance, forever.

        Thin pass-through to :meth:`VoiceListener.utterances`; kept on the
        facade so callers need only know about :class:`VoiceInput`.
        """
        async for text in self._listener.utterances():
            yield text

    # --------------------------------------------------------- brain session
    async def run(
        self,
        brain: "_BrainLike | ZoyaBrain | None" = None,
        *,
        on_response: "object | None" = None,
    ) -> None:
        """Run a continuous voice session.

        For each recognised utterance:

        1. The text is sent to ``brain.chat(text)`` (the Gemini Brain).
        2. The Brain's reply is passed to ``on_response`` if provided (defaults
           to logging + printing), and is **not** spoken — TTS is a separate
           future module.

        If ``brain`` is ``None`` the recognised text is still surfaced (logged
        + printed), which is handy for testing the mic + STT pipeline on its
        own.

        The loop runs until cancelled (``Ctrl+C`` / task cancellation). Every
        non-fatal :class:`VoiceError` during capture or transcription is logged
        and the session continues — a single bad block never kills Zoya's ears.
        """
        if not self.settings.enabled:
            _log.warning("Voice input is disabled in settings (voice.enabled=false).")
            return

        handler = on_response or _default_response_handler
        _log.info(
            "Voice session starting | languages=%s model=%s",
            self.settings.languages, self.settings.model_size,
        )

        async for text in self.utterances():
            _log.info("Heard: %r", text)
            await handler(text, source="user")

            if brain is None:
                continue

            try:
                reply = await brain.chat(text)
            except Exception as exc:
                # The Brain already self-isolates LLM errors, but we guard the
                # seam so a stray exception can't crash the listening loop.
                _log.error("Brain failed to respond to %r: %s", text, exc)
                reply = f"(brain error: {exc})"

            _log.info("Zoya: %s", reply)
            await handler(reply, source="zoya")

    # ------------------------------------------------------------------ close
    async def aclose(self) -> None:
        """Release any underlying resources (currently nothing owned)."""
        _log.debug("VoiceInput closed.")


async def _default_response_handler(text: str, *, source: str) -> None:
    """Default sink for :meth:`VoiceInput.run`: print + log each line."""
    label = "You" if source == "user" else "Zoya"
    line = f"{label}: {text}"
    print(line, flush=True)
    _log.info(line)


__all__ = ["VoiceInput", "load_voice_settings"]
