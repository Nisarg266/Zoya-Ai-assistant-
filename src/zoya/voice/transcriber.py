"""Asynchronous Speech-to-Text via Faster-Whisper.

Faster-Whisper (`CTranslate2`-backed Whisper) is fast and accurate but its
:meth:`transcribe` call is **blocking** and CPU/GPU-bound. To stay compatible
with Zoya's ``asyncio`` event loop we run every transcription in a worker
thread through :func:`asyncio.to_thread`, so the mic keeps capturing and the
rest of the assistant stays responsive while a segment is being decoded.

Dependency safety
-----------------
``faster_whisper`` and ``numpy`` are imported **lazily** (inside ``__init__`` /
the call site) so that merely importing :mod:`zoya.voice` never fails when the
optional STT stack is not installed. A missing dependency surfaces as a clear
:class:`~zoya.voice.exceptions.VoiceUnavailableError` only when the feature is
actually used — matching the contract documented in
:mod:`zoya.voice.exceptions`.

Languages
---------
Hindi (``hi``), English (``en``) and Gujarati (``gu``) are the supported set.
When ``language`` is left to auto-detect, any detected language *outside* the
configured ``languages`` list is discarded so Zoya never acts on speech it
cannot meaningfully handle.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from zoya.core.logging import get_logger
from zoya.voice.config import VoiceSettings
from zoya.voice.exceptions import TranscriptionError, VoiceUnavailableError

# ``numpy`` is only needed for type hints at import time; never imported for real
# at module scope so the package loads without the STT stack installed.
if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # noqa: F401

_log = get_logger("voice.transcriber")
_dbg = get_logger("voice.debug")  # TEMP DEBUG — remove after diagnosis

#: Beam-search width. 5 is a good speed/accuracy trade-off for desktop use.
_BEAM_SIZE = 5


class Transcriber:
    """Thin async wrapper around :class:`faster_whisper.WhisperModel`.

    The (relatively expensive) model load happens once, in the constructor, so
    the listener can call :meth:`transcribe` repeatedly with no per-call setup
    cost.
    """

    def __init__(self, settings: VoiceSettings) -> None:
        self._settings = settings

        # Lazy import — keeps `import zoya.voice` working without the dep.
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise VoiceUnavailableError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper",
                cause=exc,
            ) from exc

        self._language: str | None = settings.language
        self._allowed = set(settings.languages)

        _log.info(
            "Loading Faster-Whisper model | size=%s device=%s compute=%s",
            settings.model_size, settings.device, settings.compute_type,
        )
        try:
            self._model = WhisperModel(
                settings.model_size,
                device=settings.device,
                compute_type=settings.compute_type,
            )
        except Exception as exc:
            raise TranscriptionError(
                "Failed to load Faster-Whisper model",
                code="VOICE_STT_LOAD",
                context={
                    "model_size": settings.model_size,
                    "device": settings.device,
                    "compute_type": settings.compute_type,
                },
                cause=exc,
            ) from exc
        _log.info("Faster-Whisper model ready.")

    # ------------------------------------------------------------------ public
    def is_language_allowed(self, language: str | None) -> bool:
        """``True`` when ``language`` is permitted by the config.

        An explicit (forced) language always passes — the user asked for it.
        ``None`` (Whisper could not decide) is accepted so silence/noise at the
        very end of an utterance is not treated as a hard failure.
        """
        if language is None:
            return True
        return language in self._allowed

    async def transcribe(self, audio: "np.ndarray | Any") -> str:
        """Transcribe a mono 16 kHz float32 waveform to text (async).

        Args:
            audio: 1-D ``numpy`` float32 array (mono, ``sample_rate`` Hz) as
                produced by :class:`~zoya.voice.capture.AudioCapture`.

        Returns:
            The recognised text (stripped). Empty string when nothing was
            recognised, the segment was too short, or the detected language is
            not in the allowed set.

        Raises:
            TranscriptionError: if the underlying model call fails.
        """
        if audio is None:
            return ""
        try:
            length = len(audio)  # type: ignore[arg-type]
        except TypeError:
            length = 0
        if length == 0:
            return ""

        try:
            text, detected = await asyncio.to_thread(self._transcribe_sync, audio)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                "Speech-to-text failed",
                code="VOICE_STT_FAIL",
                context={"samples": length},
                cause=exc,
            ) from exc

        # Language gating (only when auto-detecting).
        if self._language is None and not self.is_language_allowed(detected):
            _dbg.info(
                "[VOICE-DBG][6.discard] DISCARDED language | detected=%r not in "
                "allowed=%s | raw_text=%r",
                detected, sorted(self._allowed), text,
            )
            return ""

        _log.info(
            "Transcribed [lang=%s]: %r", detected, text[:120] if text else text,
        )
        return text

    # ------------------------------------------------------------------ private
    def _transcribe_sync(
        self, audio: "np.ndarray"
    ) -> tuple[str, str | None]:
        """Blocking transcription (runs in a worker thread).

        Returns ``(text, detected_language)``. ``detected_language`` is the
        code Whisper reported (e.g. ``"hi"``); ``None`` when forced.
        """
        segments, info = self._model.transcribe(
            audio,
            language=self._language,          # None => auto-detect
            beam_size=_BEAM_SIZE,
            vad_filter=self._settings.vad_filter,
            vad_parameters=None,              # use Whisper's VAD defaults
        )
        # ``segments`` is a *lazy* generator — drain it now (still inside the
        # worker thread) so decoding completes before we return to the loop.
        seg_list = list(segments)
        text = " ".join(segment.text.strip() for segment in seg_list).strip()
        detected = getattr(info, "language", None) if info is not None else None

        # --- TEMP DEBUG (checkpoint 5: raw result BEFORE filtering) --------
        _dbg.info(
            "[VOICE-DBG][5.stt] RAW | segments=%d | detected_lang=%s | "
            "forced_lang=%s | vad_filter=%s | text=%r",
            len(seg_list), detected, self._language,
            self._settings.vad_filter, text,
        )
        return text, detected


__all__ = ["Transcriber"]
