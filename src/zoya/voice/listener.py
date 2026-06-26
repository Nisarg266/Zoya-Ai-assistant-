"""Continuous listening: turns an audio stream into recognised text.

The :class:`VoiceListener` is the orchestration heart of the module. It wires
:class:`~zoya.voice.capture.AudioCapture` (the mic) to
:class:`~zoya.voice.transcriber.Transcriber` (Faster-Whisper) and exposes a
single high-level primitive — an **async generator of recognised utterances**:

    async for text in listener.utterances():
        do_something_with(text)

Utterance segmentation (energy-based VAD)
-----------------------------------------
Whisper needs complete-ish utterances, not a dribble of 0.5 s blocks, so the
listener runs a small, deterministic state machine over each block:

* A block whose RMS >= ``energy_threshold`` counts as **speech**; below it is
  **silence**.
* Speech starts a recording; silence *while* recording starts a silence timer.
* An utterance is **finalised** when the trailing silence reaches
  ``silence_seconds`` **or** the recording exceeds ``max_utterance_seconds``.
* Finalised segments shorter than ``min_utterance_seconds`` are discarded
  (coughs / false starts), keeping the Brain free of noise.

The whole loop is ``async``: capture blocks arrive asynchronously, and each
transcription runs in a worker thread (see :mod:`zoya.voice.transcriber`) so the
mic never stalls while Whisper is thinking.

Testability
-----------
The dependency on :class:`AudioCapture` is injected, so tests can feed a fake
capture that replays a scripted sequence of blocks and assert exactly which
utterances the listener emits.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol

from zoya.core.logging import get_logger
from zoya.voice.capture import AudioCapture
from zoya.voice.config import VoiceSettings
from zoya.voice.transcriber import Transcriber

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np  # noqa: F401

_log = get_logger("voice.listener")
_dbg = get_logger("voice.debug")  # TEMP DEBUG — remove after diagnosis


class _CaptureLike(Protocol):
    """Structural type accepted in place of :class:`AudioCapture` (for tests)."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def blocks(self) -> AsyncIterator["np.ndarray"]: ...


def _rms(block: "np.ndarray") -> float:
    """Root-mean-square level of a block — the speech/silence discriminator.

    Kept dependency-light: ``numpy`` is imported lazily so this module imports
    cleanly without the audio stack. RMS is a robust loudness proxy on the
    already-normalised [-1, 1] float32 PCM coming from the capture.
    """
    import numpy as np

    arr = np.asarray(block, dtype="float32")
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


class VoiceListener:
    """Continuously listens to the mic and yields transcribed utterances."""

    def __init__(
        self,
        settings: VoiceSettings,
        *,
        transcriber: Transcriber | None = None,
        capture: _CaptureLike | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._settings = settings
        # Build a real capture only when none is injected (tests pass a fake).
        self._capture: _CaptureLike = capture or AudioCapture(
            settings, loop=loop
        )
        self._transcriber = transcriber or Transcriber(settings)

    # ------------------------------------------------------------------ public
    async def utterances(self) -> AsyncIterator[str]:
        """Yield recognised text, one string per detected utterance.

        Runs forever until the caller ``break``s out of the ``async for`` (or
        cancels the task). Capture/transcriber lifecycle is handled internally
        and always cleaned up via ``finally``.
        """
        await self._capture.start()
        _log.info(
            "Continuous listening started | threshold=%.3f silence=%.2fs",
            self._settings.energy_threshold,
            self._settings.silence_seconds,
        )
        try:
            async for text in self._segment_and_transcribe():
                if text:
                    yield text
        finally:
            await self._capture.stop()
            _log.info("Continuous listening stopped.")

    # --------------------------------------------------- segmentation core
    async def _segment_and_transcribe(self) -> AsyncIterator[str]:
        """The VAD state machine + transcription loop.

        Yields recognised text for every utterance that survives the
        min-duration filter and language gating.

        Durations are derived from the **block count** (``blocks *
        block_duration``) rather than wall-clock time. This makes segmentation
        deterministic and clock-jitter-free — crucial for tests and robust on a
        busy machine where ``asyncio`` scheduling can stretch real elapsed time.
        """
        s = self._settings
        block_seconds = s.block_duration

        recording = False
        buffer: list[Any] = []
        silence_blocks = 0
        recorded_blocks = 0
        block_idx = 0  # TEMP DEBUG

        async for block in self._capture.blocks():
            block_idx += 1
            rms = _rms(block)
            loud = rms >= s.energy_threshold

            # --- TEMP DEBUG (checkpoints 2 & 3: RMS + VAD class) ------------
            _dbg.info(
                "[VOICE-DBG][2-3.vad] block #%d | rms=%.5f | thr=%.5f -> %s",
                block_idx, rms, s.energy_threshold,
                "SPEECH" if loud else "silence",
            )

            # --- start recording on the first loud block after idle -------
            if not recording:
                if not loud:
                    continue  # idle: ignore leading silence
                recording = True
                silence_blocks = 0
                recorded_blocks = 0
                buffer = []
                _dbg.info("[VOICE-DBG][3.vad] >>> SPEECH STARTED (block #%d)", block_idx)

            # --- accumulate the block (speech or trailing silence) --------
            buffer.append(block)
            recorded_blocks += 1
            if loud:
                silence_blocks = 0
            else:
                silence_blocks += 1

            trailing_silence = silence_blocks * block_seconds
            utterance_duration = recorded_blocks * block_seconds

            # --- finalise? ------------------------------------------------
            if not (
                trailing_silence >= s.silence_seconds
                or utterance_duration >= s.max_utterance_seconds
            ):
                continue

            # --- TEMP DEBUG (checkpoint 4: utterance duration) -------------
            _dbg.info(
                "[VOICE-DBG][4.utterance] FINALISE | duration=%.2fs | "
                "recorded_blocks=%d | trailing_silence=%.2fs",
                utterance_duration, recorded_blocks, trailing_silence,
            )
            text = await self._finalise(buffer, utterance_duration)
            recording = False
            buffer = []
            silence_blocks = 0
            recorded_blocks = 0
            if text:
                _dbg.info("[VOICE-DBG][4.utterance] EMITTED text=%r", text)
                yield text

    async def _finalise(
        self, buffer: list[Any], spoken_duration: float
    ) -> str:
        """Concatenate buffered blocks, gate by duration, and transcribe."""
        s = self._settings

        if spoken_duration < s.min_utterance_seconds:
            # --- TEMP DEBUG (checkpoint 6: why discarded) ------------------
            _dbg.info(
                "[VOICE-DBG][6.discard] DISCARDED too-short | duration=%.2fs "
                "< min=%.2fs", spoken_duration, s.min_utterance_seconds,
            )
            return ""

        if not buffer:
            return ""

        try:
            import numpy as np

            audio = np.concatenate(buffer)
        except Exception as exc:  # pragma: no cover - defensive
            _log.error("Failed to assemble utterance audio: %s", exc)
            return ""

        _log.debug(
            "Finalising utterance (%.2fs, %d samples) for transcription.",
            spoken_duration, int(audio.size),
        )
        return await self._transcriber.transcribe(audio)


__all__ = ["VoiceListener", "_rms"]
