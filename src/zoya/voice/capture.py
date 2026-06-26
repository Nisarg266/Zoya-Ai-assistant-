"""Asynchronous microphone capture for the Voice Input Module.

Responsibilities
----------------
* Open the default (or configured) input device at 16 kHz mono float32 — the
  format Faster-Whisper expects.
* Stream fixed-duration audio blocks to the ``asyncio`` loop via an
  :class:`asyncio.Queue`, bridging sounddevice's real-time **callback thread**
  and Zoya's event loop safely with :meth:`loop.call_soon_threadsafe`.
* Own the stream lifecycle through ``start()`` / ``stop()`` and an async
  iterator interface (:meth:`blocks`) so the :class:`~zoya.voice.listener.VoiceListener`
  can consume blocks with a plain ``async for``.

This layer does **no** speech detection — it simply turns the mic into an async
stream of numpy blocks. Utterance segmentation (energy VAD) lives one layer up
in the listener.

Dependency safety
-----------------
``sounddevice`` and ``numpy`` are imported lazily inside ``__init__`` /
:meth:`start`, so importing :mod:`zoya.voice` never requires the audio stack to
be installed. A missing stack surfaces as
:class:`~zoya.voice.exceptions.VoiceUnavailableError` on first use.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

from zoya.core.logging import get_logger
from zoya.voice.config import VoiceSettings
from zoya.voice.exceptions import MicrophoneError, VoiceUnavailableError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np  # noqa: F401

_log = get_logger("voice.capture")
_dbg = get_logger("voice.debug")  # TEMP DEBUG — remove after diagnosis

#: Sentinel pushed onto the queue by :meth:`AudioCapture.stop` to signal the
#: consumer (:meth:`blocks`) to terminate its ``async for`` cleanly.
_STOP = object()


class AudioCapture:
    """Bridges a PortAudio input stream into the asyncio world.

    Example::

        capture = AudioCapture(settings)
        await capture.start()
        try:
            async for block in capture.blocks():
                process(block)
        finally:
            await capture.stop()
    """

    def __init__(
        self,
        settings: VoiceSettings,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        device: int | None = None,
    ) -> None:
        self._settings = settings
        self._device = device
        # Resolve the loop lazily so construction outside a running loop works.
        self._loop: asyncio.AbstractEventLoop | None = loop
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stream: "object | None" = None
        self._running = False
        self._block_count = 0  # TEMP DEBUG — remove after diagnosis
        #: Number of audio frames per emitted block (sample_rate * block_duration).
        self.frames_per_block: int = max(
            1, int(settings.sample_rate * settings.block_duration)
        )

    # ------------------------------------------------------------------ props
    @property
    def is_running(self) -> bool:
        """``True`` between :meth:`start` and :meth:`stop`."""
        return self._running

    # ------------------------------------------------------------------ start
    async def start(self) -> None:
        """Open and start the input stream.

        Must be called from the same event loop that later consumes
        :meth:`blocks`. Raises :class:`VoiceUnavailableError` if the audio
        stack is missing, or :class:`MicrophoneError` if the device cannot be
        opened (busy / not found / permissions).
        """
        if self._running:
            return

        # Lazy imports — keep `import zoya.voice` dep-free.
        try:
            import numpy as np  # noqa: F401  (validates numpy presence)
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - env dependent
            raise VoiceUnavailableError(
                "sounddevice/numpy are not installed. "
                "Install them with: pip install sounddevice numpy",
                cause=exc,
            ) from exc

        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        try:
            self._stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frames_per_block,
                device=self._device,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            raise MicrophoneError(
                "Could not open microphone input stream",
                code="VOICE_MIC_OPEN",
                context={
                    "device": self._device,
                    "sample_rate": self._settings.sample_rate,
                },
                cause=exc,
            ) from exc

        self._running = True
        _log.info(
            "Microphone capture started | rate=%sHz block=%.2fs frames=%d",
            self._settings.sample_rate,
            self._settings.block_duration,
            self.frames_per_block,
        )

        # --- TEMP DEBUG (checkpoint 1: is the device actually open?) --------
        try:
            import sounddevice as _sd  # noqa
            default_in = _sd.query_devices(_sd.default.device[0])
            _dbg.info(
                "[VOICE-DBG][1.capture] stream active=%s | device=%r | "
                "default_input=%r | active_stream=%s",
                bool(getattr(self._stream, "active", None)),
                self._device,
                default_in.get("name") if isinstance(default_in, dict) else default_in,
                bool(getattr(self._stream, "active", None)),
            )
        except Exception as exc:  # pragma: no cover - debug only
            _dbg.warning("[VOICE-DBG][1.capture] could not query device info: %s", exc)

    # ------------------------------------------------------------------ stop
    async def stop(self) -> None:
        """Stop the stream and unblock any pending :meth:`blocks` consumer."""
        if not self._running:
            return
        self._running = False

        # Tell the consumer to stop, then close the hardware stream.
        self._enqueue_sentinel()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover - defensive
                _log.warning("Error while closing mic stream: %s", exc)
            self._stream = None

        _log.info("Microphone capture stopped.")

    # ----------------------------------------------------------- async stream
    async def blocks(self) -> AsyncIterator["np.ndarray"]:
        """Yield mono float32 audio blocks until :meth:`stop` is called.

        Each value is a 1-D ``numpy`` array of length ``frames_per_block`` (the
        final block of a run may be shorter if the device under-fills it).
        """
        while True:
            item = await self._queue.get()
            if item is _STOP:
                return
            yield item  # type: ignore[misc]

    # --------------------------------------------------------- PortAudio cb
    def _on_audio(
        self,
        indata,  # type: ignore[no-untyped-def]
        frames: int,
        time_info,  # type: ignore[no-untyped-def]
        status,  # type: ignore[no-untyped-def]
    ) -> None:
        """PortAudio callback — runs on sounddevice's high-priority thread.

        We must not block here. We copy the (read-only) buffer, then hand it to
        the asyncio loop thread-safely. ``status`` carries under/overflow flags
        worth logging but never worth crashing over.
        """
        if status:
            _log.debug("Audio stream status: %s", status)

        try:
            # ``indata`` is shape (frames, 1); squeeze to a 1-D mono buffer.
            block = indata[:, 0].copy()
        except Exception:  # pragma: no cover - defensive against odd shapes
            block = indata.copy().reshape(-1)

        # --- TEMP DEBUG (checkpoint 1: is the mic physically delivering?) ---
        # Computed on the audio thread; logging is thread-safe in stdlib.
        self._block_count += 1
        try:
            import numpy as _np  # noqa
            _rms = float(_np.sqrt(_np.mean(_np.square(block))))
            _peak = float(_np.max(_np.abs(block)))
        except Exception:  # pragma: no cover - debug only
            _rms = _peak = float("nan")
        _dbg.info(
            "[VOICE-DBG][1.capture] block #%d | frames=%d | rms=%.5f | "
            "peak=%.5f | status=%s",
            self._block_count, frames, _rms, _peak, status or "ok",
        )

        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._queue.put_nowait, block)
        except RuntimeError:
            # Loop closed between the check above and the call — drop the block.
            pass

    def _enqueue_sentinel(self) -> None:
        """Push the stop sentinel onto the queue (thread-safe)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            try:
                self._queue.put_nowait(_STOP)
            except Exception:  # pragma: no cover
                pass
            return
        try:
            loop.call_soon_threadsafe(self._queue.put_nowait, _STOP)
        except RuntimeError:  # pragma: no cover - loop already torn down
            pass


__all__ = ["AudioCapture"]
