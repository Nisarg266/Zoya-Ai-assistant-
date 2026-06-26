"""Demo script for the Zoya Voice Input Module.

Run this to test microphone capture + Faster-Whisper Speech-to-Text.

Two modes:

* **STT-only** (no API key): prints every recognised utterance. Use this to
  verify the mic + Whisper pipeline end-to-end.
* **Brain-coupled** (with GEMINI_API_KEY): pipes each utterance into the Gemini
  Brain and prints Zoya's text reply (no TTS — voice output is a future module).

Usage::

    python scripts\\demo_voice.py

Press Ctrl+C to stop. Requires the optional audio stack::

    pip install faster-whisper sounddevice numpy
"""

import asyncio
import logging
import sys
from pathlib import Path

# Make `import zoya` work without an editable install.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from zoya.core.config import load_settings
from zoya.core.logging import setup_logging
from zoya.voice import VoiceInput, VoiceSettings, load_voice_settings
from zoya.voice.exceptions import VoiceError

setup_logging(level=logging.INFO)
_log = logging.getLogger("zoya.scripts.demo_voice")


async def _build_brain():
    """Return a ZoyaBrain if an API key is configured, else None."""
    try:
        from zoya.llm.facade import ZoyaBrain
    except Exception as exc:  # pragma: no cover - llm optional in this demo
        _log.warning("Could not import ZoyaBrain: %s", exc)
        return None

    settings = load_settings()
    if not settings.app.has_api_key:
        _log.warning(
            "GEMINI_API_KEY not set — running in STT-only mode (no Brain)."
        )
        return None
    return ZoyaBrain(settings=settings, tools=[])


async def main() -> None:
    voice_settings: VoiceSettings = load_voice_settings()
    if not voice_settings.enabled:
        print("Voice input is disabled in config/settings.yaml (voice.enabled=false).")
        return

    print("=" * 60)
    print(" Zoya Voice Input — demo")
    print(f" Languages : {voice_settings.languages}")
    print(f" Model     : faster-whisper/{voice_settings.model_size}")
    print(" Speak now. Press Ctrl+C to stop.")
    print("=" * 60)

    brain = await _build_brain()
    voice = VoiceInput(voice_settings)

    try:
        await voice.run(brain)
    except VoiceError as exc:
        _log.error("Voice module error: %s", exc)
    except KeyboardInterrupt:
        pass
    finally:
        await voice.aclose()
        print("\nStopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
