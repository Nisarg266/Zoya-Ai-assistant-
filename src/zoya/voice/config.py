"""Configuration for the Voice Input Module.

Design note — *why a local settings model?*
--------------------------------------------
The rest of Zoya fuses every YAML block into the composite
:class:`~zoya.core.config.models.ZoyaSettings` inside ``core.config``. The voice
module deliberately does **not** touch any core file (see the philosophy stated
in :mod:`zoya.voice.exceptions`): instead it owns its own
:class:`VoiceSettings` model and reads the already-present ``voice:`` block of
``config/settings.yaml`` via :func:`load_voice_settings`.

This keeps the module fully self-contained — adding/removing voice never
requires editing ``core.config.yaml_settings`` / ``models`` / ``manager`` — while
still sharing the single source of truth (the same ``settings.yaml`` file) and
the same validation rigour (Pydantic, ``extra="forbid"``).

The block consumed here already lives in ``config/settings.yaml``::

    voice:
      enabled: true
      sample_rate: 16000
      model_size: base
      device: cpu
      compute_type: int8
      language: null        # auto-detect
      languages: [en, hi, gu]
      ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from zoya.core.exceptions import ConfigurationError
from zoya.core.paths import PATHS

# Allowed language codes for Speech-to-Text (Hindi, English, Gujarati).
# Anything Whisper auto-detects outside this set is discarded at transcription
# time (see :meth:`Transcriber.transcribe`).
SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "hi", "gu")


class VoiceSettings(BaseModel):
    """Tunable, validated settings for the Voice Input Module.

    Every field maps 1:1 to a key under ``voice:`` in ``settings.yaml`` and
    carries a sensible default, so a missing/empty block still yields a working
    configuration. ``extra="forbid"`` (the project-wide convention) turns a
    typo'd key into an immediate, attributed error rather than a silent miss.
    """

    model_config = {"extra": "forbid"}

    # --- Master switch -----------------------------------------------------
    #: When ``False`` the pipeline refuses to start (the feature is disabled).
    enabled: bool = Field(True, description="Master switch for voice input.")

    # --- Audio capture -----------------------------------------------------
    #: Sample rate in Hz. Faster-Whisper / Whisper expect 16 kHz mono PCM.
    sample_rate: int = Field(16000, ge=8000, le=48000)
    #: Seconds of audio captured per block. Smaller = lower latency, more CPU.
    block_duration: float = Field(0.5, gt=0, le=2.0)

    # --- Speech-to-Text model ---------------------------------------------
    #: Faster-Whisper checkpoint size: tiny | base | small | medium | large-v3
    model_size: str = Field("base", min_length=1)
    #: Inference device: ``cpu`` or ``cuda`` (GPU).
    device: str = Field("cpu")
    #: Quantisation / precision: int8 | int8_float16 | float16 | float32
    compute_type: str = Field("int8")

    # --- Language handling -------------------------------------------------
    #: ``None`` = let Whisper auto-detect; otherwise force ``en`` | ``hi`` | ``gu``.
    language: str | None = Field(None, description="Force a language, or auto-detect.")
    #: Languages the assistant will accept. Auto-detected speech outside this
    #: set is discarded so Zoya never acts on a language it does not "speak".
    languages: list[str] = Field(
        default_factory=lambda: list(SUPPORTED_LANGUAGES)
    )

    # --- Voice-activity detection / segmentation --------------------------
    #: Use Faster-Whisper's built-in Silero VAD to strip non-speech.
    vad_filter: bool = Field(True)
    #: Block RMS level at/above which audio counts as "speech".
    energy_threshold: float = Field(0.02, ge=0.0, le=1.0)
    #: Trailing silence (seconds) that finalises an utterance.
    silence_seconds: float = Field(0.6, ge=0.0, le=5.0)
    #: Discard segments shorter than this (noise / coughs / false starts).
    min_utterance_seconds: float = Field(0.5, ge=0.0)
    #: Force-finalise segments longer than this (prevents runaway capture).
    max_utterance_seconds: float = Field(15.0, gt=0.0)

    # ------------------------------------------------------------------ checks
    @field_validator("language")
    @classmethod
    def _language_is_known(cls, value: str | None) -> str | None:
        """A forced language must be one Zoya actually supports."""
        if value is None:
            return None
        value = value.strip().lower()
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language {value!r} is not supported. "
                f"Use one of {list(SUPPORTED_LANGUAGES)} or null for auto-detect."
            )
        return value

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "VoiceSettings":
        """Enforce invariants pydantic can't express per-field."""
        if self.language is not None and self.language not in self.languages:
            raise ValueError(
                f"Forced language {self.language!r} is not listed in "
                f"`languages` {self.languages!r}."
            )
        if self.min_utterance_seconds >= self.max_utterance_seconds:
            raise ValueError(
                "min_utterance_seconds must be strictly less than "
                "max_utterance_seconds."
            )
        if not self.languages:
            raise ValueError("`languages` must contain at least one code.")
        return self


def load_voice_settings(config_path: str | Path | None = None) -> VoiceSettings:
    """Read the ``voice:`` block from ``settings.yaml`` into :class:`VoiceSettings`.

    Reuses the project's canonical config path (:data:`zoya.core.paths.PATHS`)
    so it always reads the *same* file the rest of Zoya reads — just the voice
    slice of it. A missing file or missing block yields the model defaults; a
    malformed file / block raises an attributed
    :class:`~zoya.core.exceptions.ConfigurationError`.
    """
    path = Path(config_path).resolve() if config_path else PATHS.config_file

    if not path.exists():
        raw: dict[str, Any] = {}
    else:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Failed to parse YAML config at {path}",
                code="CFG_YAML_PARSE",
                context={"path": str(path)},
                cause=exc,
            ) from exc
        raw = data.get("voice", {}) if isinstance(data, dict) else {}

    try:
        return VoiceSettings(**raw)
    except Exception as exc:  # pydantic.ValidationError
        raise ConfigurationError(
            "Invalid values in the `voice:` block of settings.yaml",
            code="CFG_VOICE_INVALID",
            context={"path": str(path)},
            cause=exc,
        ) from exc


__all__ = ["VoiceSettings", "SUPPORTED_LANGUAGES", "load_voice_settings"]
