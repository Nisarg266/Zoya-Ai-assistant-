"""Tests for :mod:`zoya.voice.config`.

These run without the optional audio/STT stack — they only exercise the Pydantic
model + YAML loader, which depend on pydantic/PyYAML (already required).
"""

import pytest

from zoya.core.exceptions import ConfigurationError
from zoya.voice.config import SUPPORTED_LANGUAGES, VoiceSettings, load_voice_settings


# --------------------------------------------------------------- defaults
def test_defaults_match_supported_languages():
    s = VoiceSettings()
    assert s.enabled is True
    assert s.sample_rate == 16000
    assert s.model_size == "base"
    assert s.language is None  # auto-detect by default
    assert s.languages == ["en", "hi", "gu"]
    assert s.languages == list(SUPPORTED_LANGUAGES)
    assert s.min_utterance_seconds < s.max_utterance_seconds


# --------------------------------------------------------------- language
def test_forced_language_must_be_supported():
    s = VoiceSettings(language="hi")
    assert s.language == "hi"


@pytest.mark.parametrize("bad", ["fr", "ja", "ENGLISH", "", "  "])
def test_unknown_forced_language_rejected(bad):
    with pytest.raises(Exception):
        VoiceSettings(language=bad)


def test_forced_language_must_be_in_languages_list():
    # `language` is valid (hi) but removed from the allowed list -> error.
    with pytest.raises(Exception):
        VoiceSettings(language="hi", languages=["en", "gu"])


# --------------------------------------------------------------- invariants
def test_min_must_be_less_than_max():
    with pytest.raises(Exception):
        VoiceSettings(min_utterance_seconds=5.0, max_utterance_seconds=5.0)


def test_empty_languages_rejected():
    with pytest.raises(Exception):
        VoiceSettings(languages=[])


def test_extra_keys_forbidden():
    with pytest.raises(Exception):
        VoiceSettings(unknown_key=True)


def test_numeric_bounds_enforced():
    with pytest.raises(Exception):
        VoiceSettings(sample_rate=1000)        # below 8000
    with pytest.raises(Exception):
        VoiceSettings(block_duration=0)        # must be > 0
    with pytest.raises(Exception):
        VoiceSettings(energy_threshold=-0.1)   # below 0


# --------------------------------------------------------------- loader
def test_load_from_yaml(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        "voice:\n"
        "  enabled: true\n"
        "  model_size: small\n"
        "  language: gu\n"
        "  languages: [en, hi, gu]\n",
        encoding="utf-8",
    )
    s = load_voice_settings(cfg)
    assert s.model_size == "small"
    assert s.language == "gu"


def test_load_defaults_when_block_missing(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("automation:\n  default_type_interval: 0.0\n", encoding="utf-8")
    s = load_voice_settings(cfg)
    assert s.model_size == "base"   # default applied
    assert s.enabled is True


def test_load_defaults_when_file_missing(tmp_path):
    s = load_voice_settings(tmp_path / "does_not_exist.yaml")
    assert s.model_size == "base"


def test_load_malformed_yaml_raises(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("voice: [unbalanced\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_voice_settings(cfg)


def test_load_invalid_values_raise(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("voice:\n  sample_rate: 100\n", encoding="utf-8")  # below 8000
    with pytest.raises(ConfigurationError):
        load_voice_settings(cfg)
