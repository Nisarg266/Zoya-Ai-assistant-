"""Tests for :mod:`zoya.voice.listener` — the VAD segmentation state machine.

These exercise the pure segmentation + transcription-wiring logic using a fake
capture (scripted blocks) and a fake transcriber, so no microphone and no
Faster-Whisper model are required. ``numpy`` *is* needed to build audio blocks,
so each test skips gracefully when it is not installed.
"""

from typing import AsyncIterator

import pytest

np = pytest.importorskip("numpy")

from zoya.voice.config import VoiceSettings  # noqa: E402
from zoya.voice.listener import VoiceListener, _rms  # noqa: E402


# ----------------------------------------------------------- test doubles
class FakeCapture:
    """Replays a scripted list of blocks, then ends the stream."""

    def __init__(self, blocks):
        self._blocks = blocks

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def blocks(self) -> AsyncIterator:
        for block in self._blocks:
            yield block


class FakeTranscriber:
    """Records every call and returns scripted text."""

    def __init__(self, text: str = "hello"):
        self.text = text
        self.calls = []

    async def transcribe(self, audio):
        self.calls.append(audio)
        return self.text


# ----------------------------------------------------------- helpers
def _settings(**overrides) -> VoiceSettings:
    base = dict(
        sample_rate=8000,
        block_duration=0.5,
        silence_seconds=0.6,
        min_utterance_seconds=0.5,
        max_utterance_seconds=15.0,
        energy_threshold=0.02,
    )
    base.update(overrides)
    return VoiceSettings(**base)


def _speech(n: int = 4000):
    return np.ones(n, dtype="float32")  # rms == 1.0  (loud)


def _silence(n: int = 4000):
    return np.zeros(n, dtype="float32")  # rms == 0.0  (silent)


# ----------------------------------------------------------- rms unit
def test_rms_of_silence_and_full_scale():
    assert _rms(_silence()) == 0.0
    assert _rms(_speech()) == pytest.approx(1.0)


# ----------------------------------------------------------- happy path
async def test_emits_one_utterance_after_trailing_silence():
    # 4 loud blocks (2.0s) + 2 silent blocks (1.0s >= 0.6s silence) -> finalise.
    blocks = [_speech()] * 4 + [_silence()] * 2
    tx = FakeTranscriber("namaste")
    listener = VoiceListener(
        _settings(), transcriber=tx, capture=FakeCapture(blocks)
    )

    texts = [t async for t in listener.utterances()]

    assert texts == ["namaste"]
    assert len(tx.calls) == 1
    # buffer at finalise = 4 speech + 2 silence = 6 blocks
    assert tx.calls[0].size == 6 * 4000


# ----------------------------------------------------------- discard short
async def test_short_utterance_is_discarded():
    # 1 loud block (0.5s) + silence, but min is 2.0s -> discarded, nothing emitted.
    blocks = [_speech()] + [_silence()] * 2
    tx = FakeTranscriber("should not appear")
    listener = VoiceListener(
        _settings(min_utterance_seconds=2.0),
        transcriber=tx,
        capture=FakeCapture(blocks),
    )

    texts = [t async for t in listener.utterances()]

    assert texts == []
    assert tx.calls == []  # never transcribed


# ----------------------------------------------------------- max-length cap
async def test_max_utterance_seconds_forces_finalise():
    # Continuous loud speech with no silence; cap at 1.0s forces a finalise.
    blocks = [_speech()] * 10  # 5.0s of speech, cap = 1.0s
    tx = FakeTranscriber("forced")
    listener = VoiceListener(
        _settings(max_utterance_seconds=1.0),
        transcriber=tx,
        capture=FakeCapture(blocks),
    )

    texts = [t async for t in listener.utterances()]

    # Each finalised chunk holds exactly max/block_duration = 2 blocks.
    assert len(texts) == 5
    assert texts == ["forced"] * 5
    assert all(call.size == 2 * 4000 for call in tx.calls)


# ----------------------------------------------------------- idle silence
async def test_leading_silence_is_ignored():
    blocks = [_silence()] * 5 + [_speech()] * 2 + [_silence()] * 2
    tx = FakeTranscriber("ok")
    listener = VoiceListener(
        _settings(), transcriber=tx, capture=FakeCapture(blocks)
    )

    texts = [t async for t in listener.utterances()]

    assert texts == ["ok"]
    assert len(tx.calls) == 1
    # 2 speech + 2 trailing silence
    assert tx.calls[0].size == 4 * 4000


# ----------------------------------------------------------- multi-utterance
async def test_multiple_utterances_in_one_stream():
    utterance = [_speech()] * 2 + [_silence()] * 2
    blocks = utterance + utterance + utterance
    tx = FakeTranscriber("word")
    listener = VoiceListener(
        _settings(), transcriber=tx, capture=FakeCapture(blocks)
    )

    texts = [t async for t in listener.utterances()]

    assert texts == ["word", "word", "word"]
    assert len(tx.calls) == 3
