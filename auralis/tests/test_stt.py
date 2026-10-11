"""
STT tests. Run with: pytest tests/test_stt.py
Requires models to be downloadable (internet access) on first run.
"""

import numpy as np
import pytest
from backend.models.model_manager import model_manager
from backend.services import stt_service


@pytest.fixture(scope="module", autouse=True)
def load_models():
    model_manager.load_all()


def test_empty_audio_returns_empty_string():
    empty = np.zeros(0, dtype=np.float32)
    assert stt_service.transcribe(empty) == ""


def test_silence_does_not_crash():
    silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    result = stt_service.transcribe(silence)
    assert isinstance(result, str)


def test_short_speech_from_file():
    """
    Manual test: place a short WAV clip (16kHz mono) at tests/fixtures/short_speech.wav
    saying e.g. "Hello, this is a test" and run this test to confirm transcription.
    """
    import soundfile as sf
    import os

    path = "tests/fixtures/short_speech.wav"
    if not os.path.exists(path):
        pytest.skip("No fixture file - add tests/fixtures/short_speech.wav to run this test")
    audio, sr = sf.read(path)
    assert sr == 16000
    text = stt_service.transcribe(audio.astype(np.float32))
    assert len(text) > 0
    print("Transcribed:", text)
