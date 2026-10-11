"""
Emotion detection tests. Run with: pytest tests/test_emotion.py
"""

import numpy as np
import pytest
from backend.models.model_manager import model_manager
from backend.services import emotion_service


@pytest.fixture(scope="module", autouse=True)
def load_models():
    model_manager.load_all()


def test_too_short_audio_returns_unknown():
    tiny = np.zeros(100, dtype=np.float32)
    result = emotion_service.detect_emotion(tiny)
    assert result["emotion"] == "unknown"
    assert result["confidence"] == 0.0


def test_confidence_is_between_0_and_1():
    noise = (np.random.rand(16000) * 2 - 1).astype(np.float32) * 0.1
    result = emotion_service.detect_emotion(noise)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["emotion"], str)


def test_multiple_samples_from_fixtures():
    """
    Manual test: place labeled clips at tests/fixtures/emotion_<label>.wav
    (e.g. emotion_angry.wav, emotion_happy.wav) to sanity-check predictions
    against known emotional recordings.
    """
    import soundfile as sf
    import os
    import glob

    fixtures = glob.glob("tests/fixtures/emotion_*.wav")
    if not fixtures:
        pytest.skip("No emotion fixture files found")
    for path in fixtures:
        audio, sr = sf.read(path)
        result = emotion_service.detect_emotion(audio.astype(np.float32), sample_rate=sr)
        print(os.path.basename(path), "->", result)
