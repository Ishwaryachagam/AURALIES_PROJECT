"""
VAD tests. Run with: pytest tests/test_vad.py
"""

import numpy as np
import pytest
from backend.models.model_manager import model_manager
from backend.services.vad_service import VADSession


@pytest.fixture(scope="module", autouse=True)
def load_models():
    model_manager.load_all()


def test_silence_does_not_trigger_speech():
    vad = VADSession()
    silence = np.zeros(480, dtype=np.float32)  # 30ms at 16kHz
    for _ in range(20):  # ~600ms of silence
        event = vad.process_chunk(silence, chunk_duration_ms=30)
    assert vad.is_speaking is False


def test_synthetic_tone_as_speech_proxy():
    """
    Silero VAD is trained on real speech, so a pure sine tone won't
    reliably trigger it - this just confirms process_chunk runs without
    crashing on non-trivial audio. For a real speech-detection test, feed
    a real recorded speech .wav (see test_stt.py's fixture pattern).
    """
    vad = VADSession()
    t = np.linspace(0, 0.03, 480, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    event = vad.process_chunk(tone, chunk_duration_ms=30)
    assert event in ("silent", "speaking", "speech_start")
