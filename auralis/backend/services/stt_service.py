"""
Speech-to-text via Faster-Whisper (loaded once in model_manager).
"""

import numpy as np
from backend.models.model_manager import model_manager


def transcribe(float32_audio: np.ndarray, language: str = "en") -> str:
    """Transcribe a float32 mono 16kHz numpy array. Returns "" for empty/unclear audio."""
    if float32_audio.size == 0:
        return ""

    segments, _info = model_manager.whisper_model.transcribe(
        float32_audio, language=language, beam_size=1, vad_filter=False
    )
    text = "".join(seg.text for seg in segments).strip()
    return text
