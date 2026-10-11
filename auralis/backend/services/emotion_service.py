"""
Emotion classification via a pretrained Wav2Vec2 speech-emotion-recognition model.

Model in use (see config.py / .env): ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
Trained on RAVDESS. Its label set (used AS-IS, not invented) is:
    angry, calm, disgust, fearful, happy, neutral, sad, surprised

If you swap EMOTION_MODEL for a different checkpoint, the label set will change
accordingly - always report whatever the model itself returns.
"""

import numpy as np
from backend.models.model_manager import model_manager


def detect_emotion(float32_audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Returns {"emotion": <label>, "confidence": <float 0-1>} using the top
    prediction from the pretrained classifier, or a neutral fallback if
    the audio is too short to classify.
    """
    if float32_audio.size < sample_rate * 0.3:  # less than ~300ms, too short to be reliable
        return {"emotion": "unknown", "confidence": 0.0}

    predictions = model_manager.emotion_pipeline(
        {"array": float32_audio, "sampling_rate": sample_rate}
    )
    # transformers audio-classification pipeline returns a list sorted by score desc
    top = predictions[0]
    return {"emotion": top["label"], "confidence": round(float(top["score"]), 3)}
