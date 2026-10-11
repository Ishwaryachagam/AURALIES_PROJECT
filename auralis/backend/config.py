"""
Central configuration for Auralis backend.
All values are read from environment variables (.env), never hard-coded.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Network ---
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # CORS: comma-separated list of allowed origins in production
    ALLOWED_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", FRONTEND_URL).split(",")
        if origin.strip()
    ]

    # --- Ollama / LLM ---
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")
    LLM_MAX_HISTORY_TURNS: int = int(os.getenv("LLM_MAX_HISTORY_TURNS", "6"))
    LLM_MAX_RESPONSE_TOKENS: int = int(os.getenv("LLM_MAX_RESPONSE_TOKENS", "120"))

    # --- STT ---
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda"
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # --- Emotion model ---
    EMOTION_MODEL: str = os.getenv(
        "EMOTION_MODEL", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    EMOTION_DEVICE: str = os.getenv("EMOTION_DEVICE", "cpu")

    # --- TTS ---
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "xtts")  # "xtts" or "bark"
    TTS_MODEL: str = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    TTS_DEVICE: str = os.getenv("TTS_DEVICE", "cpu")
    TTS_SPEAKER_WAV: str = os.getenv("TTS_SPEAKER_WAV", "models/speaker_reference.wav")
    TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "en")

    # --- VAD ---
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    VAD_SILENCE_MS: int = int(os.getenv("VAD_SILENCE_MS", "500"))
    VAD_MIN_SPEECH_MS: int = int(os.getenv("VAD_MIN_SPEECH_MS", "200"))

    # --- Audio ---
    TARGET_SAMPLE_RATE: int = 16000  # required by Whisper, Silero VAD, Wav2Vec2


settings = Settings()
