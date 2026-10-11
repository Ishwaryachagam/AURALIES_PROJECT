"""
Loads every ML model exactly once at application startup and holds them
as singletons. Services pull models from here instead of loading their
own copies per-request.

GPU vs CPU:
- Set WHISPER_DEVICE / EMOTION_DEVICE / TTS_DEVICE to "cuda" in .env if you
  have an NVIDIA GPU with a working CUDA + matching torch build installed.
- Everything defaults to "cpu", which works everywhere but is significantly
  slower, especially XTTS. This is expected - see README "Limitations".
"""

import logging
import torch
from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline

from backend.config import settings

logger = logging.getLogger("auralis.models")


class ModelManager:
    def __init__(self):
        self.whisper_model: WhisperModel | None = None
        self.vad_model = None
        self.vad_utils = None
        self.emotion_pipeline = None
        self.tts_model = None
        self.loaded = False

    def load_all(self):
        if self.loaded:
            return

        logger.info("Loading Faster-Whisper (%s, %s)...", settings.WHISPER_MODEL, settings.WHISPER_DEVICE)
        self.whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper loaded.")

        logger.info("Loading Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        logger.info("Silero VAD loaded.")

        logger.info("Loading Wav2Vec2 emotion model (%s)...", settings.EMOTION_MODEL)
        device_index = 0 if settings.EMOTION_DEVICE == "cuda" else -1
        self.emotion_pipeline = hf_pipeline(
            "audio-classification",
            model=settings.EMOTION_MODEL,
            device=device_index,
        )
        logger.info("Emotion model loaded.")

        logger.info("Loading TTS engine (%s)...", settings.TTS_ENGINE)
        if settings.TTS_ENGINE == "xtts":
            from TTS.api import TTS

            self.tts_model = TTS(settings.TTS_MODEL)
            if settings.TTS_DEVICE == "cuda":
                self.tts_model.to("cuda")
        elif settings.TTS_ENGINE == "bark":
            from bark import preload_models

            preload_models()
            self.tts_model = "bark"  # bark's API is functional, not object-based
        else:
            raise ValueError(f"Unknown TTS_ENGINE: {settings.TTS_ENGINE}")
        logger.info("TTS engine loaded.")

        self.loaded = True
        logger.info("All models loaded. Auralis is ready.")


model_manager = ModelManager()
