"""
Silero VAD wrapper.

Silero VAD expects 16kHz mono float32 audio in fixed-size windows
(512 samples for 16kHz). We feed it chunk-by-chunk and maintain a small
state machine that reports speech_start / speech_end events, matching the
project document's "detect when user starts/stops speaking" requirement.
"""

import numpy as np

from backend.config import settings
from backend.models.model_manager import model_manager

SILERO_WINDOW_SIZE = 512  # required window size for 16kHz audio


class VADSession:
    """One instance per active connection - VAD has internal state (silence counters)."""

    def __init__(self):
        self.is_speaking = False
        self.silence_ms = 0
        self.speech_ms = 0
        self._frame_buffer = np.zeros(0, dtype=np.float32)

    def _get_speech_prob(self, window: np.ndarray) -> float:
        import torch

        tensor = torch.from_numpy(window)
        with torch.no_grad():
            prob = model_manager.vad_model(tensor, settings.TARGET_SAMPLE_RATE).item()
        return prob

    def process_chunk(self, float32_audio: np.ndarray, chunk_duration_ms: float):
        """
        Feed a chunk of float32 audio (16kHz mono). Returns one of:
        "speech_start", "speech_end", "speaking", "silent" (no state change).
        """
        self._frame_buffer = np.concatenate([self._frame_buffer, float32_audio])

        speech_prob = 0.0
        windows_processed = 0
        while self._frame_buffer.shape[0] >= SILERO_WINDOW_SIZE:
            window = self._frame_buffer[:SILERO_WINDOW_SIZE]
            self._frame_buffer = self._frame_buffer[SILERO_WINDOW_SIZE:]
            speech_prob = max(speech_prob, self._get_speech_prob(window))
            windows_processed += 1

        if windows_processed == 0:
            return "silent" if not self.is_speaking else "speaking"

        is_speech_now = speech_prob >= settings.VAD_THRESHOLD

        if is_speech_now:
            self.speech_ms += chunk_duration_ms
            self.silence_ms = 0
            if not self.is_speaking and self.speech_ms >= settings.VAD_MIN_SPEECH_MS:
                self.is_speaking = True
                return "speech_start"
            return "speaking" if self.is_speaking else "silent"
        else:
            if self.is_speaking:
                self.silence_ms += chunk_duration_ms
                if self.silence_ms >= settings.VAD_SILENCE_MS:
                    self.is_speaking = False
                    self.speech_ms = 0
                    self.silence_ms = 0
                    return "speech_end"
                return "speaking"
            self.speech_ms = 0
            return "silent"
