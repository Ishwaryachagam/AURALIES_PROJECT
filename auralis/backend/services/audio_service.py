"""
Incoming audio handling: wraps the aiortc MediaStreamTrack the browser
sends us, converts each frame to 16kHz mono float32, and hands fixed-size
chunks to the VAD.
"""

import asyncio
import numpy as np

from aiortc import MediaStreamTrack
from backend.config import settings
from backend.utils.audio_utils import av_frame_to_pcm16_mono_16k, pcm16_to_float32


class IncomingAudioReader:
    """
    Pulls frames from the browser's WebRTC audio track and yields fixed-size
    16kHz mono float32 chunks (default ~30ms) for the VAD/streaming pipeline.
    """

    def __init__(self, track: MediaStreamTrack, chunk_ms: int = 30):
        self.track = track
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(settings.TARGET_SAMPLE_RATE * chunk_ms / 1000)
        self._buffer = np.zeros(0, dtype=np.int16)
        self.stopped = False

    async def read_chunks(self):
        """Async generator: yields (float32_chunk, chunk_ms) until the track ends or is stopped."""
        while not self.stopped:
            try:
                frame = await self.track.recv()
            except Exception:
                break  # track ended (e.g. peer disconnected)

            pcm16 = av_frame_to_pcm16_mono_16k(frame, settings.TARGET_SAMPLE_RATE)
            self._buffer = np.concatenate([self._buffer, pcm16])

            while self._buffer.shape[0] >= self.chunk_samples:
                chunk = self._buffer[: self.chunk_samples]
                self._buffer = self._buffer[self.chunk_samples :]
                yield pcm16_to_float32(chunk), self.chunk_ms

    def stop(self):
        self.stopped = True
