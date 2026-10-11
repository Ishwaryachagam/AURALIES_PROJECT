"""
Core orchestration: ties VAD -> STT -> Emotion -> LLM -> TTS together per
conversation turn, streams AI audio back over WebRTC, and implements
interruption handling (Week 4 requirement) - if the user starts speaking
while the AI is talking, the current TTS/response task is cancelled and
the AI stops immediately.
"""

import asyncio
import fractions
import json
import logging
import time

import av
import numpy as np
from aiortc import MediaStreamTrack

from backend.config import settings
from backend.services.audio_service import IncomingAudioReader
from backend.services.vad_service import VADSession
from backend.services import stt_service, emotion_service, llm_service, tts_service
from backend.utils.latency import LatencyTracker

logger = logging.getLogger("auralis.streaming")


class AIVoiceTrack(MediaStreamTrack):
    """Outgoing audio track: pulls PCM chunks pushed by the pipeline and streams them to the browser."""

    kind = "audio"

    def __init__(self, sample_rate: int = 24000):
        super().__init__()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.sample_rate = sample_rate
        self._timestamp = 0

    def push_chunk(self, pcm_int16: np.ndarray, sample_rate: int):
        self.sample_rate = sample_rate
        self.queue.put_nowait(pcm_int16)

    def clear_queue(self):
        """Used on interruption - drop any AI audio that hasn't been sent yet."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def recv(self):
        try:
            pcm_chunk = await asyncio.wait_for(self.queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            # Keep the stream alive with a short silence frame when idle
            pcm_chunk = np.zeros(int(self.sample_rate * 0.02), dtype=np.int16)

        frame = av.AudioFrame.from_ndarray(
            pcm_chunk.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp += pcm_chunk.shape[0]
        return frame


class ConversationSession:
    """One instance per connected browser tab / WebRTC connection."""

    def __init__(self, incoming_track: MediaStreamTrack, ai_track: AIVoiceTrack, data_channel):
        self.reader = IncomingAudioReader(incoming_track)
        self.ai_track = ai_track
        self.data_channel = data_channel
        self.vad = VADSession()
        self.history = llm_service.ConversationHistory()

        self._speech_buffer: list[np.ndarray] = []
        self._ai_speaking_task: asyncio.Task | None = None
        self._main_task: asyncio.Task | None = None

    def start(self):
        self._main_task = asyncio.ensure_future(self._run())

    def stop(self):
        self.reader.stop()
        if self._main_task:
            self._main_task.cancel()
        self._interrupt_ai_if_speaking()

    def _send_event(self, payload: dict):
        if self.data_channel and self.data_channel.readyState == "open":
            try:
                self.data_channel.send(json.dumps(payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send data channel message: %s", exc)

    def _interrupt_ai_if_speaking(self):
        """Week 4: cancel in-flight TTS/LLM generation and clear queued audio."""
        if self._ai_speaking_task and not self._ai_speaking_task.done():
            self._ai_speaking_task.cancel()
            self.ai_track.clear_queue()
            self._send_event({"type": "interrupted"})
            logger.info("AI speech interrupted by user.")

    async def _run(self):
        self._send_event({"type": "status", "status": "listening"})
        try:
            async for float32_chunk, chunk_ms in self.reader.read_chunks():
                event = self.vad.process_chunk(float32_chunk, chunk_ms)

                if event == "speech_start":
                    self._interrupt_ai_if_speaking()  # Week 4 interruption trigger
                    self._speech_buffer = [float32_chunk]
                    self._send_event({"type": "speech_start"})

                elif event == "speaking":
                    self._speech_buffer.append(float32_chunk)

                elif event == "speech_end":
                    self._speech_buffer.append(float32_chunk)
                    full_audio = np.concatenate(self._speech_buffer) if self._speech_buffer else np.zeros(0, dtype=np.float32)
                    self._speech_buffer = []
                    self._send_event({"type": "speech_end"})
                    self._ai_speaking_task = asyncio.ensure_future(self._handle_turn(full_audio))

        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("Session error: %s", exc)
            self._send_event({"type": "error", "message": str(exc)})

    async def _handle_turn(self, user_audio: np.ndarray):
        """Runs STT -> Emotion -> LLM -> TTS for one user utterance."""
        latency = LatencyTracker()
        latency.mark("speech_end")

        try:
            # --- STT ---
            latency.mark("stt_start")
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(None, stt_service.transcribe, user_audio)
            latency.mark("stt_end")

            if not transcript:
                self._send_event({"type": "transcript", "text": "", "note": "empty_or_unclear"})
                return

            self._send_event({"type": "transcript", "text": transcript, **latency.summary()})

            # --- Emotion ---
            emotion_result = await loop.run_in_executor(
                None, emotion_service.detect_emotion, user_audio
            )
            self._send_event({"type": "emotion", **emotion_result})

            # --- LLM (streamed, TTFT measured) ---
            prompt = llm_service.build_prompt(
                transcript, emotion_result["emotion"], emotion_result["confidence"], self.history
            )
            latency.mark("llm_start")
            full_response = ""

            def on_first_token():
                latency.mark("first_token")

            for chunk in llm_service.stream_llm_response(prompt, on_first_token=on_first_token):
                full_response += chunk
            latency.mark("llm_end")

            self.history.add(transcript, full_response)
            self._send_event(
                {"type": "llm_response", "text": full_response, **latency.summary()}
            )

            # --- TTS (chunked) ---
            latency.mark("tts_start")
            self._send_event({"type": "status", "status": "speaking"})

            first_chunk_sent = False
            for pcm_chunk, sample_rate in await loop.run_in_executor(
                None, lambda: list(tts_service.synthesize_chunks(full_response, emotion_result["emotion"]))
            ):
                self.ai_track.push_chunk(pcm_chunk, sample_rate)
                if not first_chunk_sent:
                    latency.mark("first_audio_chunk")
                    first_chunk_sent = True
            latency.mark("tts_end")

            self._send_event({"type": "latency", **latency.summary()})
            self._send_event({"type": "status", "status": "listening"})

        except asyncio.CancelledError:
            logger.info("Turn cancelled (interruption).")
            self._send_event({"type": "status", "status": "listening"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Turn handling error: %s", exc)
            self._send_event({"type": "error", "message": str(exc)})
