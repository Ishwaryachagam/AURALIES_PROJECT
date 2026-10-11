"""
Latency tracking utility.

Records real timestamps at each pipeline stage and computes the metrics
the project document requires: STT latency, TTFT, LLM latency, TTS latency,
first-audio latency, and total turn latency.

Only values that were actually measured are ever reported - nothing here
is estimated or hard-coded.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LatencyTracker:
    speech_end: Optional[float] = None
    stt_start: Optional[float] = None
    stt_end: Optional[float] = None
    llm_start: Optional[float] = None
    first_token: Optional[float] = None
    llm_end: Optional[float] = None
    tts_start: Optional[float] = None
    first_audio_chunk: Optional[float] = None
    tts_end: Optional[float] = None

    def mark(self, field_name: str):
        setattr(self, field_name, time.time())

    def _ms(self, start, end) -> Optional[float]:
        if start is None or end is None:
            return None
        return round((end - start) * 1000, 1)

    def summary(self) -> dict:
        """Return only the metrics that were actually recorded."""
        result = {}

        stt_latency = self._ms(self.stt_start, self.stt_end)
        if stt_latency is not None:
            result["stt_ms"] = stt_latency

        ttft = self._ms(self.llm_start, self.first_token)
        if ttft is not None:
            result["ttft_ms"] = ttft

        llm_latency = self._ms(self.llm_start, self.llm_end)
        if llm_latency is not None:
            result["llm_ms"] = llm_latency

        tts_latency = self._ms(self.tts_start, self.tts_end)
        if tts_latency is not None:
            result["tts_ms"] = tts_latency

        first_audio_latency = self._ms(self.speech_end, self.first_audio_chunk)
        if first_audio_latency is not None:
            result["first_audio_ms"] = first_audio_latency

        total_latency = self._ms(self.speech_end, self.tts_end)
        if total_latency is not None:
            result["total_ms"] = total_latency

        return result
