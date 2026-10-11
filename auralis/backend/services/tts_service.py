"""
Text-to-speech service supporting XTTSv2 (default) or Bark, selected via
TTS_ENGINE in .env.

Honesty note on "streaming": Coqui's high-level TTS().tts() API and Bark's
generate_audio() both synthesize the *entire* utterance before returning -
they do not generate audio incrementally token-by-token. What we do here is
generate the full waveform once, then split it into small chunks that are
fed to the outgoing WebRTC track one at a time, so playback of the first
chunk can start as soon as it's ready rather than waiting on the whole
response being queued into the peer connection. This matches the project
document's "chunked audio generation/streaming" requirement, but it does
NOT reduce raw TTS compute latency - that would require XTTS's lower-level
`inference_stream` API, which is a reasonable future upgrade (see README).

Emotion conditioning: XTTS only supports zero-shot voice cloning via a
reference speaker .wav - it has no explicit "make this angry" parameter.
We condition emotion the practical way the model actually supports: by
letting the emotion-aware LLM response text itself carry tone (e.g. calmer
phrasing for a fearful user), and by selecting a slightly different
reference speaker clip per emotion bucket if you provide one (optional -
falls back to a single default speaker otherwise).
"""

import os
import numpy as np

from backend.config import settings
from backend.models.model_manager import model_manager
from backend.utils.audio_utils import float32_to_pcm16

CHUNK_SAMPLES = 4800  # ~200ms at 24kHz


def _resolve_speaker_wav(emotion: str) -> str:
    """Optionally use an emotion-specific reference clip if present, else the default."""
    emotion_specific = f"models/speaker_{emotion}.wav"
    if os.path.exists(emotion_specific):
        return emotion_specific
    return settings.TTS_SPEAKER_WAV


def synthesize(text: str, emotion: str = "neutral") -> tuple[np.ndarray, int]:
    """Generate the full utterance. Returns (float32 waveform, sample_rate)."""
    if not text.strip():
        return np.zeros(0, dtype=np.float32), settings.TARGET_SAMPLE_RATE

    if settings.TTS_ENGINE == "xtts":
        speaker_wav = _resolve_speaker_wav(emotion)
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(
                f"XTTS requires a speaker reference clip at '{speaker_wav}'. "
                f"Record a 5-10 second clean speech sample and place it there, "
                f"or set TTS_SPEAKER_WAV in .env."
            )
        wav_list = model_manager.tts_model.tts(
            text=text, speaker_wav=speaker_wav, language=settings.TTS_LANGUAGE
        )
        waveform = np.array(wav_list, dtype=np.float32)
        sample_rate = 24000  # XTTSv2's native output rate
        return waveform, sample_rate

    elif settings.TTS_ENGINE == "bark":
        from bark import generate_audio, SAMPLE_RATE as BARK_SAMPLE_RATE

        waveform = generate_audio(text)
        return waveform.astype(np.float32), BARK_SAMPLE_RATE

    else:
        raise ValueError(f"Unknown TTS_ENGINE: {settings.TTS_ENGINE}")


def synthesize_chunks(text: str, emotion: str = "neutral"):
    """Generator yielding (int16 pcm chunk, sample_rate) tuples for streaming playback."""
    waveform, sample_rate = synthesize(text, emotion)
    pcm = float32_to_pcm16(waveform)
    for start in range(0, pcm.shape[0], CHUNK_SAMPLES):
        yield pcm[start : start + CHUNK_SAMPLES], sample_rate
