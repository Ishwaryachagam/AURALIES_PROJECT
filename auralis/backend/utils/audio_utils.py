"""
Reusable audio conversion helpers.

The browser sends audio over WebRTC (decoded by aiortc/PyAV into raw PCM
frames at whatever sample rate/channel layout the browser negotiated,
typically 48000 Hz stereo or mono). Whisper, Silero VAD and the Wav2Vec2
emotion model all require 16 kHz mono float32/int16 PCM. These helpers
handle that conversion so no service has to duplicate it.
"""

import numpy as np


def stereo_to_mono(samples: np.ndarray) -> np.ndarray:
    """Average stereo channels down to mono. samples shape: (channels, n) or (n, channels)."""
    if samples.ndim == 1:
        return samples
    if samples.shape[0] <= 2:  # (channels, n)
        return samples.mean(axis=0)
    return samples.mean(axis=1)  # (n, channels)


def resample_int16(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resampler for int16 PCM. Good enough for speech ML input."""
    if orig_sr == target_sr or pcm.size == 0:
        return pcm.astype(np.int16)

    duration = pcm.shape[0] / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    orig_indices = np.linspace(0, pcm.shape[0] - 1, num=pcm.shape[0])
    target_indices = np.linspace(0, pcm.shape[0] - 1, num=target_len)
    resampled = np.interp(target_indices, orig_indices, pcm.astype(np.float32))
    return resampled.astype(np.int16)


def av_frame_to_pcm16_mono_16k(frame, target_sr: int = 16000) -> np.ndarray:
    """
    Convert a PyAV AudioFrame (as delivered by aiortc's incoming track) into
    a 1-D int16 numpy array at 16 kHz mono.
    """
    array = frame.to_ndarray()  # shape depends on format; usually (channels, samples) for planar
    if array.dtype != np.int16:
        # aiortc audio frames are typically already s16; guard just in case
        array = array.astype(np.int16)

    mono = stereo_to_mono(array)
    mono = mono.astype(np.int16)
    resampled = resample_int16(mono, frame.sample_rate, target_sr)
    return resampled


def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    """Int16 PCM -> float32 in range [-1, 1], as required by Whisper/Silero/Wav2Vec2."""
    return pcm.astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Float32 [-1, 1] audio (e.g. TTS output) -> int16 PCM for streaming back over WebRTC."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)
