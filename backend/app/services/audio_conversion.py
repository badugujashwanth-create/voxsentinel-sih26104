"""Canonical conversion for complete feeder audio containers."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

CANONICAL_SAMPLE_RATE = 16_000
SILENCE_PEAK_THRESHOLD = 1e-5
SUPPORTED_CONTENT_TYPES = frozenset({"audio/wav", "audio/flac"})


class AudioConversionError(ValueError):
    """Raised when a complete audio container cannot be canonicalized."""


@dataclass(frozen=True)
class CanonicalAudio:
    """Finite mono float32 audio at the backend canonical sample rate."""

    samples: np.ndarray
    sample_rate: int


def decode_and_canonicalize(container: bytes, content_type: str) -> CanonicalAudio:
    """Decodes one complete WAV/FLAC container into canonical PCM."""
    normalized_content_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    if normalized_content_type not in SUPPORTED_CONTENT_TYPES:
        raise AudioConversionError("content type must be audio/wav or audio/flac")
    if not container:
        raise AudioConversionError("audio container is empty")
    try:
        samples, sample_rate = sf.read(io.BytesIO(container), dtype="float32", always_2d=False)
    except (RuntimeError, ValueError, OSError) as error:
        raise AudioConversionError("audio container is not independently decodable") from error
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        raise AudioConversionError("audio contains no samples")
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise AudioConversionError("audio must be finite mono or stereo PCM")
    if sample_rate != CANONICAL_SAMPLE_RATE:
        audio = resample_poly(audio, CANONICAL_SAMPLE_RATE, sample_rate).astype(np.float32, copy=False)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise AudioConversionError("resampled audio is not finite")
    if float(np.max(np.abs(audio))) <= SILENCE_PEAK_THRESHOLD:
        raise AudioConversionError("audio is silent")
    return CanonicalAudio(samples=audio, sample_rate=CANONICAL_SAMPLE_RATE)
