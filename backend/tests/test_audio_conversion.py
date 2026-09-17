"""Tests for backend-owned canonical audio conversion."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from app.services.audio_conversion import AudioConversionError, decode_and_canonicalize


def wav_bytes(samples: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """Encodes PCM16 samples as one independently decodable WAV container."""
    data = np.asarray(samples, dtype=np.int16)
    if channels == 1:
        data = data.reshape(-1, 1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(data.tobytes())
    return buffer.getvalue()


def test_mono_16khz_is_canonicalized_to_float32() -> None:
    """Canonical mono audio keeps its sample count and finite values."""
    source = np.array([0, 1000, -1000, 0], dtype=np.int16)
    result = decode_and_canonicalize(wav_bytes(source, 16_000, 1), "audio/wav")
    assert result.sample_rate == 16_000
    assert result.samples.dtype == np.float32
    assert result.samples.shape == (4,)


def test_stereo_is_downmixed_and_8khz_is_resampled() -> None:
    """Backend conversion produces mono 16 kHz from a stereo source."""
    stereo = np.column_stack([np.full(8, 1000, dtype=np.int16), np.full(8, -900, dtype=np.int16)])
    result = decode_and_canonicalize(wav_bytes(stereo, 8_000, 2), "audio/wav")
    assert result.sample_rate == 16_000
    assert result.samples.shape == (16,)
    assert np.all(result.samples > 0)


@pytest.mark.parametrize("container, content_type", [(b"not audio", "audio/wav"), (b"", "audio/flac"), (b"audio", "application/octet-stream")])
def test_invalid_or_unsupported_audio_is_rejected(container: bytes, content_type: str) -> None:
    """Only supported complete, decodable containers are accepted."""
    with pytest.raises(AudioConversionError):
        decode_and_canonicalize(container, content_type)


def test_silence_is_rejected() -> None:
    """Silent input cannot provide useful spoof evidence."""
    with pytest.raises(AudioConversionError, match="silent"):
        decode_and_canonicalize(wav_bytes(np.zeros(16_000, dtype=np.int16), 16_000, 1), "audio/wav")
