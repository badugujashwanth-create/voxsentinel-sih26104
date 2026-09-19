"""Tests for stateful source-to-canonical audio conversion."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.streaming_audio_canonicalizer import StreamingAudioCanonicalizer


def _source(rate: int, channels: int = 1) -> np.ndarray:
    """Creates deterministic finite interleaved PCM for conversion tests."""
    time = np.arange(rate + 137, dtype=np.float32) / rate
    mono = (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    return np.column_stack((mono, -mono)).reshape(-1) if channels == 2 else mono


@pytest.mark.parametrize("rate", [48_000, 44_100])
def test_chunked_streaming_matches_continuous_stream(rate: int) -> None:
    """Arbitrary transport boundaries do not reset resampler phase."""
    source = _source(rate)
    continuous = StreamingAudioCanonicalizer(rate, 1)
    continuous_output = np.concatenate((continuous.push(source), continuous.flush()))

    chunked = StreamingAudioCanonicalizer(rate, 1)
    outputs: list[np.ndarray] = []
    start = 0
    for size in (1, 37, 1024, 7, 333, 819, 2_001):
        end = min(source.size, start + size)
        if end <= start:
            break
        outputs.append(chunked.push(source[start:end]))
        start = end
    if start < source.size:
        outputs.append(chunked.push(source[start:]))
    outputs.append(chunked.flush())
    chunked_output = np.concatenate(outputs)

    assert chunked_output.dtype == np.float32
    assert np.isfinite(chunked_output).all()
    np.testing.assert_allclose(chunked_output, continuous_output, rtol=0, atol=1e-6)


def test_stereo_is_downmixed_before_resampling() -> None:
    """Stereo input becomes one finite mono canonical stream."""
    canonicalizer = StreamingAudioCanonicalizer(48_000, 2)
    output = np.concatenate((canonicalizer.push(_source(48_000, 2)), canonicalizer.flush()))
    assert output.ndim == 1
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_flush_does_not_pad_or_accept_after_close() -> None:
    """Flush only drains received audio and closed sessions reject input."""
    canonicalizer = StreamingAudioCanonicalizer(48_000, 1)
    output = canonicalizer.push(np.zeros(31, dtype=np.float32))
    flushed = canonicalizer.flush()
    assert output.size + flushed.size <= 31
    canonicalizer.close()
    with pytest.raises(RuntimeError, match="closed"):
        canonicalizer.push(np.zeros(1, dtype=np.float32))

