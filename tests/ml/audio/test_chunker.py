"""Streaming audio chunker tests."""

from __future__ import annotations

import pytest

from ml.audio.chunker import AudioChunker

WINDOW_SAMPLES = 32_000
HOP_SAMPLES = 8_000


def ramp(count: int, start: int = 0) -> list[float]:
    """Builds a distinguishable ascending sample run."""
    return [float(value) for value in range(start, start + count)]


def test_defaults_match_the_agreed_window_and_hop() -> None:
    """16kHz with a 2.0s window and 0.5s hop resolves to 32000/8000 samples."""
    chunker = AudioChunker()
    assert chunker.sample_rate == 16_000
    assert chunker.window_samples == WINDOW_SAMPLES
    assert chunker.hop_samples == HOP_SAMPLES


def test_insufficient_audio_yields_no_window() -> None:
    """One sample short of a window emits nothing."""
    chunker = AudioChunker()
    assert chunker.push(ramp(WINDOW_SAMPLES - 1)) == []
    assert chunker.buffered_samples == WINDOW_SAMPLES - 1


def test_exact_window_yields_one_window() -> None:
    """Exactly 32000 samples emits the first window, samples 0..31999."""
    chunker = AudioChunker()
    windows = chunker.push(ramp(WINDOW_SAMPLES))
    assert len(windows) == 1
    assert list(windows[0]) == ramp(WINDOW_SAMPLES)


def test_one_hop_later_yields_a_second_overlapping_window() -> None:
    """A further 8000 samples emits samples 8000..39999."""
    chunker = AudioChunker()
    chunker.push(ramp(WINDOW_SAMPLES))
    windows = chunker.push(ramp(HOP_SAMPLES, start=WINDOW_SAMPLES))
    assert len(windows) == 1
    assert list(windows[0]) == ramp(WINDOW_SAMPLES, start=HOP_SAMPLES)


def test_third_window_continues_the_hop_pattern() -> None:
    """The third window covers samples 16000..47999."""
    chunker = AudioChunker()
    chunker.push(ramp(WINDOW_SAMPLES))
    chunker.push(ramp(HOP_SAMPLES, start=WINDOW_SAMPLES))
    windows = chunker.push(ramp(HOP_SAMPLES, start=WINDOW_SAMPLES + HOP_SAMPLES))
    assert len(windows) == 1
    assert list(windows[0]) == ramp(WINDOW_SAMPLES, start=2 * HOP_SAMPLES)


def test_windows_overlap_by_window_minus_hop() -> None:
    """Consecutive windows share 24000 samples."""
    chunker = AudioChunker()
    first = chunker.push(ramp(WINDOW_SAMPLES))[0]
    second = chunker.push(ramp(HOP_SAMPLES, start=WINDOW_SAMPLES))[0]
    assert list(first[HOP_SAMPLES:]) == list(second[: WINDOW_SAMPLES - HOP_SAMPLES])


def test_sample_ordering_is_preserved_across_many_small_pushes() -> None:
    """Fragmented input rebuilds the original ordering exactly."""
    chunker = AudioChunker()
    samples = ramp(WINDOW_SAMPLES)
    windows: list = []
    for offset in range(0, WINDOW_SAMPLES, 1_000):
        windows.extend(chunker.push(samples[offset : offset + 1_000]))
    assert len(windows) == 1
    assert list(windows[0]) == samples


def test_a_single_large_push_emits_every_completed_window() -> None:
    """One oversized push drains all windows it completes."""
    chunker = AudioChunker()
    windows = chunker.push(ramp(WINDOW_SAMPLES + 2 * HOP_SAMPLES))
    assert len(windows) == 3
    assert list(windows[2]) == ramp(WINDOW_SAMPLES, start=2 * HOP_SAMPLES)


def test_reset_clears_buffered_state() -> None:
    """Reset discards buffered audio so windowing restarts cleanly."""
    chunker = AudioChunker()
    chunker.push(ramp(WINDOW_SAMPLES - 1))
    chunker.reset()
    assert chunker.buffered_samples == 0
    assert chunker.push(ramp(HOP_SAMPLES)) == []
    assert chunker.push(ramp(WINDOW_SAMPLES - HOP_SAMPLES, start=HOP_SAMPLES)) != []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": -16_000},
        {"window_seconds": 0},
        {"window_seconds": -1.0},
        {"hop_seconds": 0},
        {"hop_seconds": -0.5},
        {"window_seconds": 0.5, "hop_seconds": 2.0},
        {"sample_rate": 1, "window_seconds": 0.1},
    ],
)
def test_invalid_settings_fail_clearly(kwargs: dict) -> None:
    """Bad configuration raises ValueError rather than misbehaving silently."""
    with pytest.raises(ValueError):
        AudioChunker(**kwargs)
