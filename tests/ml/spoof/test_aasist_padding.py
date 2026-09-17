"""AASIST windowing tests.

These exercise the fixed-length windowing the released checkpoint requires.
They import the adapter module but never load the checkpoint.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="ML runtime not installed in this environment")
pytest.importorskip("torch", reason="ML runtime not installed in this environment")
pytest.importorskip("scipy", reason="ML runtime not installed in this environment")

import numpy as np  # noqa: E402
from ml.spoof.aasist import AASIST_INPUT_SAMPLES, AASIST_SAMPLE_RATE, pad_to_model_length  # noqa: E402

def ramp(count: int) -> np.ndarray:
    """Builds a distinguishable ascending sample run."""
    return np.arange(count, dtype=np.float32)


def test_exact_length_audio_is_untouched() -> None:
    """Audio already at the model length passes through without warning."""
    samples = ramp(AASIST_INPUT_SAMPLES)
    window, warnings = pad_to_model_length(samples)
    assert warnings == []
    assert np.array_equal(window, samples)


def test_longer_audio_is_truncated_and_reported() -> None:
    """Extra audio is dropped, and the caller is told."""
    samples = ramp(AASIST_INPUT_SAMPLES * 2)
    window, warnings = pad_to_model_length(samples)
    assert len(window) == AASIST_INPUT_SAMPLES
    assert np.array_equal(window, samples[:AASIST_INPUT_SAMPLES])
    assert len(warnings) == 1
    assert "truncated" in warnings[0]


def test_shorter_audio_is_tiled_and_reported() -> None:
    """Short audio is repeated to fill the window, with a warning."""
    samples = ramp(AASIST_SAMPLE_RATE)  # 1 second
    window, warnings = pad_to_model_length(samples)
    assert len(window) == AASIST_INPUT_SAMPLES
    assert np.array_equal(window[:AASIST_SAMPLE_RATE], samples)
    assert np.array_equal(window[AASIST_SAMPLE_RATE : 2 * AASIST_SAMPLE_RATE], samples)
    assert len(warnings) == 1
    assert "tiled" in warnings[0]


def test_tiling_warning_names_the_weakened_result() -> None:
    """The warning says the score is weaker, not just that padding happened."""
    _, warnings = pad_to_model_length(ramp(AASIST_SAMPLE_RATE))
    assert "weakens" in warnings[0]


def test_window_length_matches_the_documented_duration() -> None:
    """64600 samples at 16kHz is the 4.04s window the checkpoint expects."""
    assert AASIST_INPUT_SAMPLES / AASIST_SAMPLE_RATE == pytest.approx(4.0375)