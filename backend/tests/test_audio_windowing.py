"""Tests for exact AASIST window scheduling."""

from __future__ import annotations

import numpy as np

from app.services.audio_windowing import AASISTWindowScheduler, AASIST_WINDOW_SAMPLES, AASIST_WINDOW_HOP_SAMPLES


def test_scheduler_waits_for_a_complete_model_window() -> None:
    """Fewer than 64,600 samples produce no model window."""
    scheduler = AASISTWindowScheduler()
    assert scheduler.push(np.ones(AASIST_WINDOW_SAMPLES - 1, dtype=np.float32)) == []
    windows = scheduler.push(np.ones(1, dtype=np.float32))
    assert len(windows) == 1
    assert windows[0].shape == (AASIST_WINDOW_SAMPLES,)


def test_scheduler_emits_overlapping_windows_at_the_configured_hop() -> None:
    """Adding one hop after the first window emits the next overlap."""
    scheduler = AASISTWindowScheduler()
    first = scheduler.push(np.arange(AASIST_WINDOW_SAMPLES, dtype=np.float32))
    second = scheduler.push(np.arange(AASIST_WINDOW_HOP_SAMPLES, dtype=np.float32))
    assert len(first) == 1
    assert len(second) == 1
    assert second[0][0] == AASIST_WINDOW_HOP_SAMPLES
