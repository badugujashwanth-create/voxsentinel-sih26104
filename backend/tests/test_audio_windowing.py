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


def test_scheduler_propagates_causal_source_metadata() -> None:
    """Overlapping windows retain source ranges from their contributing chunks."""
    from app.services.audio_source_ledger import AudioSourceSegment

    scheduler = AASISTWindowScheduler(window_samples=4, hop_samples=2)
    assert scheduler.push_with_metadata(np.ones(3, dtype=np.float32), AudioSourceSegment(100, 102, 1, 1, 0)) == []
    windows = scheduler.push_with_metadata(np.ones(3, dtype=np.float32), AudioSourceSegment(103, 105, 2, 2, 0))
    assert len(windows) == 2
    samples, metadata = windows[0]
    assert samples.size == 4
    assert metadata.audio_source_frame_start == 100
    assert metadata.audio_source_frame_end == 105
    assert metadata.audio_source_transport_sequence_end == 2
