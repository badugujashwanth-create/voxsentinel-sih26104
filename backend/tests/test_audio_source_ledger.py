"""Tests for causal source-range correlation across canonical windows."""

from __future__ import annotations

import pytest

from app.services.audio_source_ledger import AudioSourceSegment, SourceSegmentLedger


def test_window_metadata_unions_contributing_source_segments() -> None:
    """A window reports the source range that made its canonical samples available."""
    ledger = SourceSegmentLedger()
    ledger.append(0, 4, AudioSourceSegment(100, 109, 1, 1, 0))
    ledger.append(4, 4, AudioSourceSegment(110, 119, 2, 2, 0))
    metadata = ledger.metadata_for_window(2, 8, 7)
    assert metadata.audio_source_frame_start == 100
    assert metadata.audio_source_frame_end == 119
    assert metadata.audio_source_transport_sequence_start == 1
    assert metadata.audio_source_transport_sequence_end == 2
    assert metadata.audio_source_gap_count == 0
    assert metadata.audio_window_sequence == 7


def test_window_metadata_preserves_source_gaps() -> None:
    """Forward transport/source gaps remain telemetry on the resulting window."""
    ledger = SourceSegmentLedger()
    ledger.append(0, 2, AudioSourceSegment(0, 1, 1, 1, 0))
    ledger.append(2, 2, AudioSourceSegment(5, 6, 3, 3, 3))
    metadata = ledger.metadata_for_window(0, 4, 1)
    assert metadata.audio_source_frame_start == 0
    assert metadata.audio_source_frame_end == 6
    assert metadata.audio_source_transport_sequence_start == 1
    assert metadata.audio_source_transport_sequence_end == 3
    assert metadata.audio_source_gap_count == 3


def test_ledger_rejects_non_monotonic_canonical_ranges() -> None:
    """Canonical output ranges cannot overlap in the correlation ledger."""
    ledger = SourceSegmentLedger()
    ledger.append(0, 4, AudioSourceSegment(0, 3, 1, 1, 0))
    with pytest.raises(ValueError, match="overlap"):
        ledger.append(3, 2, AudioSourceSegment(4, 5, 2, 2, 0))

