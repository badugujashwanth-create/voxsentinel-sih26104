"""Bounded causal mapping from canonical samples to accepted source ranges."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioSourceSegment:
    """Conservative source availability range for a canonical output segment."""

    source_frame_start: int
    source_frame_end: int
    transport_sequence_start: int
    transport_sequence_end: int
    source_gap_count: int


@dataclass(frozen=True)
class AudioWindowMetadata:
    """Correlation metadata attached to one AASIST window."""

    audio_source_frame_start: int
    audio_source_frame_end: int
    audio_source_transport_sequence_start: int
    audio_source_transport_sequence_end: int
    audio_source_gap_count: int
    audio_window_sequence: int
    canonical_start_sample: int = 0
    canonical_end_sample: int = 0


@dataclass(frozen=True)
class _CanonicalSegment:
    """A canonical sample range with its conservative availability attribution."""

    canonical_start: int
    canonical_end: int
    source: AudioSourceSegment


class SourceSegmentLedger:
    """Stores only the bounded source metadata needed by retained windows."""

    def __init__(self) -> None:
        """Creates an empty ordered ledger."""
        self._segments: list[_CanonicalSegment] = []

    def append(self, canonical_start: int, sample_count: int, source: AudioSourceSegment) -> None:
        """Appends one non-overlapping canonical segment."""
        if canonical_start < 0 or sample_count < 1:
            raise ValueError("canonical segment range is invalid")
        canonical_end = canonical_start + sample_count
        if self._segments and canonical_start < self._segments[-1].canonical_end:
            raise ValueError("canonical segment ranges overlap")
        self._segments.append(_CanonicalSegment(canonical_start, canonical_end, source))

    def metadata_for_window(self, canonical_start: int, canonical_end: int, window_sequence: int) -> AudioWindowMetadata:
        """Builds conservative availability metadata from segments touched by a window."""
        if canonical_end <= canonical_start:
            raise ValueError("window range is invalid")
        contributing = [segment.source for segment in self._segments if segment.canonical_end > canonical_start and segment.canonical_start < canonical_end]
        if not contributing:
            raise ValueError("window has no source metadata")
        return AudioWindowMetadata(
            audio_source_frame_start=min(segment.source_frame_start for segment in contributing),
            audio_source_frame_end=max(segment.source_frame_end for segment in contributing),
            audio_source_transport_sequence_start=min(segment.transport_sequence_start for segment in contributing),
            audio_source_transport_sequence_end=max(segment.transport_sequence_end for segment in contributing),
            audio_source_gap_count=sum(segment.source_gap_count for segment in contributing),
            audio_window_sequence=window_sequence,
            canonical_start_sample=canonical_start,
            canonical_end_sample=canonical_end,
        )

    def prune_before(self, canonical_position: int) -> None:
        """Drops segments that cannot contribute to any retained future window."""
        self._segments = [segment for segment in self._segments if segment.canonical_end > canonical_position]
