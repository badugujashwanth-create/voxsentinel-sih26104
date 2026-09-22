"""Deterministic canonical-audio windows for speaker verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.audio_source_ledger import AudioSourceSegment, SourceSegmentLedger

SPEAKER_WINDOW_SAMPLES = 32_000
SPEAKER_WINDOW_HOP_SAMPLES = 16_000


@dataclass(frozen=True)
class SpeakerProbeMetadata:
    """Identifies the canonical interval represented by one speaker probe."""

    speaker_window_sequence: int
    canonical_start_sample: int
    canonical_end_sample: int
    source_frame_start: int | None = None
    source_frame_end: int | None = None
    transport_sequence_start: int | None = None
    transport_sequence_end: int | None = None
    source_gap_count: int = 0


@dataclass(frozen=True)
class SpeakerProbe:
    """One immutable speaker probe and its causal audio metadata."""

    samples: np.ndarray
    metadata: SpeakerProbeMetadata | None

    def __array__(self, dtype=None):
        """Allows legacy numerical consumers to view the probe samples."""
        return np.asarray(self.samples, dtype=dtype)


class SpeakerProbeScheduler:
    """Builds bounded two-second probe windows from continuous canonical PCM."""

    def __init__(self) -> None:
        """Creates an empty speaker probe buffer."""
        self._buffer = np.empty(0, dtype=np.float32)
        self._next_start = 0
        self._buffer_origin = 0
        self._ledger = SourceSegmentLedger()
        self._window_sequence = 0

    def push(self, samples: np.ndarray, source_segment: AudioSourceSegment | None = None) -> list[SpeakerProbe]:
        """Returns complete overlapping speaker windows without padding."""
        incoming = np.asarray(samples, dtype=np.float32)
        if incoming.ndim != 1 or not np.isfinite(incoming).all():
            raise ValueError("speaker scheduler requires finite one-dimensional samples")
        if incoming.size == 0:
            return []
        canonical_start = self._buffer_origin + self._buffer.size
        if source_segment is not None:
            self._ledger.append(canonical_start, incoming.size, source_segment)
        self._buffer = np.concatenate((self._buffer, incoming))
        probes: list[SpeakerProbe] = []
        while self._next_start + SPEAKER_WINDOW_SAMPLES <= self._buffer.size:
            start = self._next_start
            end = start + SPEAKER_WINDOW_SAMPLES
            self._window_sequence += 1
            global_start = self._buffer_origin + start
            global_end = self._buffer_origin + end
            source_metadata = self._ledger.metadata_for_window(global_start, global_end, self._window_sequence) if source_segment is not None else None
            metadata = None if source_metadata is None else SpeakerProbeMetadata(
                speaker_window_sequence=self._window_sequence,
                canonical_start_sample=global_start,
                canonical_end_sample=global_end,
                source_frame_start=source_metadata.audio_source_frame_start,
                source_frame_end=source_metadata.audio_source_frame_end,
                transport_sequence_start=source_metadata.audio_source_transport_sequence_start,
                transport_sequence_end=source_metadata.audio_source_transport_sequence_end,
                source_gap_count=source_metadata.audio_source_gap_count,
            )
            probes.append(SpeakerProbe(np.ascontiguousarray(self._buffer[start:end], dtype=np.float32), metadata))
            self._next_start += SPEAKER_WINDOW_HOP_SAMPLES
        if self._next_start:
            discarded = self._next_start
            self._buffer = self._buffer[discarded:]
            self._buffer_origin += discarded
            self._ledger.prune_before(self._buffer_origin)
            self._next_start = 0
        return probes