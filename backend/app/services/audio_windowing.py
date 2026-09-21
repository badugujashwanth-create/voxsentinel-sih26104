"""Backend-owned exact AASIST window scheduling."""

from __future__ import annotations

import numpy as np

from app.services.audio_source_ledger import AudioSourceSegment, AudioWindowMetadata, SourceSegmentLedger

AASIST_WINDOW_SAMPLES = 64_600
AASIST_WINDOW_HOP_SAMPLES = 8_000


class AASISTWindowScheduler:
    """Builds overlapping exact-size windows from canonical PCM."""

    def __init__(self, window_samples: int = AASIST_WINDOW_SAMPLES, hop_samples: int = AASIST_WINDOW_HOP_SAMPLES) -> None:
        """Creates a scheduler with validated window and hop sizes."""
        if window_samples < 1 or hop_samples < 1 or hop_samples > window_samples:
            raise ValueError("window and hop sizes are invalid")
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self._buffer = np.empty(0, dtype=np.float32)
        self._next_start = 0
        self._buffer_origin = 0
        self._ledger = SourceSegmentLedger()
        self._window_sequence = 0

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Adds canonical samples and returns all newly complete windows."""
        return [window for window, _ in self.push_with_metadata(samples, None)]

    def push_with_metadata(
        self,
        samples: np.ndarray,
        source_segment: AudioSourceSegment | None,
    ) -> list[tuple[np.ndarray, AudioWindowMetadata | None]]:
        """Adds canonical samples and optionally returns causal window metadata."""
        incoming = np.asarray(samples, dtype=np.float32)
        if incoming.ndim != 1 or not np.isfinite(incoming).all():
            raise ValueError("window scheduler requires finite one-dimensional samples")
        if incoming.size == 0:
            return []
        canonical_start = self._buffer_origin + self._buffer.size
        if source_segment is not None:
            self._ledger.append(canonical_start, incoming.size, source_segment)
        self._buffer = np.concatenate((self._buffer, incoming))
        windows: list[tuple[np.ndarray, AudioWindowMetadata | None]] = []
        while self._next_start + self.window_samples <= self._buffer.size:
            start = self._next_start
            end = start + self.window_samples
            self._window_sequence += 1
            global_start = self._buffer_origin + start
            global_end = self._buffer_origin + end
            metadata = self._ledger.metadata_for_window(global_start, global_end, self._window_sequence) if source_segment is not None else None
            windows.append((np.ascontiguousarray(self._buffer[start:end], dtype=np.float32), metadata))
            self._next_start += self.hop_samples
        if self._next_start:
            discarded = self._next_start
            self._buffer = self._buffer[self._next_start :]
            self._buffer_origin += discarded
            self._ledger.prune_before(self._buffer_origin)
            self._next_start = 0
        return windows
