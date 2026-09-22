"""Deterministic canonical-audio windows for speaker verification."""

from __future__ import annotations

import numpy as np

SPEAKER_WINDOW_SAMPLES = 32_000
SPEAKER_WINDOW_HOP_SAMPLES = 16_000


class SpeakerProbeScheduler:
    """Builds bounded two-second probe windows from continuous canonical PCM."""

    def __init__(self) -> None:
        """Creates an empty speaker probe buffer."""
        self._buffer = np.empty(0, dtype=np.float32)
        self._next_start = 0

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Returns complete overlapping speaker windows without padding."""
        incoming = np.asarray(samples, dtype=np.float32)
        if incoming.ndim != 1 or not np.isfinite(incoming).all():
            raise ValueError("speaker scheduler requires finite one-dimensional samples")
        if incoming.size == 0:
            return []
        self._buffer = np.concatenate((self._buffer, incoming))
        windows: list[np.ndarray] = []
        while self._next_start + SPEAKER_WINDOW_SAMPLES <= self._buffer.size:
            start = self._next_start
            windows.append(np.ascontiguousarray(self._buffer[start : start + SPEAKER_WINDOW_SAMPLES], dtype=np.float32))
            self._next_start += SPEAKER_WINDOW_HOP_SAMPLES
        if self._next_start:
            self._buffer = self._buffer[self._next_start :]
            self._next_start = 0
        return windows
