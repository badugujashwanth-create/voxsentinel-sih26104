"""Backend-owned exact AASIST window scheduling."""

from __future__ import annotations

import numpy as np

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

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Adds canonical samples and returns all newly complete windows."""
        incoming = np.asarray(samples, dtype=np.float32)
        if incoming.ndim != 1 or not np.isfinite(incoming).all():
            raise ValueError("window scheduler requires finite one-dimensional samples")
        self._buffer = np.concatenate((self._buffer, incoming))
        windows: list[np.ndarray] = []
        while self._next_start + self.window_samples <= self._buffer.size:
            start = self._next_start
            end = start + self.window_samples
            windows.append(np.ascontiguousarray(self._buffer[start:end], dtype=np.float32))
            self._next_start += self.hop_samples
        if self._next_start:
            self._buffer = self._buffer[self._next_start :]
            self._next_start = 0
        return windows
