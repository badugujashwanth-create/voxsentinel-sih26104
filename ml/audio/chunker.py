"""Streaming audio windowing.

Buffering only: this module performs no classification, no feature extraction,
and no inference. It turns a stream of mono samples into the fixed overlapping
windows a future detector will consume.

Raw audio is never written to disk or logged here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_HOP_SECONDS = 0.5


class AudioChunker:
    """Splits a mono sample stream into fixed overlapping windows.

    With the defaults, a 32000-sample window advances by an 8000-sample hop, so
    consecutive windows share 24000 samples.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        hop_seconds: float = DEFAULT_HOP_SECONDS,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        if hop_seconds <= 0:
            raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")

        window_samples = int(sample_rate * window_seconds)
        hop_samples = int(sample_rate * hop_seconds)
        if window_samples <= 0:
            raise ValueError(f"window of {window_seconds}s at {sample_rate}Hz rounds to zero samples")
        if hop_samples <= 0:
            raise ValueError(f"hop of {hop_seconds}s at {sample_rate}Hz rounds to zero samples")
        if hop_samples > window_samples:
            raise ValueError(f"hop ({hop_samples} samples) must not exceed window ({window_samples} samples)")

        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self._buffer: list[float] = []

    @property
    def buffered_samples(self) -> int:
        """Number of samples held but not yet advanced past."""
        return len(self._buffer)

    def push(self, samples: Iterable[float]) -> list[Sequence[float]]:
        """Appends samples and returns every window they complete.

        Returns an empty list until enough audio has accumulated.
        """
        self._buffer.extend(samples)

        windows: list[Sequence[float]] = []
        while len(self._buffer) >= self.window_samples:
            windows.append(self._buffer[: self.window_samples])
            # ponytail: O(n) list trim per window. Fine at 16kHz with a 0.5s
            # hop; move to a ring buffer only if profiling says this matters.
            del self._buffer[: self.hop_samples]
        return windows

    def reset(self) -> None:
        """Drops all buffered audio and returns to the initial state."""
        self._buffer.clear()
