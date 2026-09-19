"""Stateful browser-audio canonicalization for the backend pipeline."""

from __future__ import annotations

import numpy as np
import soxr

from app.services.audio_source_ledger import AudioSourceSegment

CANONICAL_SAMPLE_RATE = 16_000


class StreamingAudioCanonicalizer:
    """Downmixes and continuously resamples one live audio producer."""

    def __init__(self, source_rate: int, channels: int) -> None:
        """Creates one persistent resampler for an active producer."""
        if source_rate < 8_000 or source_rate > 96_000:
            raise ValueError("source rate must be between 8000 and 96000 Hz")
        if channels not in (1, 2):
            raise ValueError("channels must be mono or stereo")
        self.source_rate = source_rate
        self.channels = channels
        self._resampler = soxr.ResampleStream(source_rate, CANONICAL_SAMPLE_RATE, 1, dtype="float32")
        self._closed = False
        self._flushed = False
        self._last_source_segment: AudioSourceSegment | None = None

    def push(self, interleaved_samples: np.ndarray) -> np.ndarray:
        """Downmixes and resamples a finite interleaved source chunk."""
        output, _ = self.push_with_metadata(interleaved_samples, None)
        return output

    def push_with_metadata(
        self,
        interleaved_samples: np.ndarray,
        source_segment: AudioSourceSegment | None,
    ) -> tuple[np.ndarray, AudioSourceSegment | None]:
        """Converts a chunk and attributes output to its causally available source range."""
        self._ensure_open()
        samples = np.asarray(interleaved_samples, dtype=np.float32)
        if samples.ndim != 1 or samples.size % self.channels:
            raise ValueError("audio chunk must be one-dimensional and channel aligned")
        if not np.isfinite(samples).all():
            raise ValueError("audio chunk must contain finite PCM")
        if source_segment is not None:
            self._last_source_segment = source_segment
        mono = samples if self.channels == 1 else samples.reshape(-1, self.channels).mean(axis=1, dtype=np.float32)
        output = np.asarray(self._resampler.resample_chunk(mono, last=False), dtype=np.float32)
        return output, source_segment if output.size else None

    def flush(self) -> np.ndarray:
        """Flushes only source-derived samples and closes the stream for input."""
        self._ensure_open()
        if self._flushed:
            return np.empty(0, dtype=np.float32)
        self._flushed = True
        output = np.asarray(self._resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True), dtype=np.float32)
        return output

    def close(self) -> None:
        """Disposes the resampler and prevents future input."""
        self._closed = True
        self._resampler = None

    def _ensure_open(self) -> None:
        """Rejects input after disposal or end-of-stream."""
        if self._closed or self._flushed:
            raise RuntimeError("audio canonicalizer is closed")
