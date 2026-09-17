"""Typed acknowledgements for backend audio ingestion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioIngestAcknowledgement:
    """Reports canonicalization and bounded-queue results without audio data."""

    call_id: str
    chunk_sequence: int
    accepted_sample_count: int
    windows_enqueued: int
    dropped_window_count: int
    preprocessing_ms: float = 0.0
