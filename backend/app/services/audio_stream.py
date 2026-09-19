"""Call-scoped microphone transport over the shared ML runtime session."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.async_session import AsyncCallSession, AudioSessionRegistry, SessionClosedError
from app.services.audio_protocol import AudioProtocolError, AudioStartMetadata, ParsedAudioFrame, parse_audio_frame, validate_frame_continuity
from app.services.audio_source_ledger import AudioSourceSegment
from app.services.streaming_audio_canonicalizer import StreamingAudioCanonicalizer


@dataclass(frozen=True)
class CanonicalAudioChunk:
    """Canonical PCM and its conservative source attribution."""

    samples: np.ndarray
    source_segment: AudioSourceSegment | None


class AudioStreamHandler:
    """Owns one producer, canonicalizer, and source timeline for a call."""

    def __init__(self, registry: AudioSessionRegistry, session: AsyncCallSession) -> None:
        """Creates a handler over the already-registered runtime session."""
        self.registry = registry
        self.session = session
        self.metadata: AudioStartMetadata | None = None
        self.canonicalizer: StreamingAudioCanonicalizer | None = None
        self.previous_sequence: int | None = None
        self.previous_source_end: int | None = None
        self.transport_gap_count = 0
        self.source_gap_count = 0
        self._claimed = False
        self._closed = False

    async def start(self, metadata: AudioStartMetadata) -> None:
        """Claims the producer and creates one persistent canonicalizer."""
        if self._closed:
            raise RuntimeError("audio producer is closed")
        if not self.registry.claim_audio(self.session.call_id, self.session.generation_token):
            raise RuntimeError("audio producer is already claimed")
        self._claimed = True
        self.metadata = metadata
        self.canonicalizer = StreamingAudioCanonicalizer(metadata.sample_rate, metadata.channels)

    async def receive_binary(self, payload: bytes) -> None:
        """Validates, canonicalizes, and schedules one binary transport frame."""
        if self._closed or not self._claimed or self.metadata is None or self.canonicalizer is None:
            raise SessionClosedError("audio producer is not ready")
        frame = parse_audio_frame(payload, self.metadata)
        sequence_gap, source_gap = validate_frame_continuity(
            frame.frame_sequence,
            frame.first_sample_frame,
            frame.sample_count_per_channel,
            self.previous_sequence or 0,
            self.previous_source_end,
        )
        self.transport_gap_count += sequence_gap
        self.source_gap_count += source_gap
        source = AudioSourceSegment(
            frame.first_sample_frame,
            frame.source_frame_end,
            self.previous_sequence or frame.frame_sequence,
            frame.frame_sequence,
            source_gap,
        )
        self.previous_sequence = frame.frame_sequence
        self.previous_source_end = frame.source_frame_end
        canonical, _ = self.canonicalizer.push_with_metadata(frame.samples, source)
        if canonical.size:
            self.registry.ingest_canonical(self.session.call_id, CanonicalAudioChunk(canonical, source), frame.frame_sequence)

    async def close(self, normal: bool) -> None:
        """Flushes normal input and releases producer ownership idempotently."""
        if self._closed:
            return
        self._closed = True
        if normal and self.canonicalizer is not None:
            flushed = self.canonicalizer.flush()
            if flushed.size and self._last_source_segment is not None:
                self.registry.ingest_canonical(self.session.call_id, CanonicalAudioChunk(flushed, self._last_source_segment), (self.previous_sequence or 0) + 1)
        if self.canonicalizer is not None:
            self.canonicalizer.close()
        if self._claimed:
            self.registry.release_audio(self.session.call_id, self.session.generation_token)
        if not normal:
            await self.registry.close(self.session.call_id, self.session.generation_token)

    @property
    def _last_source_segment(self) -> AudioSourceSegment | None:
        """Returns the last accepted source segment for conservative flush attribution."""
        if self.previous_sequence is None or self.previous_source_end is None:
            return None
        return AudioSourceSegment(self.previous_source_end, self.previous_source_end, self.previous_sequence, self.previous_sequence, 0)

