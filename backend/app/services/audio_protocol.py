"""Validation and encoding for the versioned VoxSentinel audio transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np

VXAF_MAGIC = b"VXAF"
AUDIO_PROTOCOL_VERSION = 1
AUDIO_HEADER_LENGTH = 32
AUDIO_FLAGS = 0
MIN_SOURCE_SAMPLE_RATE = 8_000
MAX_SOURCE_SAMPLE_RATE = 96_000
MAX_TRANSPORT_SAMPLES = 1_024
UINT64_MAX = 2**64 - 1
SUPPORTED_CHANNELS = frozenset({1, 2})


class AudioProtocolError(ValueError):
    """Raised when a microphone transport message violates the protocol."""


@dataclass(frozen=True)
class AudioStartMetadata:
    """Validated metadata announced before binary PCM frames."""

    sample_rate: int
    channels: int
    sample_format: str = "float32le"
    protocol_version: int = AUDIO_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        """Validates the negotiated source format."""
        if self.protocol_version != AUDIO_PROTOCOL_VERSION:
            raise AudioProtocolError("unsupported audio protocol version")
        if not isinstance(self.sample_rate, int) or not MIN_SOURCE_SAMPLE_RATE <= self.sample_rate <= MAX_SOURCE_SAMPLE_RATE:
            raise AudioProtocolError("unsupported source sample rate")
        if self.channels not in SUPPORTED_CHANNELS:
            raise AudioProtocolError("unsupported channel count")
        if self.sample_format != "float32le":
            raise AudioProtocolError("unsupported sample format")


@dataclass(frozen=True)
class ParsedAudioFrame:
    """A validated binary PCM frame and its source-timeline metadata."""

    frame_sequence: int
    first_sample_frame: int
    sample_count_per_channel: int
    payload_byte_length: int
    channels: int
    samples: np.ndarray
    source_frame_end: int


def encode_audio_frame(
    *,
    frame_sequence: int,
    first_sample_frame: int,
    sample_count_per_channel: int,
    channels: int,
    payload: bytes,
) -> bytes:
    """Encodes a validated VXAF header followed by interleaved float32 PCM."""
    _validate_header_values(frame_sequence, first_sample_frame, sample_count_per_channel, channels, len(payload))
    if len(payload) != sample_count_per_channel * channels * 4:
        raise AudioProtocolError("payload length does not match sample count")
    return struct.pack(
        "<4sBBHQQII",
        VXAF_MAGIC,
        AUDIO_PROTOCOL_VERSION,
        AUDIO_HEADER_LENGTH,
        AUDIO_FLAGS,
        frame_sequence,
        first_sample_frame,
        sample_count_per_channel,
        len(payload),
    ) + payload


def parse_audio_frame(payload: bytes, metadata: AudioStartMetadata) -> ParsedAudioFrame:
    """Parses and validates one VXAF frame against negotiated metadata."""
    if len(payload) < AUDIO_HEADER_LENGTH:
        raise AudioProtocolError("audio frame is shorter than header")
    magic, version, header_length, flags, sequence, first_frame, count, payload_length = struct.unpack_from(
        "<4sBBHQQII", payload, 0
    )
    if magic != VXAF_MAGIC:
        raise AudioProtocolError("invalid audio frame magic")
    if version != AUDIO_PROTOCOL_VERSION:
        raise AudioProtocolError("unsupported audio frame version")
    if header_length != AUDIO_HEADER_LENGTH:
        raise AudioProtocolError("invalid audio frame header length")
    if flags != AUDIO_FLAGS:
        raise AudioProtocolError("unsupported audio frame flags")
    _validate_header_values(sequence, first_frame, count, metadata.channels, payload_length)
    expected_payload_length = count * metadata.channels * 4
    if payload_length != expected_payload_length or len(payload) != AUDIO_HEADER_LENGTH + payload_length:
        raise AudioProtocolError("audio frame payload length is invalid")
    audio_payload = payload[AUDIO_HEADER_LENGTH:]
    samples = np.frombuffer(audio_payload, dtype="<f4").copy()
    if not np.isfinite(samples).all():
        raise AudioProtocolError("audio frame contains non-finite PCM")
    source_end = first_frame + count - 1
    return ParsedAudioFrame(sequence, first_frame, count, payload_length, metadata.channels, samples, source_end)


def validate_source_frame_range(first_sample_frame: int, sample_count_per_channel: int, previous_source_frame_end: int | None) -> int:
    """Validates strict source-range progress and returns a gap count."""
    if not isinstance(first_sample_frame, int) or not 0 <= first_sample_frame <= UINT64_MAX:
        raise AudioProtocolError("source frame start is outside uint64 range")
    if not isinstance(sample_count_per_channel, int) or not 1 <= sample_count_per_channel <= MAX_TRANSPORT_SAMPLES:
        raise AudioProtocolError("sample count is invalid")
    source_end = first_sample_frame + sample_count_per_channel - 1
    if source_end > UINT64_MAX:
        raise AudioProtocolError("source frame range overflow")
    if previous_source_frame_end is None:
        return 0
    if first_sample_frame <= previous_source_frame_end:
        raise AudioProtocolError("source frame range overlaps a previously accepted range")
    return max(0, first_sample_frame - previous_source_frame_end - 1)


def validate_frame_continuity(
    frame_sequence: int,
    first_sample_frame: int,
    sample_count_per_channel: int,
    previous_frame_sequence: int,
    previous_source_frame_end: int | None = None,
) -> tuple[int, int]:
    """Validates transport/source progress and returns sequence/source gaps."""
    if frame_sequence <= previous_frame_sequence:
        raise AudioProtocolError("frame sequence must increase")
    source_gap = validate_source_frame_range(first_sample_frame, sample_count_per_channel, previous_source_frame_end)
    return frame_sequence - previous_frame_sequence - 1, source_gap


def _validate_header_values(sequence: int, first_frame: int, count: int, channels: int, payload_length: int) -> None:
    """Validates fixed-width VXAF numeric fields."""
    if not isinstance(sequence, int) or not 1 <= sequence <= UINT64_MAX:
        raise AudioProtocolError("frame sequence is outside uint64 range")
    if not isinstance(first_frame, int) or not 0 <= first_frame <= UINT64_MAX:
        raise AudioProtocolError("source frame start is outside uint64 range")
    if not isinstance(count, int) or not 1 <= count <= MAX_TRANSPORT_SAMPLES:
        raise AudioProtocolError("sample count is invalid")
    if channels not in SUPPORTED_CHANNELS:
        raise AudioProtocolError("unsupported channel count")
    if not isinstance(payload_length, int) or payload_length < 0:
        raise AudioProtocolError("payload length is invalid")
    if not math.isfinite(float(payload_length)):
        raise AudioProtocolError("payload length is invalid")
