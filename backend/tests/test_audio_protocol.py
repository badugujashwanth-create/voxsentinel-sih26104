"""Tests for the versioned VXAF microphone transport contract."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from app.services.audio_protocol import (
    AUDIO_HEADER_LENGTH,
    AudioProtocolError,
    AudioStartMetadata,
    encode_audio_frame,
    parse_audio_frame,
    validate_frame_continuity,
    validate_source_frame_range,
)


def _frame(
    *,
    sequence: int = 1,
    first_sample_frame: int = 0,
    samples: tuple[float, ...] = (0.25, -0.5),
    channels: int = 1,
) -> bytes:
    """Builds a valid frame for protocol tests."""
    payload = np.asarray(samples, dtype="<f4").tobytes()
    return encode_audio_frame(
        frame_sequence=sequence,
        first_sample_frame=first_sample_frame,
        sample_count_per_channel=len(samples) // channels,
        channels=channels,
        payload=payload,
    )


def test_vxaf_frame_has_exact_little_endian_header() -> None:
    """The encoder writes the locked 32-byte VXAF header."""
    payload = _frame(sequence=7, first_sample_frame=19)
    assert len(payload) == AUDIO_HEADER_LENGTH + 8
    assert payload[:4] == b"VXAF"
    assert payload[4] == 1
    assert payload[5] == 32
    assert struct.unpack_from("<H", payload, 6)[0] == 0
    assert struct.unpack_from("<Q", payload, 8)[0] == 7
    assert struct.unpack_from("<Q", payload, 16)[0] == 19
    assert struct.unpack_from("<I", payload, 24)[0] == 2
    assert struct.unpack_from("<I", payload, 28)[0] == 8


def test_parser_rejects_invalid_header_and_payload() -> None:
    """Malformed frames fail before any audio reaches the canonicalizer."""
    metadata = AudioStartMetadata(sample_rate=48_000, channels=1)
    malformed = bytearray(_frame())
    malformed[0:4] = b"NOPE"
    with pytest.raises(AudioProtocolError, match="magic"):
        parse_audio_frame(bytes(malformed), metadata)

    malformed = bytearray(_frame())
    struct.pack_into("<I", malformed, 28, 99)
    with pytest.raises(AudioProtocolError, match="payload"):
        parse_audio_frame(bytes(malformed), metadata)


def test_source_frame_ranges_require_strict_forward_progress() -> None:
    """Overlapping/replayed source ranges are rejected while gaps are observable."""
    assert validate_source_frame_range(0, 2, None) == 0
    assert validate_source_frame_range(2, 2, 1) == 0
    assert validate_source_frame_range(8, 2, 3) == 4
    with pytest.raises(AudioProtocolError, match="overlap"):
        validate_source_frame_range(3, 2, 3)


def test_source_frame_range_rejects_uint64_overflow() -> None:
    """Source end positions cannot wrap around the uint64 timeline."""
    with pytest.raises(AudioProtocolError, match="overflow"):
        validate_source_frame_range(2**64 - 1, 2, None)


def test_frame_continuity_rejects_sequence_regression_and_source_overlap() -> None:
    """Transport and source timelines are each monotonic and independently validated."""
    assert validate_frame_continuity(2, 2, 1, 1) == (0, 0)
    with pytest.raises(AudioProtocolError, match="sequence"):
        validate_frame_continuity(1, 4, 2, 2)
    with pytest.raises(AudioProtocolError, match="overlap"):
        validate_frame_continuity(3, 2, 3, 2, 2)
