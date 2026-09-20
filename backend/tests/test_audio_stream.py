"""Tests for the call-scoped microphone audio stream."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.services.async_session import AudioSessionRegistry
from app.services.audio_protocol import AudioStartMetadata, AudioProtocolError, encode_audio_frame
from app.services.audio_stream import AudioStreamHandler


def test_audio_stream_reuses_registered_session_and_enqueues_windows() -> None:
    """Accepted microphone frames feed the pre-registered session only."""
    async def exercise() -> None:
        """Runs the shared-session audio stream assertion."""
        registry = AudioSessionRegistry()
        session = registry.register("call-1")
        handler = AudioStreamHandler(registry, session)
        await handler.start(AudioStartMetadata(sample_rate=16_000, channels=1))
        for sequence in range(1, 6):
            payload = np.zeros(1_024, dtype="<f4").tobytes()
            await handler.receive_binary(encode_audio_frame(
                frame_sequence=sequence,
                first_sample_frame=(sequence - 1) * 1_024,
                sample_count_per_channel=1_024,
                channels=1,
                payload=payload,
            ))
        assert registry.get("call-1") is session
        assert session.window_scheduler is not None
        assert session.queue.qsize() == 0
        await handler.close(normal=True)

    asyncio.run(exercise())


def test_audio_stream_rejects_overlapping_source_ranges() -> None:
    """Replayed/overlapping source ranges never enter the canonicalizer."""
    async def exercise() -> None:
        """Runs the overlap rejection assertion."""
        registry = AudioSessionRegistry()
        session = registry.register("call-1")
        handler = AudioStreamHandler(registry, session)
        await handler.start(AudioStartMetadata(sample_rate=16_000, channels=1))
        payload = np.zeros(16, dtype="<f4").tobytes()
        await handler.receive_binary(encode_audio_frame(frame_sequence=1, first_sample_frame=0, sample_count_per_channel=16, channels=1, payload=payload))
        with pytest.raises(AudioProtocolError, match="overlap"):
            await handler.receive_binary(encode_audio_frame(frame_sequence=2, first_sample_frame=8, sample_count_per_channel=16, channels=1, payload=payload))
        await handler.close(normal=False)

    asyncio.run(exercise())


def test_audio_stream_rejects_duplicate_owner() -> None:
    """A second producer cannot claim the active call session."""
    registry = AudioSessionRegistry()
    session = registry.register("call-1")
    first = AudioStreamHandler(registry, session)
    second = AudioStreamHandler(registry, session)
    asyncio.run(first.start(AudioStartMetadata(sample_rate=16_000, channels=1)))
    with pytest.raises(RuntimeError, match="producer"):
        asyncio.run(second.start(AudioStartMetadata(sample_rate=16_000, channels=1)))
    asyncio.run(first.close(normal=True))


def test_normal_close_accounts_for_resampler_flush_output() -> None:
    """Normal-stop flush output is included in acceptance counters and windows."""
    async def exercise() -> None:
        """Runs the flush accounting assertion."""
        from app.services.acceptance_telemetry import AcceptanceTelemetry

        registry = AudioSessionRegistry()
        session = registry.register("call-1")
        telemetry = AcceptanceTelemetry("call-1")
        handler = AudioStreamHandler(registry, session, telemetry)
        await handler.start(AudioStartMetadata(sample_rate=48_000, channels=1))
        payload = np.zeros(1_024, dtype="<f4").tobytes()
        await handler.receive_binary(encode_audio_frame(frame_sequence=1, first_sample_frame=0, sample_count_per_channel=1_024, channels=1, payload=payload))
        before = telemetry.canonical_samples
        await handler.close(normal=True)
        assert telemetry.canonical_samples > before

    asyncio.run(exercise())