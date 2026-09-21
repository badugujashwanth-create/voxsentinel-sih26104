"""Regression coverage for duplicate microphone-producer cleanup."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.services.async_session import AudioSessionRegistry
from app.services.audio_protocol import AudioStartMetadata, encode_audio_frame
from app.services.audio_stream import AudioStreamHandler


def test_rejected_duplicate_cleanup_preserves_original_owner_and_session() -> None:
    """A rejected producer cannot close or disrupt the active producer session."""
    async def exercise() -> None:
        """Runs the duplicate cleanup isolation assertion."""
        registry = AudioSessionRegistry()
        session = registry.register("call-1")
        first = AudioStreamHandler(registry, session)
        second = AudioStreamHandler(registry, session)
        metadata = AudioStartMetadata(sample_rate=16_000, channels=1)
        payload = np.zeros(1_024, dtype="<f4").tobytes()

        await first.start(metadata)
        with pytest.raises(RuntimeError, match="producer"):
            await second.start(metadata)

        await second.close(normal=False)

        assert not session.is_closed
        assert session._audio_claimed
        assert second.canonicalizer is None
        await first.receive_binary(encode_audio_frame(
            frame_sequence=1,
            first_sample_frame=0,
            sample_count_per_channel=1_024,
            channels=1,
            payload=payload,
        ))
        await first.close(normal=True)
        assert not session.is_closed

    asyncio.run(exercise())
