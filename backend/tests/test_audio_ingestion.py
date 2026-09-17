"""Tests for bounded audio-window session ingestion."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.async_session import AsyncCallSession, AudioWindow, SessionClosedError


def test_queue_drops_oldest_pending_window_and_keeps_newest() -> None:
    """Queue saturation evicts the oldest pending window deterministically."""
    session = AsyncCallSession(call_id="call-1", max_pending_windows=2)
    session.enqueue_window(AudioWindow(sequence=1, samples=np.array([1], dtype=np.float32)))
    session.enqueue_window(AudioWindow(sequence=2, samples=np.array([2], dtype=np.float32)))
    session.enqueue_window(AudioWindow(sequence=3, samples=np.array([3], dtype=np.float32)))
    assert session.dropped_window_count == 1
    assert [session.queue.get_nowait().sequence, session.queue.get_nowait().sequence] == [2, 3]


def test_closed_queue_rejects_new_windows() -> None:
    """Cancelled sessions reject input instead of accepting dead-queue work."""
    async def exercise() -> None:
        """Runs the asynchronous session close assertion."""
        session = AsyncCallSession(call_id="call-1")
        await session.close()
        with pytest.raises(SessionClosedError):
            session.enqueue_window(AudioWindow(sequence=1, samples=np.zeros(1, dtype=np.float32)))

    import asyncio

    asyncio.run(exercise())
