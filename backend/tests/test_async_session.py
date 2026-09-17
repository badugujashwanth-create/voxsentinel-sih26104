"""Tests for the shared asynchronous ML-session boundary."""

from __future__ import annotations

import asyncio

import pytest

from app.services.async_session import AsyncCallSession, AudioSessionRegistry, SessionClosedError


def test_session_close_unblocks_waiting_consumer() -> None:
    """Closing a session releases a consumer waiting for its next window."""
    async def exercise() -> None:
        """Runs the asynchronous close/unblock assertion."""
        session = AsyncCallSession(call_id="call-1", max_pending_windows=2)
        consumer = asyncio.create_task(session.next_window())
        await asyncio.sleep(0)
        await session.close()
        with pytest.raises(SessionClosedError):
            await consumer

    asyncio.run(exercise())


def test_registry_rejects_duplicate_registration() -> None:
    """A call may have only one runtime session."""
    registry = AudioSessionRegistry()
    registry.register("call-1")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("call-1")


def test_closed_session_rejects_input_and_cleanup_is_idempotent() -> None:
    """Closed sessions reject windows and repeated cleanup is harmless."""
    async def exercise() -> None:
        """Runs the asynchronous cleanup assertions."""
        registry = AudioSessionRegistry()
        session = registry.register("call-1")
        await session.close()
        with pytest.raises(SessionClosedError):
            session.enqueue_window("window")
        await registry.close("call-1", session.generation_token)
        registry.remove("call-1", session.generation_token)
        registry.remove("call-1", session.generation_token)
        assert registry.get("call-1") is None

    asyncio.run(exercise())


def test_generation_token_invalidates_late_results() -> None:
    """Removing a session makes its generation token unusable."""
    registry = AudioSessionRegistry()
    session = registry.register("call-1")
    registry.remove("call-1", session.generation_token)
    assert not registry.is_current("call-1", session.generation_token)


def test_registry_allows_one_stream_claim_and_releases_it() -> None:
    """One call cannot have competing queue consumers."""
    registry = AudioSessionRegistry()
    session = registry.register("call-1")
    assert registry.claim_stream("call-1", session.generation_token)
    assert not registry.claim_stream("call-1", session.generation_token)
    registry.release_stream("call-1", session.generation_token)
    assert registry.claim_stream("call-1", session.generation_token)


def test_session_allocates_monotonic_window_sequences() -> None:
    """A runtime session owns window sequence allocation."""
    session = AsyncCallSession(call_id="call-1")
    assert session.allocate_window_sequence() == 1
    assert session.allocate_window_sequence() == 2
