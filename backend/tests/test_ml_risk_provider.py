"""Tests for the real ML risk provider boundary."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.models.ml_evidence import RawSpoofEvidence
from app.models.risk import RecommendedAction, RiskLevel
from app.services.async_session import AudioSessionRegistry, AudioWindow
from app.services.risk_provider import MLRiskProvider


class ClientDouble:
    """Async ML client double returning controlled raw evidence."""

    def __init__(self, scores: list[float]) -> None:
        """Stores scores to return in request order."""
        self.scores = iter(scores)
        self.health_calls = 0

    async def health(self) -> None:
        """Reports a ready ML service."""
        self.health_calls += 1

    async def infer(self, _: np.ndarray) -> RawSpoofEvidence:
        """Returns the next controlled score."""
        return RawSpoofEvidence(model_id="AASIST/test", raw_spoof_score=next(self.scores), inference_ms=1.0)


class BlockingClientDouble(ClientDouble):
    """Blocks inference so reset races can be tested deterministically."""

    def __init__(self) -> None:
        """Creates synchronization gates for one inference."""
        super().__init__([0.8])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def infer(self, _: np.ndarray) -> RawSpoofEvidence:
        """Waits until the test releases the in-flight inference."""
        self.started.set()
        await self.release.wait()
        return RawSpoofEvidence(model_id="AASIST/test", raw_spoof_score=0.8, inference_ms=1.0)


def test_two_persistent_windows_emit_elevated_review_without_blocking() -> None:
    """Real ML policy emits HIGH callback review after persistence."""
    async def exercise() -> None:
        """Runs the asynchronous provider stream assertion."""
        registry = AudioSessionRegistry()
        client = ClientDouble([0.8, 0.8])
        provider = MLRiskProvider(registry=registry, client=client)
        session = await provider.prepare_session("call-1")
        session.enqueue_window(AudioWindow(1, np.zeros(64_600, dtype=np.float32)))
        session.enqueue_window(AudioWindow(2, np.zeros(64_600, dtype=np.float32)))
        events = []
        cancellation = asyncio.Event()
        async for event in provider.stream(session, cancellation):
            events.append(event)
            if len(events) == 2:
                await session.close()
        assert events[-1].overall_risk_score == 70
        assert events[-1].risk_level is RiskLevel.HIGH
        assert events[-1].recommended_action is RecommendedAction.REQUIRE_CALLBACK
        assert events[-1].speaker_match_score == 0.0
        assert events[-1].synthetic_score_semantics == "uncalibrated"
        assert events[-1].evidence_availability["speaker_match_score"] == "NOT_EVALUATED"
        assert client.health_calls == 1

    asyncio.run(exercise())


def test_reset_discards_in_flight_result() -> None:
    """A result from a closed generation is never emitted."""
    async def exercise() -> None:
        """Runs the reset race assertion."""
        registry = AudioSessionRegistry()
        client = BlockingClientDouble()
        provider = MLRiskProvider(registry=registry, client=client)
        session = await provider.prepare_session("call-2")
        session.enqueue_window(AudioWindow(1, np.zeros(64_600, dtype=np.float32)))
        cancellation = asyncio.Event()
        stream = provider.stream(session, cancellation)
        task = asyncio.create_task(stream.__anext__())
        await client.started.wait()
        await provider.close_session("call-2", session.generation_token)
        cancellation.set()
        client.release.set()
        with pytest.raises(StopAsyncIteration):
            await task

    asyncio.run(exercise())


def test_aggregation_survives_stream_reconnect() -> None:
    """A reconnect continues the same call's persistence state."""
    async def exercise() -> None:
        """Runs the reconnect persistence assertion."""
        registry = AudioSessionRegistry()
        client = ClientDouble([0.8, 0.8])
        provider = MLRiskProvider(registry=registry, client=client)
        session = await provider.prepare_session("call-3")
        session.enqueue_window(AudioWindow(1, np.zeros(64_600, dtype=np.float32)))
        first = provider.stream(session, asyncio.Event())
        await first.__anext__()
        session.enqueue_window(AudioWindow(2, np.zeros(64_600, dtype=np.float32)))
        second = provider.stream(session, asyncio.Event())
        event = await second.__anext__()
        assert event.overall_risk_score == 70

    asyncio.run(exercise())
