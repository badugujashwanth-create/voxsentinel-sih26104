"""Deterministic mock risk engine tests.

The expected values here are transcribed from ``SCENARIOS`` in
``frontend/src/scenarios/scenarios.ts``. The backend must replay those fixtures
exactly so the console looks the same offline and live.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from app.models.call import CallSession, CallStatus, ScenarioId
from app.models.risk import LiveRiskEvent, RecommendedAction, RiskLevel
from app.services.risk_provider import (
    BASELINE_REASON,
    EMERGING_DRIFT_REASON,
    EMERGING_SYNTHESIS_REASON,
    FINANCIAL_REQUEST_REASON,
    INCREASING_MISMATCH_REASON,
    MISMATCH_REASON,
    MONITORED_FINANCIAL_REASON,
    PROSODY_REASON,
    SENSITIVE_REQUEST_REASON,
    SYNTHETIC_REASON,
    MockRiskProvider,
)

PROBABILITY_FIELDS = (
    "synthetic_probability",
    "speaker_match_score",
    "speaker_mismatch_score",
    "prosody_anomaly_score",
    "replay_risk_score",
    "context_risk_score",
)

MONITOR = RecommendedAction.MONITOR
BLOCK = RecommendedAction.BLOCK_ACTION
CALLBACK = RecommendedAction.REQUIRE_CALLBACK

#: (overall_risk_score, recommended_action, reasons) per step, per scenario,
#: mirroring the frontend fixtures exactly.
EXPECTED_STEPS: dict[ScenarioId, list[tuple[int, RecommendedAction, tuple[str, ...]]]] = {
    ScenarioId.GENUINE: [
        (12, MONITOR, (BASELINE_REASON,)),
        (10, MONITOR, (BASELINE_REASON,)),
        (14, MONITOR, (BASELINE_REASON,)),
        (11, MONITOR, (BASELINE_REASON,)),
        (13, MONITOR, (BASELINE_REASON,)),
    ],
    ScenarioId.HUMAN_IMPOSTOR: [
        (18, MONITOR, (BASELINE_REASON,)),
        (26, MONITOR, (EMERGING_DRIFT_REASON,)),
        (39, MONITOR, (INCREASING_MISMATCH_REASON,)),
        (54, MONITOR, (MISMATCH_REASON, SENSITIVE_REQUEST_REASON)),
        (67, MONITOR, (MISMATCH_REASON, FINANCIAL_REQUEST_REASON)),
        (76, MONITOR, (MISMATCH_REASON, FINANCIAL_REQUEST_REASON)),
    ],
    ScenarioId.AI_CLONE: [
        (20, MONITOR, (BASELINE_REASON,)),
        (31, MONITOR, (EMERGING_SYNTHESIS_REASON,)),
        (46, MONITOR, (SYNTHETIC_REASON,)),
        (63, MONITOR, (SYNTHETIC_REASON, PROSODY_REASON)),
        (78, MONITOR, (SYNTHETIC_REASON, MISMATCH_REASON)),
        (88, BLOCK, (SYNTHETIC_REASON, MISMATCH_REASON, PROSODY_REASON)),
    ],
    ScenarioId.HIGH_VALUE_TRANSFER_ATTACK: [
        (18, MONITOR, (BASELINE_REASON,)),
        (27, MONITOR, (MONITORED_FINANCIAL_REASON,)),
        (43, MONITOR, (EMERGING_SYNTHESIS_REASON, FINANCIAL_REQUEST_REASON)),
        (61, MONITOR, (SYNTHETIC_REASON, MISMATCH_REASON)),
        (79, CALLBACK, (SYNTHETIC_REASON, MISMATCH_REASON, FINANCIAL_REQUEST_REASON)),
        (92, BLOCK, (SYNTHETIC_REASON, MISMATCH_REASON, FINANCIAL_REQUEST_REASON)),
    ],
}

EXPECTED_FINAL_LEVELS = {
    ScenarioId.GENUINE: RiskLevel.LOW,
    ScenarioId.HUMAN_IMPOSTOR: RiskLevel.HIGH,
    ScenarioId.AI_CLONE: RiskLevel.CRITICAL,
    ScenarioId.HIGH_VALUE_TRANSFER_ATTACK: RiskLevel.CRITICAL,
}

#: Full field-by-field final event of each scenario, from the same fixtures.
EXPECTED_FINAL_SCORES: dict[ScenarioId, dict[str, float]] = {
    ScenarioId.GENUINE: {"synthetic_probability": 0.08, "speaker_match_score": 0.92, "speaker_mismatch_score": 0.08, "prosody_anomaly_score": 0.11, "replay_risk_score": 0.06, "context_risk_score": 0.12},
    ScenarioId.HUMAN_IMPOSTOR: {"synthetic_probability": 0.08, "speaker_match_score": 0.24, "speaker_mismatch_score": 0.76, "prosody_anomaly_score": 0.11, "replay_risk_score": 0.06, "context_risk_score": 0.82},
    ScenarioId.AI_CLONE: {"synthetic_probability": 0.88, "speaker_match_score": 0.22, "speaker_mismatch_score": 0.78, "prosody_anomaly_score": 0.86, "replay_risk_score": 0.06, "context_risk_score": 0.12},
    ScenarioId.HIGH_VALUE_TRANSFER_ATTACK: {"synthetic_probability": 0.91, "speaker_match_score": 0.28, "speaker_mismatch_score": 0.72, "prosody_anomaly_score": 0.74, "replay_risk_score": 0.21, "context_risk_score": 0.95},
}


def make_session(scenario: ScenarioId) -> CallSession:
    """Builds a live session for one scenario."""
    return CallSession(call_id="test-call", status=CallStatus.LIVE, claimed_identity="CEO Demo", scenario=scenario, created_at=datetime.now(UTC))


def events_for(scenario: ScenarioId) -> list[LiveRiskEvent]:
    """Collects the full scripted stream for one scenario."""
    async def collect() -> list[LiveRiskEvent]:
        """Collects events from the asynchronous provider stream."""
        return [event async for event in MockRiskProvider(emit_interval_ms=0).stream(make_session(scenario), asyncio.Event())]

    return asyncio.run(collect())


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_steps_match_the_frontend_fixtures(scenario: ScenarioId) -> None:
    """Score, action, and reasons match the fixtures step for step."""
    actual = [(event.overall_risk_score, event.recommended_action, tuple(event.reasons)) for event in events_for(scenario)]
    assert actual == EXPECTED_STEPS[scenario]


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_progression_matches_specification(scenario: ScenarioId) -> None:
    """Each scenario replays its agreed score progression."""
    expected = [score for score, _, _ in EXPECTED_STEPS[scenario]]
    assert [event.overall_risk_score for event in events_for(scenario)] == expected


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_final_risk_level(scenario: ScenarioId) -> None:
    """Each scenario lands on its agreed final severity."""
    assert events_for(scenario)[-1].risk_level is EXPECTED_FINAL_LEVELS[scenario]


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_final_event_scores_match_the_frontend_fixtures(scenario: ScenarioId) -> None:
    """Every probability field of the final event matches the fixtures."""
    final = events_for(scenario)[-1]
    for name, value in EXPECTED_FINAL_SCORES[scenario].items():
        assert getattr(final, name) == pytest.approx(value), f"{name} differs in {scenario}"


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_scores_stay_inside_contract_ranges(scenario: ScenarioId) -> None:
    """Probabilities stay in 0..1 and the overall score in 0..100."""
    for event in events_for(scenario):
        assert 0 <= event.overall_risk_score <= 100
        for name in PROBABILITY_FIELDS:
            assert 0.0 <= getattr(event, name) <= 1.0, f"{name} out of range in {scenario}"


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_sequence_and_timestamps_increase(scenario: ScenarioId) -> None:
    """Sequence starts at 1 and both ordering fields increase."""
    events = events_for(scenario)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    timestamps = [event.timestamp_ms for event in events]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_timestamps_match_the_fixture_interval(scenario: ScenarioId) -> None:
    """timestamp_ms is sequence * 700ms, as the fixtures compute it."""
    for event in events_for(scenario):
        assert event.timestamp_ms == event.sequence * 700


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_risk_level_always_matches_score_band(scenario: ScenarioId) -> None:
    """Every event satisfies the invariant the console re-checks."""
    for event in events_for(scenario):
        assert event.risk_level is LiveRiskEvent.model_validate(event.model_dump()).risk_level


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_reasons_are_always_present_and_non_empty(scenario: ScenarioId) -> None:
    """The console rejects blank reason strings, so none are emitted."""
    for event in events_for(scenario):
        assert event.reasons
        assert all(reason.strip() for reason in event.reasons)


def test_genuine_call_stays_low_and_only_monitors() -> None:
    """A genuine caller never leaves LOW and never escalates past MONITOR."""
    events = events_for(ScenarioId.GENUINE)
    assert all(event.risk_level is RiskLevel.LOW for event in events)
    assert all(event.recommended_action is RecommendedAction.MONITOR for event in events)
    assert all(event.synthetic_probability < 0.3 for event in events)
    assert all(event.speaker_match_score > 0.7 for event in events)


def test_human_impostor_is_identity_driven_not_synthetic() -> None:
    """Human impostor keeps synthesis at baseline while mismatch climbs."""
    events = events_for(ScenarioId.HUMAN_IMPOSTOR)
    assert all(event.synthetic_probability == pytest.approx(0.08) for event in events)
    assert events[-1].speaker_mismatch_score > events[0].speaker_mismatch_score
    assert events[-1].speaker_mismatch_score > 0.7
    assert MISMATCH_REASON in events[-1].reasons


def test_ai_clone_escalates_to_a_block() -> None:
    """AI clone ends CRITICAL and blocked, as the fixtures do."""
    final = events_for(ScenarioId.AI_CLONE)[-1]
    assert final.risk_level is RiskLevel.CRITICAL
    assert final.synthetic_probability > 0.8
    assert final.prosody_anomaly_score > 0.7
    assert final.speaker_mismatch_score > 0.5
    assert final.recommended_action is RecommendedAction.BLOCK_ACTION


def test_transfer_attack_final_event_blocks_with_all_three_reasons() -> None:
    """The primary demo ends CRITICAL, BLOCK_ACTION, with the agreed reasons."""
    final = events_for(ScenarioId.HIGH_VALUE_TRANSFER_ATTACK)[-1]
    assert final.overall_risk_score == 92
    assert final.risk_level is RiskLevel.CRITICAL
    assert final.recommended_action is RecommendedAction.BLOCK_ACTION
    assert SYNTHETIC_REASON in final.reasons
    assert MISMATCH_REASON in final.reasons
    assert FINANCIAL_REQUEST_REASON in final.reasons


def test_transfer_attack_requests_callback_before_blocking() -> None:
    """The step before the block asks for a verified callback."""
    assert events_for(ScenarioId.HIGH_VALUE_TRANSFER_ATTACK)[-2].recommended_action is RecommendedAction.REQUIRE_CALLBACK


def test_stream_is_deterministic() -> None:
    """Replaying a scenario twice produces identical events."""
    first = [event.model_dump() for event in events_for(ScenarioId.HIGH_VALUE_TRANSFER_ATTACK)]
    second = [event.model_dump() for event in events_for(ScenarioId.HIGH_VALUE_TRANSFER_ATTACK)]
    assert first == second


def test_live_risk_event_accepts_optional_audio_correlation_metadata() -> None:
    """Risk transport can carry source timing without changing policy fields."""
    event = LiveRiskEvent(
        call_id="call-1", sequence=1, timestamp_ms=1, synthetic_probability=0.1,
        speaker_match_score=0.0, speaker_mismatch_score=0.0, prosody_anomaly_score=0.0,
        replay_risk_score=0.0, context_risk_score=0.0, overall_risk_score=20,
        risk_level=RiskLevel.LOW, reasons=[BASELINE_REASON], recommended_action=MONITOR,
        audio_source_frame_start=100, audio_source_frame_end=200,
        audio_source_transport_sequence_start=1, audio_source_transport_sequence_end=2,
        audio_source_gap_count=0, audio_window_sequence=1,
    )
    assert event.audio_source_frame_end == 200
