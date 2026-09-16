"""Deterministic mock risk engine tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.models.call import CallSession, CallStatus, ScenarioId
from app.models.risk import LiveRiskEvent, RecommendedAction, RiskLevel
from app.services.risk_provider import CONTEXT_REASON, MISMATCH_REASON, SYNTHETIC_REASON, MockRiskProvider

PROBABILITY_FIELDS = (
    "synthetic_probability",
    "speaker_match_score",
    "speaker_mismatch_score",
    "prosody_anomaly_score",
    "replay_risk_score",
    "context_risk_score",
)

EXPECTED_PROGRESSIONS = {
    ScenarioId.GENUINE: [12, 10, 14, 11, 13],
    ScenarioId.HUMAN_IMPOSTOR: [18, 26, 39, 54, 67, 76],
    ScenarioId.AI_CLONE: [20, 31, 46, 63, 78, 88],
    ScenarioId.HIGH_VALUE_TRANSFER_ATTACK: [18, 27, 43, 61, 79, 92],
}

EXPECTED_FINAL_LEVELS = {
    ScenarioId.GENUINE: RiskLevel.LOW,
    ScenarioId.HUMAN_IMPOSTOR: RiskLevel.HIGH,
    ScenarioId.AI_CLONE: RiskLevel.CRITICAL,
    ScenarioId.HIGH_VALUE_TRANSFER_ATTACK: RiskLevel.CRITICAL,
}

SECONDARY_VERIFICATION_ACTIONS = {
    RecommendedAction.REQUIRE_OTP,
    RecommendedAction.REQUIRE_CALLBACK,
    RecommendedAction.REQUIRE_VOICE_CHALLENGE,
    RecommendedAction.REQUIRE_SUPERVISOR,
}


def make_session(scenario: ScenarioId) -> CallSession:
    """Builds a live session for one scenario."""
    return CallSession(call_id="test-call", status=CallStatus.LIVE, claimed_identity="CEO Demo", scenario=scenario, created_at=datetime.now(UTC))


def events_for(scenario: ScenarioId) -> list[LiveRiskEvent]:
    """Collects the full scripted stream for one scenario."""
    return list(MockRiskProvider().stream(make_session(scenario)))


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_progression_matches_specification(scenario: ScenarioId) -> None:
    """Each scenario replays its agreed score progression."""
    assert [event.overall_risk_score for event in events_for(scenario)] == EXPECTED_PROGRESSIONS[scenario]


@pytest.mark.parametrize("scenario", list(ScenarioId))
def test_final_risk_level(scenario: ScenarioId) -> None:
    """Each scenario lands on its agreed final severity."""
    assert events_for(scenario)[-1].risk_level is EXPECTED_FINAL_LEVELS[scenario]


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


def test_genuine_call_stays_low_and_needs_no_action() -> None:
    """A genuine caller never leaves LOW and never triggers an action."""
    events = events_for(ScenarioId.GENUINE)
    assert all(event.risk_level is RiskLevel.LOW for event in events)
    assert all(event.recommended_action is RecommendedAction.NONE for event in events)
    assert all(event.synthetic_probability < 0.3 for event in events)
    assert all(event.speaker_match_score > 0.7 for event in events)


def test_human_impostor_is_identity_driven_not_synthetic() -> None:
    """Human impostor keeps synthesis low while mismatch climbs."""
    events = events_for(ScenarioId.HUMAN_IMPOSTOR)
    assert all(event.synthetic_probability < 0.3 for event in events)
    assert events[-1].speaker_mismatch_score > events[0].speaker_mismatch_score
    assert events[-1].speaker_mismatch_score > 0.7
    assert MISMATCH_REASON in events[-1].reasons


def test_ai_clone_escalates_to_secondary_verification() -> None:
    """AI clone ends CRITICAL asking for secondary verification."""
    final = events_for(ScenarioId.AI_CLONE)[-1]
    assert final.risk_level is RiskLevel.CRITICAL
    assert final.synthetic_probability > 0.8
    assert final.prosody_anomaly_score > 0.7
    assert final.speaker_mismatch_score > 0.5
    assert final.recommended_action in SECONDARY_VERIFICATION_ACTIONS


def test_transfer_attack_final_event_blocks_with_all_three_reasons() -> None:
    """The primary demo ends CRITICAL, BLOCK_ACTION, with the agreed reasons."""
    final = events_for(ScenarioId.HIGH_VALUE_TRANSFER_ATTACK)[-1]
    assert final.overall_risk_score == 92
    assert final.risk_level is RiskLevel.CRITICAL
    assert final.recommended_action is RecommendedAction.BLOCK_ACTION
    assert SYNTHETIC_REASON in final.reasons
    assert MISMATCH_REASON in final.reasons
    assert CONTEXT_REASON in final.reasons


def test_stream_is_deterministic() -> None:
    """Replaying a scenario twice produces identical events."""
    first = [event.model_dump() for event in events_for(ScenarioId.HIGH_VALUE_TRANSFER_ATTACK)]
    second = [event.model_dump() for event in events_for(ScenarioId.HIGH_VALUE_TRANSFER_ATTACK)]
    assert first == second
