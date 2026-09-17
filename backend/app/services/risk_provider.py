"""Risk scoring providers.

``RiskProvider`` is the seam the future ``MLRiskProvider`` plugs into. The API
and WebSocket layers only ever see this interface, so swapping the demo engine
for real inference does not touch routing code.

Everything ``MockRiskProvider`` returns is HAND-WRITTEN DEMO DATA. It performs
no audio analysis and no inference of any kind.

The scenario tables below mirror ``SCENARIOS`` in
``frontend/src/scenarios/scenarios.ts`` field for field, so the console renders
identically whether it runs its own offline fixtures or streams from this
backend. If the fixtures there change, change these to match.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.models.call import CallSession, ScenarioId
from app.models.risk import LiveRiskEvent, RecommendedAction, risk_level_for

#: Spacing between scenario events, mirroring ``EVENT_INTERVAL_MS`` in
#: ``frontend/src/scenarios/scenarios.ts``. This fixes ``timestamp_ms`` in the
#: fixture data and is deliberately independent of how fast the socket emits.
SCENARIO_EVENT_INTERVAL_MS = 700

BASELINE_REASON = "Voice signal remains within the enrolled speaker baseline"
SYNTHETIC_REASON = "Synthetic speech characteristics detected"
MISMATCH_REASON = "Speaker identity mismatch"
FINANCIAL_REQUEST_REASON = "High-value financial request"
PROSODY_REASON = "Prosody anomaly detected"
EMERGING_SYNTHESIS_REASON = "Synthesis artifact is emerging"
EMERGING_DRIFT_REASON = "Speaker identity drift is emerging"
INCREASING_MISMATCH_REASON = "Speaker identity mismatch is increasing"
SENSITIVE_REQUEST_REASON = "Sensitive request requires review"
MONITORED_FINANCIAL_REASON = "Sensitive financial request is being monitored"


@dataclass(frozen=True)
class ScenarioStep:
    """One scripted observation in a demo scenario."""

    overall_risk_score: int
    synthetic_probability: float
    speaker_match_score: float
    speaker_mismatch_score: float
    prosody_anomaly_score: float
    replay_risk_score: float
    context_risk_score: float
    recommended_action: RecommendedAction
    reasons: tuple[str, ...] = field(default=(BASELINE_REASON,))


def _calm(
    score: int,
    action: RecommendedAction = RecommendedAction.MONITOR,
    reasons: tuple[str, ...] = (BASELINE_REASON,),
    **overrides: float,
) -> ScenarioStep:
    """Builds a step from the quiet baseline profile with named overrides.

    Mirrors ``calmProfile`` and its spread overrides in the frontend fixtures.
    """
    baseline = {
        "synthetic_probability": 0.08,
        "speaker_match_score": 0.92,
        "speaker_mismatch_score": 0.08,
        "prosody_anomaly_score": 0.11,
        "replay_risk_score": 0.06,
        "context_risk_score": 0.12,
    }
    return ScenarioStep(overall_risk_score=score, recommended_action=action, reasons=reasons, **(baseline | overrides))


# Scenario A - genuine caller. Risk stays flat and low.
_GENUINE: tuple[ScenarioStep, ...] = (
    _calm(12),
    _calm(10),
    _calm(14),
    _calm(11),
    _calm(13),
)

# Scenario B - real human, wrong identity. Synthetic probability stays at the
# baseline throughout; the speaker mismatch is what drives the score up.
_HUMAN_IMPOSTOR: tuple[ScenarioStep, ...] = (
    _calm(18),
    _calm(26, reasons=(EMERGING_DRIFT_REASON,), speaker_match_score=0.74, speaker_mismatch_score=0.26),
    _calm(39, reasons=(INCREASING_MISMATCH_REASON,), speaker_match_score=0.61, speaker_mismatch_score=0.39, context_risk_score=0.40),
    _calm(54, reasons=(MISMATCH_REASON, SENSITIVE_REQUEST_REASON), speaker_match_score=0.46, speaker_mismatch_score=0.54, context_risk_score=0.62),
    _calm(67, reasons=(MISMATCH_REASON, FINANCIAL_REQUEST_REASON), speaker_match_score=0.33, speaker_mismatch_score=0.67, context_risk_score=0.76),
    _calm(76, reasons=(MISMATCH_REASON, FINANCIAL_REQUEST_REASON), speaker_match_score=0.24, speaker_mismatch_score=0.76, context_risk_score=0.82),
)

# Scenario C - AI voice clone. Synthetic probability and prosody anomaly climb
# together until the call is blocked.
_AI_CLONE: tuple[ScenarioStep, ...] = (
    _calm(20),
    _calm(31, reasons=(EMERGING_SYNTHESIS_REASON,), synthetic_probability=0.31, prosody_anomaly_score=0.38),
    _calm(46, reasons=(SYNTHETIC_REASON,), synthetic_probability=0.46, prosody_anomaly_score=0.55, speaker_match_score=0.68, speaker_mismatch_score=0.32),
    _calm(63, reasons=(SYNTHETIC_REASON, PROSODY_REASON), synthetic_probability=0.63, prosody_anomaly_score=0.67, speaker_match_score=0.48, speaker_mismatch_score=0.52),
    _calm(78, reasons=(SYNTHETIC_REASON, MISMATCH_REASON), synthetic_probability=0.78, prosody_anomaly_score=0.78, speaker_match_score=0.34, speaker_mismatch_score=0.66),
    _calm(88, RecommendedAction.BLOCK_ACTION, (SYNTHETIC_REASON, MISMATCH_REASON, PROSODY_REASON), synthetic_probability=0.88, prosody_anomaly_score=0.86, speaker_match_score=0.22, speaker_mismatch_score=0.78),
)

# Scenario D - primary SIH demo. Cloned executive voice authorising a large
# transfer; ends BLOCK_ACTION at CRITICAL.
_HIGH_VALUE_TRANSFER_ATTACK: tuple[ScenarioStep, ...] = (
    _calm(18, context_risk_score=0.18),
    _calm(27, reasons=(MONITORED_FINANCIAL_REASON,), context_risk_score=0.27),
    _calm(43, reasons=(EMERGING_SYNTHESIS_REASON, FINANCIAL_REQUEST_REASON), synthetic_probability=0.43, prosody_anomaly_score=0.36, context_risk_score=0.43),
    _calm(61, reasons=(SYNTHETIC_REASON, MISMATCH_REASON), synthetic_probability=0.61, speaker_match_score=0.48, speaker_mismatch_score=0.52, prosody_anomaly_score=0.58, context_risk_score=0.61),
    _calm(79, RecommendedAction.REQUIRE_CALLBACK, (SYNTHETIC_REASON, MISMATCH_REASON, FINANCIAL_REQUEST_REASON), synthetic_probability=0.79, speaker_match_score=0.31, speaker_mismatch_score=0.69, prosody_anomaly_score=0.72, context_risk_score=0.79),
    _calm(92, RecommendedAction.BLOCK_ACTION, (SYNTHETIC_REASON, MISMATCH_REASON, FINANCIAL_REQUEST_REASON), synthetic_probability=0.91, speaker_match_score=0.28, speaker_mismatch_score=0.72, prosody_anomaly_score=0.74, replay_risk_score=0.21, context_risk_score=0.95),
)

SCENARIO_STEPS: dict[ScenarioId, tuple[ScenarioStep, ...]] = {
    ScenarioId.GENUINE: _GENUINE,
    ScenarioId.HUMAN_IMPOSTOR: _HUMAN_IMPOSTOR,
    ScenarioId.AI_CLONE: _AI_CLONE,
    ScenarioId.HIGH_VALUE_TRANSFER_ATTACK: _HIGH_VALUE_TRANSFER_ATTACK,
}


class RiskProvider(ABC):
    """Produces the live risk events for one call session."""

    @abstractmethod
    def stream(self, session: CallSession, cancellation: asyncio.Event) -> AsyncIterator[LiveRiskEvent]:
        """Yields risk events asynchronously in ascending sequence order."""


class MockRiskProvider(RiskProvider):
    """Replays a scripted scenario. DEMO DATA ONLY - no audio, no inference."""

    def __init__(self, emit_interval_ms: int = SCENARIO_EVENT_INTERVAL_MS) -> None:
        """Configures deterministic pacing for the mock stream."""
        self.emit_interval_ms = emit_interval_ms

    async def stream(self, session: CallSession, cancellation: asyncio.Event) -> AsyncIterator[LiveRiskEvent]:
        """Yields the scripted events for the session's scenario."""
        for index, step in enumerate(SCENARIO_STEPS[session.scenario], start=1):
            if cancellation.is_set():
                return
            yield self._to_event(session.call_id, index, step)
            if index < len(SCENARIO_STEPS[session.scenario]) and self.emit_interval_ms:
                await asyncio.sleep(self.emit_interval_ms / 1000)

    @staticmethod
    def _to_event(call_id: str, sequence: int, step: ScenarioStep) -> LiveRiskEvent:
        """Renders one scripted step as a contract-shaped risk event."""
        return LiveRiskEvent(
            call_id=call_id,
            sequence=sequence,
            timestamp_ms=sequence * SCENARIO_EVENT_INTERVAL_MS,
            synthetic_probability=step.synthetic_probability,
            speaker_match_score=step.speaker_match_score,
            speaker_mismatch_score=step.speaker_mismatch_score,
            prosody_anomaly_score=step.prosody_anomaly_score,
            replay_risk_score=step.replay_risk_score,
            context_risk_score=step.context_risk_score,
            overall_risk_score=step.overall_risk_score,
            risk_level=risk_level_for(step.overall_risk_score),
            reasons=list(step.reasons),
            recommended_action=step.recommended_action,
        )
