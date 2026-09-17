"""Live risk event contract shared with the VoxSentinel console frontend.

Field names, value ranges, and the severity banding mirror
``frontend/src/domain/risk.ts``. That module rejects any event whose
``risk_level`` disagrees with ``overall_risk_score``, so ``risk_level_for`` must
stay identical to its ``getRiskLevel``. Changing either side alone breaks the
console at runtime.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class RiskLevel(StrEnum):
    """Severity bands the console renders."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendedAction(StrEnum):
    """Operator-facing actions the console can surface."""

    NONE = "NONE"
    MONITOR = "MONITOR"
    REQUIRE_OTP = "REQUIRE_OTP"
    REQUIRE_CALLBACK = "REQUIRE_CALLBACK"
    REQUIRE_VOICE_CHALLENGE = "REQUIRE_VOICE_CHALLENGE"
    REQUIRE_SUPERVISOR = "REQUIRE_SUPERVISOR"
    BLOCK_ACTION = "BLOCK_ACTION"


class EvidenceAvailability(StrEnum):
    """Whether a detector dimension was evaluated for an event."""

    MEASURED = "MEASURED"
    NOT_EVALUATED = "NOT_EVALUATED"


def risk_level_for(score: int) -> RiskLevel:
    """Maps an overall 0-100 score to its severity band."""
    if score < 30:
        return RiskLevel.LOW
    if score < 60:
        return RiskLevel.MEDIUM
    if score < 80:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class LiveRiskEvent(BaseModel):
    """One risk observation emitted on the live stream."""

    call_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    timestamp_ms: int = Field(ge=0)
    synthetic_probability: Probability
    speaker_match_score: Probability
    speaker_mismatch_score: Probability
    prosody_anomaly_score: Probability
    replay_risk_score: Probability
    context_risk_score: Probability
    overall_risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    reasons: list[str]
    recommended_action: RecommendedAction
    inference_latency_ms: float | None = Field(default=None, ge=0.0)
    provider_round_trip_ms: float | None = Field(default=None, ge=0.0)
    synthetic_score_semantics: Literal["uncalibrated"] | None = None
    evidence_availability: dict[str, EvidenceAvailability] | None = None

    @model_validator(mode="after")
    def _check_frontend_invariants(self) -> LiveRiskEvent:
        """Rejects events the console's validator would throw away."""
        expected = risk_level_for(self.overall_risk_score)
        if self.risk_level is not expected:
            raise ValueError(f"risk_level {self.risk_level} does not match score {self.overall_risk_score} (expected {expected})")
        if any(not reason.strip() for reason in self.reasons):
            raise ValueError("reasons must not contain empty strings")
        return self
