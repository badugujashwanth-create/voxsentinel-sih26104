"""Live risk event contract shared with the VoxSentinel console frontend."""

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
    VERIFY_IDENTITY = "VERIFY_IDENTITY"
    REQUIRE_OTP = "REQUIRE_OTP"
    REQUIRE_CALLBACK = "REQUIRE_CALLBACK"
    REQUIRE_VOICE_CHALLENGE = "REQUIRE_VOICE_CHALLENGE"
    HOLD_SENSITIVE_ACTION = "HOLD_SENSITIVE_ACTION"
    REQUIRE_SUPERVISOR = "REQUIRE_SUPERVISOR"
    BLOCK_ACTION = "BLOCK_ACTION"


class EvidenceAvailability(StrEnum):
    """Whether a detector dimension was evaluated for an event."""
    MEASURED = "MEASURED"
    EVALUATED = "EVALUATED"
    INSUFFICIENT_AUDIO = "INSUFFICIENT_AUDIO"
    NO_REFERENCE = "NO_REFERENCE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    NOT_EVALUATED = "NOT_EVALUATED"


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
    preprocessing_latency_ms: float | None = Field(default=None, ge=0.0)
    synthetic_score_semantics: Literal["uncalibrated"] | None = None
    speaker_score_semantics: Literal["uncalibrated_similarity"] | None = None
    speaker_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    speaker_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    speaker_state: Literal["CONSISTENT", "INCONSISTENT", "INDETERMINATE"] | None = None
    speaker_model_id: str | None = None
    expected_speaker_id: str | None = None
    speaker_profile_id: str | None = None
    fusion_state: str | None = None
    evidence_availability: dict[str, EvidenceAvailability] | None = None
    audio_source_frame_start: int | None = Field(default=None, ge=0)
    audio_source_frame_end: int | None = Field(default=None, ge=0)
    audio_source_transport_sequence_start: int | None = Field(default=None, ge=1)
    audio_source_transport_sequence_end: int | None = Field(default=None, ge=1)
    audio_source_gap_count: int | None = Field(default=None, ge=0)
    audio_window_sequence: int | None = Field(default=None, ge=1)
    aggregate_spoof_evidence: Probability | None = None

    @model_validator(mode="after")
    def _check_frontend_invariants(self) -> "LiveRiskEvent":
        """Rejects events whose score and severity disagree."""
        expected = risk_level_for(self.overall_risk_score)
        if self.risk_level is not expected:
            raise ValueError(f"risk_level {self.risk_level} does not match score {self.overall_risk_score} (expected {expected})")
        if any(not reason.strip() for reason in self.reasons):
            raise ValueError("reasons must not contain empty strings")
        return self


def risk_level_for(score: int) -> RiskLevel:
    """Maps an overall score to its severity band."""
    if score < 30:
        return RiskLevel.LOW
    if score < 60:
        return RiskLevel.MEDIUM
    if score < 80:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL
