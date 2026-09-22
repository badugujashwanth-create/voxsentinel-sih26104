"""Call session lifecycle models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, Field


class CallStatus(StrEnum):
    """Lifecycle states a call session can occupy."""
    CREATED = "CREATED"
    LIVE = "LIVE"
    VERIFYING = "VERIFYING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ScenarioId(StrEnum):
    """Demo scenarios available from the mock risk provider."""
    GENUINE = "GENUINE"
    HUMAN_IMPOSTOR = "HUMAN_IMPOSTOR"
    AI_CLONE = "AI_CLONE"
    HIGH_VALUE_TRANSFER_ATTACK = "HIGH_VALUE_TRANSFER_ATTACK"


class CreateCallRequest(BaseModel):
    """Payload accepted when opening a call session."""
    claimed_identity: str = Field(min_length=1)
    scenario: ScenarioId
    transaction_value: float | None = Field(default=None, ge=0)
    currency: str | None = None
    speaker_profile_id: str | None = Field(default=None, min_length=1)


class CreateCallResponse(BaseModel):
    """Acknowledgement returned when a call session is opened."""
    call_id: str
    status: CallStatus
    claimed_identity: str
    scenario: ScenarioId
    speaker_profile_id: str | None = None


class CallSession(BaseModel):
    """Full current state of one call session."""
    call_id: str
    status: CallStatus
    claimed_identity: str
    scenario: ScenarioId
    transaction_value: float | None = None
    currency: str | None = None
    speaker_profile_id: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class CreateSpeakerProfileRequest(BaseModel):
    """Canonical enrollment audio sent ephemerally to the ML service."""
    expected_speaker_id: str = Field(min_length=1)
    samples_base64: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
