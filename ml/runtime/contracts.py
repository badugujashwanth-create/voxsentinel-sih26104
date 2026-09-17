"""Typed HTTP contract for the local ML inference service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AASIST_SAMPLE_RATE = 16_000
AASIST_SAMPLE_COUNT = 64_600
FLOAT32_LE_FORMAT = "float32le"


class SpoofInferenceRequest(BaseModel):
    """One exact canonical model window encoded for transport."""

    model_config = ConfigDict(extra="forbid")

    samples_base64: str = Field(min_length=1)
    sample_rate: Literal[16_000] = AASIST_SAMPLE_RATE
    channels: Literal[1] = 1
    sample_format: Literal["float32le"] = FLOAT32_LE_FORMAT
    sample_count: Literal[64_600] = AASIST_SAMPLE_COUNT


class SpoofInferenceResponse(BaseModel):
    """Raw AASIST evidence returned without call-policy semantics."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    raw_spoof_score: float = Field(ge=0.0, le=1.0)
    score_semantics: Literal["uncalibrated"] = "uncalibrated"
    threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    prediction: Literal["bonafide", "spoof"]
    audio_duration_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Model readiness response for the local runtime."""

    ready: bool
    model_id: str | None = None
    reason: str | None = None
