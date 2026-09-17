"""Backend representation of raw, uncalibrated ML evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RawSpoofEvidence(BaseModel):
    """Evidence returned by AASIST before backend aggregation or policy."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    raw_spoof_score: float = Field(ge=0.0, le=1.0)
    score_semantics: Literal["uncalibrated"] = "uncalibrated"
    threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    prediction: Literal["bonafide", "spoof"] = "bonafide"
    audio_duration_ms: float = Field(ge=0.0, default=0.0)
    inference_ms: float = Field(ge=0.0)
    preprocessing_ms: float = Field(ge=0.0, default=0.0)
    warnings: list[str] = Field(default_factory=list)
