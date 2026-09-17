"""Tests for the stateless ML service request and response contract."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from pydantic import ValidationError

from ml.runtime.contracts import SpoofInferenceRequest, SpoofInferenceResponse


def encoded_window() -> str:
    """Returns a base64-encoded exact-size float32 window."""
    return base64.b64encode(np.zeros(64_600, dtype="<f4").tobytes()).decode("ascii")


def test_valid_exact_window_request_is_accepted() -> None:
    """The contract accepts the backend's canonical window metadata."""
    request = SpoofInferenceRequest(samples_base64=encoded_window())
    assert request.sample_rate == 16_000
    assert request.sample_count == 64_600
    assert request.sample_format == "float32le"


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_rate": 8_000},
        {"channels": 2},
        {"sample_format": "int16le"},
        {"sample_count": 64_599},
    ],
)
def test_invalid_window_metadata_is_rejected(changes: dict[str, object]) -> None:
    """Wrong exact-window metadata cannot reach model inference."""
    payload = {"samples_base64": encoded_window(), **changes}
    with pytest.raises(ValidationError):
        SpoofInferenceRequest.model_validate(payload)


def test_response_preserves_uncalibrated_score_semantics() -> None:
    """Responses identify the raw model score as uncalibrated evidence."""
    response = SpoofInferenceResponse(model_id="test", raw_spoof_score=0.8, prediction="spoof", audio_duration_ms=4037.5, inference_ms=12.5)
    assert response.score_semantics == "uncalibrated"
