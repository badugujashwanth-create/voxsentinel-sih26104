"""Tests for raw spoof evidence semantics."""

from __future__ import annotations

import pytest

from app.models.ml_evidence import RawSpoofEvidence


def test_raw_evidence_preserves_uncalibrated_semantics() -> None:
    """Evidence stores the model score without calling it a probability."""
    evidence = RawSpoofEvidence(model_id="AASIST/test", raw_spoof_score=0.8, inference_ms=10.0)
    assert evidence.score_semantics == "uncalibrated"


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_raw_score_must_be_in_model_range(score: float) -> None:
    """Remote model evidence outside the score range is rejected."""
    with pytest.raises(ValueError):
        RawSpoofEvidence(model_id="test", raw_spoof_score=score, inference_ms=1.0)
