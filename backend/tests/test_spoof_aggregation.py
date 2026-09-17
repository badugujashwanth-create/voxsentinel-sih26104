"""Tests for temporal spoof evidence aggregation."""

from __future__ import annotations

from app.models.ml_evidence import RawSpoofEvidence
from app.services.spoof_aggregation import TemporalSpoofAggregator


def evidence(score: float) -> RawSpoofEvidence:
    """Builds deterministic raw evidence for an aggregation observation."""
    return RawSpoofEvidence(model_id="AASIST/test", raw_spoof_score=score, inference_ms=10.0)


def test_aggregator_uses_recent_five_median() -> None:
    """History is bounded and the aggregate is the recent median."""
    aggregator = TemporalSpoofAggregator()
    for score in (0.1, 0.2, 0.3, 0.4, 0.9, 0.8):
        aggregate = aggregator.add(evidence(score))
    assert aggregate.history == (0.2, 0.3, 0.4, 0.9, 0.8)
    assert aggregate.aggregate_score == 0.4
    assert aggregate.windows_observed == 6


def test_two_high_observations_promote_and_two_low_demote() -> None:
    """Promotion and demotion require persistence across observations."""
    aggregator = TemporalSpoofAggregator()
    assert aggregator.add(evidence(0.7)).threshold_state == "NORMAL"
    assert aggregator.add(evidence(0.7)).threshold_state == "ELEVATED_AUTHENTICITY_REVIEW"
    assert aggregator.add(evidence(0.1)).threshold_state == "ELEVATED_AUTHENTICITY_REVIEW"
    assert aggregator.add(evidence(0.1)).threshold_state == "ELEVATED_AUTHENTICITY_REVIEW"
    assert aggregator.add(evidence(0.1)).threshold_state == "NORMAL"


def test_reset_clears_history_and_state() -> None:
    """Reset returns aggregation to an unobserved normal state."""
    aggregator = TemporalSpoofAggregator()
    aggregator.add(evidence(0.8))
    aggregator.reset()
    aggregate = aggregator.add(evidence(0.1))
    assert aggregate.windows_observed == 1
    assert aggregate.threshold_state == "NORMAL"
