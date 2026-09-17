"""Tests for the conservative two-state ML operational policy."""

from __future__ import annotations

from app.models.risk import RecommendedAction, RiskLevel
from app.services.ml_risk_policy import MLRiskPolicy


def test_initial_policy_is_normal() -> None:
    """The initial ML decision monitors with an operational score of 20."""
    decision = MLRiskPolicy().initial()
    assert (decision.score, decision.level, decision.action) == (20, RiskLevel.LOW, RecommendedAction.MONITOR)


def test_persisted_spoof_state_enters_review() -> None:
    """The persisted elevated state maps to review, never automatic blocking."""
    decision = MLRiskPolicy().evaluate("ELEVATED_AUTHENTICITY_REVIEW")
    assert (decision.score, decision.level, decision.action) == (70, RiskLevel.HIGH, RecommendedAction.REQUIRE_CALLBACK)
    assert decision.action not in {RecommendedAction.REQUIRE_SUPERVISOR, RecommendedAction.BLOCK_ACTION}


def test_call_metadata_cannot_change_ml_policy() -> None:
    """Call context is not a JASH-004 input to operational policy."""
    policy = MLRiskPolicy()
    assert policy.evaluate("NORMAL", call_context={"transaction_value": 2_500_000}) == policy.evaluate("NORMAL")
