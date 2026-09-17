"""Conservative operational mapping for persisted ML spoof evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from app.models.risk import RecommendedAction, RiskLevel

NORMAL_OPERATIONAL_SCORE = 20
ELEVATED_REVIEW_OPERATIONAL_SCORE = 70
NORMAL_STATE = "NORMAL"
ELEVATED_STATE = "ELEVATED_AUTHENTICITY_REVIEW"


@dataclass(frozen=True)
class MLPolicyDecision:
    """Operational state kept separate from uncalibrated model evidence."""

    state: str
    score: int
    level: RiskLevel
    action: RecommendedAction
    reasons: tuple[str, ...]


class MLRiskPolicy:
    """Maps only persisted spoof threshold state to two safe operational states."""

    def initial(self) -> MLPolicyDecision:
        """Returns the normal monitoring decision."""
        return self.evaluate(NORMAL_STATE)

    def evaluate(self, threshold_state: str, call_context: Mapping[str, object] | None = None) -> MLPolicyDecision:
        """Returns a decision without allowing call metadata to alter it."""
        del call_context
        if threshold_state == ELEVATED_STATE:
            return MLPolicyDecision(ELEVATED_STATE, ELEVATED_REVIEW_OPERATIONAL_SCORE, RiskLevel.HIGH, RecommendedAction.REQUIRE_CALLBACK, ("Persistent spoof evidence requires authenticity review",))
        if threshold_state != NORMAL_STATE:
            raise ValueError(f"unknown ML threshold state: {threshold_state}")
        return MLPolicyDecision(NORMAL_STATE, NORMAL_OPERATIONAL_SCORE, RiskLevel.LOW, RecommendedAction.MONITOR, ("Voice signal remains within the current monitoring baseline",))
