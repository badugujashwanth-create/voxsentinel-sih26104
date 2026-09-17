"""Backend-owned temporal aggregation for raw AASIST evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

from app.models.ml_evidence import RawSpoofEvidence

AGGREGATION_HISTORY_SIZE = 5
SPOOF_REVIEW_THRESHOLD = 0.5
SPOOF_CLEAR_THRESHOLD = 0.45
SPOOF_PERSISTENCE_OBSERVATIONS = 2


@dataclass(frozen=True)
class AggregatedSpoofEvidence:
    """Recent raw evidence and its persisted threshold state."""

    latest_raw_score: float
    aggregate_score: float
    windows_observed: int
    history: tuple[float, ...]
    threshold_state: str
    inference_ms: float


class TemporalSpoofAggregator:
    """Applies a bounded median and persistence/hysteresis state machine."""

    def __init__(self) -> None:
        """Creates an empty normal aggregator."""
        self._history: deque[float] = deque(maxlen=AGGREGATION_HISTORY_SIZE)
        self._windows_observed = 0
        self._threshold_state = "NORMAL"
        self._promotion_count = 0
        self._demotion_count = 0

    def add(self, evidence: RawSpoofEvidence) -> AggregatedSpoofEvidence:
        """Adds raw evidence and returns the updated aggregate state."""
        self._history.append(evidence.raw_spoof_score)
        self._windows_observed += 1
        aggregate_score = float(median(self._history))
        self._update_state(aggregate_score)
        return AggregatedSpoofEvidence(
            latest_raw_score=evidence.raw_spoof_score,
            aggregate_score=aggregate_score,
            windows_observed=self._windows_observed,
            history=tuple(self._history),
            threshold_state=self._threshold_state,
            inference_ms=evidence.inference_ms,
        )

    def reset(self) -> None:
        """Clears history and returns the state machine to normal."""
        self._history.clear()
        self._windows_observed = 0
        self._threshold_state = "NORMAL"
        self._promotion_count = 0
        self._demotion_count = 0

    def _update_state(self, aggregate_score: float) -> None:
        """Applies two-observation promotion and demotion hysteresis."""
        if self._threshold_state == "NORMAL":
            self._demotion_count = 0
            self._promotion_count = self._promotion_count + 1 if aggregate_score >= SPOOF_REVIEW_THRESHOLD else 0
            if self._promotion_count >= SPOOF_PERSISTENCE_OBSERVATIONS:
                self._threshold_state = "ELEVATED_AUTHENTICITY_REVIEW"
                self._promotion_count = 0
            return
        self._promotion_count = 0
        self._demotion_count = self._demotion_count + 1 if aggregate_score < SPOOF_CLEAR_THRESHOLD else 0
        if self._demotion_count >= SPOOF_PERSISTENCE_OBSERVATIONS:
            self._threshold_state = "NORMAL"
            self._demotion_count = 0
