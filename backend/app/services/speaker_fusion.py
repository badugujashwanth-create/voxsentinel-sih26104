"""Deterministic fusion of independent spoof and speaker evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum


class SpeakerEvidenceState(StrEnum):
    """Semantic result of comparing a probe with the enrolled reference."""
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class SpeakerEvidence:
    """One uncalibrated speaker similarity observation."""
    availability: str
    similarity: float | None
    threshold: float | None
    state: SpeakerEvidenceState
    score_semantics: str = "uncalibrated"
    probe_metadata: object | None = None

    @classmethod
    def evaluated(cls, similarity: float, threshold: float, probe_metadata: object | None = None) -> "SpeakerEvidence":
        """Creates a finite thresholded speaker observation."""
        if not -1.0 <= similarity <= 1.0 or not 0.0 <= threshold <= 1.0:
            raise ValueError("speaker similarity must be between -1 and 1; threshold must be between 0 and 1")
        state = SpeakerEvidenceState.CONSISTENT if similarity >= threshold else SpeakerEvidenceState.INCONSISTENT
        return cls("EVALUATED", similarity, threshold, state, probe_metadata=probe_metadata)

    @classmethod
    def no_reference(cls) -> "SpeakerEvidence":
        """Creates an explicit no-reference observation."""
        return cls("NO_REFERENCE", None, None, SpeakerEvidenceState.INDETERMINATE)


@dataclass(frozen=True)
class FusionDecision:
    """Policy-neutral fused operational decision."""
    state: str
    score: int
    action: str
    reasons: tuple[str, ...]


def fuse_evidence(spoof_score: float | None, speaker: SpeakerEvidence) -> FusionDecision:
    """Maps separate evidence dimensions to a bounded deterministic decision."""
    spoof_elevated = spoof_score is not None and spoof_score >= 0.5
    if speaker.state is SpeakerEvidenceState.INDETERMINATE:
        return FusionDecision("INDETERMINATE", 20 if not spoof_elevated else 70, "VERIFY_IDENTITY", ("Speaker reference is unavailable",))
    if spoof_elevated and speaker.state is SpeakerEvidenceState.INCONSISTENT:
        return FusionDecision("HIGH_RISK_REVIEW", 70, "HOLD_SENSITIVE_ACTION", ("Synthetic evidence and speaker mismatch require review",))
    if spoof_elevated:
        return FusionDecision("AUTHENTICITY_REVIEW", 70, "REQUIRE_CALLBACK", ("Synthetic speech characteristics require review",))
    if speaker.state is SpeakerEvidenceState.INCONSISTENT:
        return FusionDecision("IDENTITY_REVIEW", 50, "VERIFY_IDENTITY", ("Speaker identity mismatch requires verification",))
    return FusionDecision("NORMAL", 20, "MONITOR", ("Voice and speaker evidence remain within the enrolled baseline",))


class TemporalFusionAccumulator:
    """Requires two consecutive corroborating observations before promotion or demotion."""

    _STATE_RANK = {
        "NORMAL": 0,
        "IDENTITY_REVIEW": 1,
        "AUTHENTICITY_REVIEW": 2,
        "HIGH_RISK_REVIEW": 3,
        "INDETERMINATE": 0,
    }

    def __init__(self, history_size: int = 5, persistence: int = 2) -> None:
        """Creates a bounded fusion history."""
        if history_size < persistence or persistence < 1:
            raise ValueError("fusion history must contain the persistence window")
        self._history: deque[tuple[float, SpeakerEvidence]] = deque(maxlen=history_size)
        self._persistence = persistence
        self._current_decision = FusionDecision("NORMAL", 20, "MONITOR", ("Temporal evidence is still forming",))
        self._pending_demotion_state: str | None = None
        self._demotion_streak = 0

    def reset(self) -> None:
        """Clears all observations and temporal state for a new call."""
        self._history.clear()
        self._current_decision = FusionDecision("NORMAL", 20, "MONITOR", ("Temporal evidence is still forming",))
        self._pending_demotion_state = None
        self._demotion_streak = 0

    def add(self, spoof_score: float, speaker: SpeakerEvidence) -> FusionDecision:
        """Adds one observation and applies symmetric deterministic persistence."""
        self._history.append((spoof_score, speaker))
        recent = list(self._history)[-self._persistence :]
        spoof_persistent = len(recent) == self._persistence and all(score >= 0.5 for score, _ in recent)
        mismatch_persistent = len(recent) == self._persistence and all(item.state is SpeakerEvidenceState.INCONSISTENT for _, item in recent)
        persistent_candidate = (
            FusionDecision("NORMAL", 20, "MONITOR", ("Temporal evidence is still forming",))
            if not spoof_persistent and not mismatch_persistent
            else fuse_evidence(spoof_score, speaker)
        )
        raw_candidate = fuse_evidence(spoof_score, speaker)
        current_rank = self._STATE_RANK[self._current_decision.state]
        candidate = persistent_candidate if current_rank == 0 else raw_candidate
        candidate_rank = self._STATE_RANK[candidate.state]
        if current_rank == 0:
            if candidate_rank > 0:
                self._current_decision = candidate
            return self._current_decision
        if candidate.state == "INDETERMINATE":
            self._pending_demotion_state = None
            self._demotion_streak = 0
            return self._current_decision
        if candidate_rank >= current_rank:
            self._pending_demotion_state = None
            self._demotion_streak = 0
            self._current_decision = candidate
            return self._current_decision
        if candidate.state != self._pending_demotion_state:
            self._pending_demotion_state = candidate.state
            self._demotion_streak = 1
        else:
            self._demotion_streak += 1
        if self._demotion_streak >= self._persistence:
            self._current_decision = candidate
            self._pending_demotion_state = None
            self._demotion_streak = 0
        return self._current_decision