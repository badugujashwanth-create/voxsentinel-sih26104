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

    @classmethod
    def evaluated(cls, similarity: float, threshold: float) -> "SpeakerEvidence":
        """Creates a finite thresholded speaker observation."""
        if not 0.0 <= similarity <= 1.0 or not 0.0 <= threshold <= 1.0:
            raise ValueError("speaker similarity and threshold must be between 0 and 1")
        state = SpeakerEvidenceState.CONSISTENT if similarity >= threshold else SpeakerEvidenceState.INCONSISTENT
        return cls("EVALUATED", similarity, threshold, state)

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
    """Requires two consecutive corroborating observations before promotion."""

    def __init__(self, history_size: int = 5, persistence: int = 2) -> None:
        """Creates a bounded fusion history."""
        if history_size < persistence or persistence < 1:
            raise ValueError("fusion history must contain the persistence window")
        self._history: deque[tuple[float, SpeakerEvidence]] = deque(maxlen=history_size)
        self._persistence = persistence

    def add(self, spoof_score: float, speaker: SpeakerEvidence) -> FusionDecision:
        """Adds one observation and applies deterministic persistence."""
        self._history.append((spoof_score, speaker))
        recent = list(self._history)[-self._persistence :]
        spoof_persistent = len(recent) == self._persistence and all(score >= 0.5 for score, _ in recent)
        mismatch_persistent = len(recent) == self._persistence and all(item.state is SpeakerEvidenceState.INCONSISTENT for _, item in recent)
        if not spoof_persistent and not mismatch_persistent:
            return FusionDecision("NORMAL", 20, "MONITOR", ("Temporal evidence is still forming",))
        return fuse_evidence(spoof_score, speaker)
