"""Deterministic fusion of independent spoof and speaker evidence."""

from __future__ import annotations

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
