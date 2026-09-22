from app.services.speaker_fusion import SpeakerEvidence, SpeakerEvidenceState, TemporalFusionAccumulator, fuse_evidence


def test_low_spoof_and_consistent_speaker_is_monitor():
    """Low spoof and consistent speaker remains monitoring."""
    result = fuse_evidence(0.1, SpeakerEvidence.evaluated(0.8, 0.55))
    assert result.state == "NORMAL"
    assert result.score == 20
    assert result.action == "MONITOR"


def test_low_spoof_and_mismatched_speaker_requires_identity_review():
    """Persistent single-dimension mismatch maps to identity review."""
    result = fuse_evidence(0.1, SpeakerEvidence.evaluated(0.2, 0.55))
    assert result.state == "IDENTITY_REVIEW"
    assert result.score == 50
    assert result.action == "VERIFY_IDENTITY"


def test_high_spoof_and_consistent_speaker_requires_authenticity_review():
    """Corroborated spoof evidence maps to authenticity review."""
    result = fuse_evidence(0.8, SpeakerEvidence.evaluated(0.8, 0.55))
    assert result.state == "AUTHENTICITY_REVIEW"
    assert result.score == 70
    assert result.action == "REQUIRE_CALLBACK"


def test_missing_speaker_is_not_treated_as_safe():
    """Missing reference is explicit indeterminate evidence."""
    result = fuse_evidence(0.1, SpeakerEvidence.no_reference())
    assert result.state == "INDETERMINATE"
    assert result.action == "VERIFY_IDENTITY"


def test_speaker_similarity_is_uncalibrated():
    """Speaker similarity carries no probability semantics."""
    evidence = SpeakerEvidence.evaluated(0.91, 0.55)
    assert evidence.score_semantics == "uncalibrated"

def test_negative_speaker_similarity_is_a_valid_mismatch():
    """Negative cosine similarity remains evaluated mismatch evidence."""
    evidence = SpeakerEvidence.evaluated(-0.2, 0.55)
    assert evidence.state is SpeakerEvidenceState.INCONSISTENT


def test_one_anomalous_observation_does_not_promote_temporal_fusion():
    """A single high observation remains bounded until persistence is met."""
    accumulator = TemporalFusionAccumulator()
    first = accumulator.add(0.8, SpeakerEvidence.evaluated(0.8, 0.55))
    assert first.score == 20
    assert first.action == "MONITOR"
    second = accumulator.add(0.8, SpeakerEvidence.evaluated(0.8, 0.55))
    assert second.score == 70
    assert second.action == "REQUIRE_CALLBACK"


def test_mismatch_promotion_requires_two_consecutive_observations():
    """Speaker mismatch promotion has deterministic persistence."""
    accumulator = TemporalFusionAccumulator()
    first = accumulator.add(0.1, SpeakerEvidence.evaluated(0.2, 0.55))
    assert first.state == "NORMAL"
    second = accumulator.add(0.1, SpeakerEvidence.evaluated(0.2, 0.55))
    assert second.state == "IDENTITY_REVIEW"
