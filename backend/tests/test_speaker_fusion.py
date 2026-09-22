from app.services.speaker_fusion import SpeakerEvidence, SpeakerEvidenceState, fuse_evidence


def test_low_spoof_and_consistent_speaker_is_monitor():
    result = fuse_evidence(0.1, SpeakerEvidence.evaluated(0.8, 0.55))
    assert result.state == "NORMAL"
    assert result.score == 20
    assert result.action == "MONITOR"


def test_low_spoof_and_mismatched_speaker_requires_identity_review():
    result = fuse_evidence(0.1, SpeakerEvidence.evaluated(0.2, 0.55))
    assert result.state == "IDENTITY_REVIEW"
    assert result.score == 50
    assert result.action == "VERIFY_IDENTITY"


def test_high_spoof_and_consistent_speaker_requires_authenticity_review():
    result = fuse_evidence(0.8, SpeakerEvidence.evaluated(0.8, 0.55))
    assert result.state == "AUTHENTICITY_REVIEW"
    assert result.score == 70
    assert result.action == "REQUIRE_CALLBACK"


def test_missing_speaker_is_not_treated_as_safe():
    result = fuse_evidence(0.1, SpeakerEvidence.no_reference())
    assert result.state == "INDETERMINATE"
    assert result.action == "VERIFY_IDENTITY"


def test_speaker_similarity_is_uncalibrated():
    evidence = SpeakerEvidence.evaluated(0.91, 0.55)
    assert evidence.score_semantics == "uncalibrated"
