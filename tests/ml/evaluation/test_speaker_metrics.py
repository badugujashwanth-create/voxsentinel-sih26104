"""Speaker-verification evaluation metric tests.

The committed numbers in ml/evaluation/sv_results.json are only trustworthy if
the arithmetic that produced them is correct and if the file can be rebuilt from
the committed predictions. These check the confusion matrix, FAR, FRR, EER, and
the threshold rule against cases whose answers are known by hand, with no model
and no audio involved.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS = REPO_ROOT / "ml" / "evaluation" / "sv_predictions.csv"
RESULTS = REPO_ROOT / "ml" / "evaluation" / "sv_results.json"
MANIFEST = REPO_ROOT / "ml" / "evaluation" / "sv_manifest.json"

from ml.scripts.evaluate_speaker import (  # noqa: E402
    ATTACK_RELATIONS,
    HUMAN_RELATIONS,
    Trial,
    build_results,
    choose_threshold,
    confusion,
    equal_error_rate,
    read_predictions,
)


def trial(relation: str, similarity: float, corpus: str = "librispeech", name: str = "t") -> Trial:
    """Builds one trial with everything except the fields under test defaulted."""
    return Trial(
        trial_id=f"{name}_{relation}_{similarity}",
        reference_speaker="spk",
        probe_sample="probe",
        relation=relation,
        corpus=corpus,
        similarity=similarity,
    )


def target(similarity: float, corpus: str = "librispeech") -> Trial:
    """A same-speaker trial."""
    return trial("same_speaker", similarity, corpus)


def nontarget(similarity: float, corpus: str = "librispeech") -> Trial:
    """A different-speaker trial."""
    return trial("different_speaker", similarity, corpus)


# --- Confusion matrix and rates ----------------------------------------------


def test_confusion_counts_each_quadrant() -> None:
    matrix = confusion([target(0.9), target(0.1), nontarget(0.8), nontarget(0.2), nontarget(0.0)], threshold=0.5)
    assert (matrix.true_positive, matrix.false_negative) == (1, 1)
    assert (matrix.false_positive, matrix.true_negative) == (1, 2)


def test_threshold_is_inclusive_at_the_boundary() -> None:
    # A score exactly on the cut point is a match, matching the result contract.
    assert confusion([target(0.5)], threshold=0.5).true_positive == 1
    assert confusion([nontarget(0.5)], threshold=0.5).false_positive == 1


def test_far_and_frr_are_computed_over_the_right_denominators() -> None:
    # 4 genuine, 1 rejected -> FRR 0.25. 5 impostors, 2 accepted -> FAR 0.4.
    trials = [target(0.9), target(0.8), target(0.7), target(0.1)]
    trials += [nontarget(0.9), nontarget(0.6), nontarget(0.2), nontarget(0.1), nontarget(0.0)]
    matrix = confusion(trials, threshold=0.5)
    assert matrix.false_reject_rate == pytest.approx(0.25)
    assert matrix.false_accept_rate == pytest.approx(0.40)


def test_metrics_match_hand_computed_values() -> None:
    # TP=3 FP=2 FN=1 TN=4: precision 3/5, recall 3/4, f1 2*.6*.75/1.35.
    trials = [target(0.9), target(0.8), target(0.6), target(0.1)]
    trials += [nontarget(0.9), nontarget(0.7), nontarget(0.4), nontarget(0.3), nontarget(0.2), nontarget(0.1)]
    matrix = confusion(trials, threshold=0.5)
    assert (matrix.true_positive, matrix.false_positive, matrix.false_negative, matrix.true_negative) == (3, 2, 1, 4)
    assert matrix.accuracy == pytest.approx(7 / 10)
    assert matrix.precision == pytest.approx(0.6)
    assert matrix.recall == pytest.approx(0.75)
    assert matrix.f1 == pytest.approx(2 * 0.6 * 0.75 / 1.35)


def test_perfect_separation_gives_perfect_metrics() -> None:
    matrix = confusion([target(0.9), target(0.8), nontarget(0.1), nontarget(0.2)], threshold=0.5)
    assert matrix.accuracy == matrix.precision == matrix.recall == matrix.f1 == 1.0
    assert matrix.false_accept_rate == matrix.false_reject_rate == 0.0


def test_empty_quadrants_do_not_divide_by_zero() -> None:
    matrix = confusion([], threshold=0.5)
    assert matrix.accuracy == matrix.precision == matrix.recall == matrix.f1 == 0.0
    assert matrix.false_accept_rate == matrix.false_reject_rate == 0.0


# --- EER ---------------------------------------------------------------------


def test_eer_is_half_when_classes_are_indistinguishable() -> None:
    scores = [0.1, 0.2, 0.3, 0.4]
    eer, _ = equal_error_rate(scores, list(scores))
    assert eer == pytest.approx(0.5)


def test_eer_is_zero_when_classes_separate() -> None:
    eer, threshold = equal_error_rate([0.8, 0.9], [0.1, 0.2])
    assert eer == pytest.approx(0.0)
    assert 0.2 < threshold <= 0.8


def test_eer_finds_the_balancing_threshold() -> None:
    # One genuine below 0.5 (FRR 1/4) and one impostor at or above it (FAR 1/4).
    eer, threshold = equal_error_rate([0.4, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.5])
    assert eer == pytest.approx(0.25)
    assert threshold == pytest.approx(0.5)


def test_eer_is_undefined_without_both_classes() -> None:
    for eer, threshold in (equal_error_rate([0.5], []), equal_error_rate([], [0.5])):
        assert math.isnan(eer) and math.isnan(threshold)


# --- Threshold rule -----------------------------------------------------------


def test_threshold_uses_the_eer_point_when_the_classes_overlap() -> None:
    trials = [target(0.4), target(0.6), target(0.7), target(0.8)]
    trials += [nontarget(0.1), nontarget(0.2), nontarget(0.3), nontarget(0.5)]
    threshold, note = choose_threshold(trials)
    assert threshold == pytest.approx(0.5)
    assert note["rule"].startswith("equal error rate")
    assert note["eer"] == pytest.approx(0.25)


def test_threshold_uses_the_margin_midpoint_when_the_classes_separate() -> None:
    # EER is zero here, so the EER "threshold" is not unique: every cut point in
    # (0.30, 0.80] scores identically. Sitting on either edge is one awkward
    # trial away from an error, so the midpoint is used instead.
    trials = [target(0.8), target(0.9), nontarget(0.1), nontarget(0.3)]
    threshold, note = choose_threshold(trials)
    assert note["eer"] == pytest.approx(0.0)
    assert note["separating_interval"] == [0.3, 0.8]
    assert threshold == pytest.approx(0.55)
    assert "not unique" in note["rule"]


def test_threshold_is_fitted_on_one_corpus_only() -> None:
    # The held-out corpus must not move the number, or the clone result would be
    # measured against a cut point fitted to the same audio it came from.
    fitted = [target(0.8), target(0.9), nontarget(0.1), nontarget(0.3)]
    held_out = [target(0.2, "cmu_arctic"), nontarget(0.99, "cmu_arctic")]
    attacks = [trial("clone", 0.95, "cmu_arctic"), trial("synthetic_non_target", 0.05, "cmu_arctic")]
    assert choose_threshold(fitted + held_out + attacks)[0] == choose_threshold(fitted)[0]


# --- Attack trials stay out of the error rates --------------------------------


def test_attacks_are_reported_but_never_folded_into_far() -> None:
    # FAR is defined over human impostors. A clone accepted at the threshold
    # must show up in the attack block and nowhere in the confusion matrix,
    # because counting it would quietly change what FAR means.
    human = [target(0.9), nontarget(0.1)]
    attacks = [trial("clone", 0.95, "cmu_arctic"), trial("synthetic_non_target", 0.02, "cmu_arctic")]
    results = build_results(human + attacks, 0.5, {"rule": "test"}, "test-model", {})

    assert results["overall"]["positive_pairs"] == 1
    assert results["overall"]["negative_pairs"] == 1
    assert results["overall"]["metrics"]["false_accept_rate"] == 0.0
    assert results["attacks"]["clone"]["accepted_at_threshold"] == 1
    assert results["attacks"]["clone"]["acceptance_rate"] == pytest.approx(1.0)
    assert results["attacks"]["synthetic_non_target"]["accepted_at_threshold"] == 0


def test_human_and_attack_relations_do_not_overlap() -> None:
    assert not set(HUMAN_RELATIONS) & set(ATTACK_RELATIONS)


# --- The committed artefacts --------------------------------------------------


def committed(path: Path) -> dict | list:
    """Loads a committed artefact, or skips when it is absent."""
    if not path.is_file():
        pytest.skip(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_results_are_reproducible_from_committed_predictions() -> None:
    # This is what makes the published numbers auditable: nothing is hardcoded,
    # so the whole summary re-derives from the per-trial evidence alone.
    if not PREDICTIONS.is_file():
        pytest.skip(f"{PREDICTIONS} is missing")
    published = committed(RESULTS)
    trials = read_predictions(PREDICTIONS)

    threshold, _ = choose_threshold(trials)
    assert threshold == published["threshold"]

    rebuilt = build_results(trials, threshold, published["threshold_note"], published["model"], published["latency"])
    for section in ("counts", "overall", "by_corpus", "attacks"):
        assert rebuilt[section] == published[section], f"{section} does not re-derive from sv_predictions.csv"


def test_committed_predictions_cover_every_manifest_trial() -> None:
    if not PREDICTIONS.is_file():
        pytest.skip(f"{PREDICTIONS} is missing")
    manifest = committed(MANIFEST)
    with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["trial_id"] for row in rows} == {t["trial_id"] for t in manifest["trials"]}
    assert len(rows) == manifest["counts"]["trials"]


def test_every_trial_similarity_is_a_cosine() -> None:
    if not PREDICTIONS.is_file():
        pytest.skip(f"{PREDICTIONS} is missing")
    for row in read_predictions(PREDICTIONS):
        assert -1.0 <= row.similarity <= 1.0, row.trial_id


def test_manifest_records_enough_to_rebuild_every_sample() -> None:
    manifest = committed(MANIFEST)
    for sample in manifest["samples"]:
        assert {"sample_id", "source_type", "filename", "sha256"} <= set(sample)
        if sample["source_type"] == "piper_tts":
            # Regenerating a clone needs the voice, the speaker, the text, the
            # synthesis parameters, and something to verify the result against.
            assert {"voice_sha256", "voice_config_sha256", "voice_speaker_index", "text", "synthesis", "sample_count", "pcm_envelope_db"} <= set(sample)
        else:
            assert "source_relpath" in sample and "license" in sample


def test_manifest_enrolment_and_probes_never_share_a_recording_session() -> None:
    # Two clips from one chapter share microphone and room, so a verifier can
    # score them alike without having learned anything about the voice.
    manifest = committed(MANIFEST)
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    for speaker in manifest["speakers"]:
        if speaker["corpus"] != "LibriSpeech dev-clean":
            continue
        enrolment = {samples[s]["chapter_id"] for s in speaker["enrolment"]}
        probes = {samples[s]["chapter_id"] for s in speaker["probes"]}
        assert not enrolment & probes, speaker["speaker_key"]


def test_clone_trials_point_at_the_speaker_they_clone() -> None:
    manifest = committed(MANIFEST)
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    clone_trials = [t for t in manifest["trials"] if t["relation"] == "clone"]
    assert clone_trials
    for entry in clone_trials:
        assert samples[entry["probe_sample"]]["clone_of"] == entry["reference_speaker"]
    for entry in (t for t in manifest["trials"] if t["relation"] == "synthetic_non_target"):
        assert samples[entry["probe_sample"]]["clone_of"] != entry["reference_speaker"]


def test_published_caveats_admit_the_set_is_small() -> None:
    results = committed(RESULTS)
    assert results["underpowered"] is True
    assert any("NOT production accuracy" in caveat for caveat in results["caveats"])
    assert "UNCALIBRATED" in results["score_note"]
