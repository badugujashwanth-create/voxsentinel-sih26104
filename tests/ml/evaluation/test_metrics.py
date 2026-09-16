"""Evaluation metric tests.

The committed numbers in ml/evaluation/ are only trustworthy if the arithmetic
that produced them is correct. These check the confusion matrix, the derived
metrics, and EER against cases whose answers are known by hand, with no model
and no audio involved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS = REPO_ROOT / "ml" / "evaluation" / "predictions.csv"
RESULTS = REPO_ROOT / "ml" / "evaluation" / "evaluation_results.json"
MANIFEST = REPO_ROOT / "ml" / "evaluation" / "eval_manifest.json"

import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from ml.scripts.evaluate import (  # noqa: E402
    Prediction,
    build_results,
    confusion,
    equal_error_rate,
    read_predictions,
)


def make(ground_truth: str, score: float, sample_id: str = "s") -> Prediction:
    """Builds one prediction row."""
    return Prediction(sample_id=sample_id, ground_truth=ground_truth, model_score=score, predicted_label="spoof" if score >= 0.5 else "bonafide")


def test_confusion_counts_each_quadrant() -> None:
    """Each of the four outcomes lands in the right cell."""
    rows = [
        make("spoof", 0.9),      # true positive
        make("spoof", 0.1),      # false negative
        make("bonafide", 0.8),   # false positive
        make("bonafide", 0.2),   # true negative
    ]
    metrics = confusion(rows, 0.5)
    assert (metrics.true_positive, metrics.false_negative, metrics.false_positive, metrics.true_negative) == (1, 1, 1, 1)


def test_threshold_is_inclusive_at_the_boundary() -> None:
    """A score exactly at the threshold counts as spoof."""
    assert confusion([make("spoof", 0.5)], 0.5).true_positive == 1
    assert confusion([make("spoof", 0.4999)], 0.5).false_negative == 1


def test_perfect_separation_gives_perfect_metrics() -> None:
    """Fully separated scores score 1.0 across the board."""
    rows = [make("spoof", 0.99) for _ in range(5)] + [make("bonafide", 0.01) for _ in range(5)]
    metrics = confusion(rows, 0.5)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert equal_error_rate(rows)[0] == 0.0


def test_metrics_match_hand_computed_values() -> None:
    """A worked example: TP=3 FN=1 FP=1 TN=5."""
    rows = [make("spoof", 0.9)] * 3 + [make("spoof", 0.1)] + [make("bonafide", 0.9)] + [make("bonafide", 0.1)] * 5
    metrics = confusion(rows, 0.5)
    assert (metrics.true_positive, metrics.false_negative, metrics.false_positive, metrics.true_negative) == (3, 1, 1, 5)
    assert metrics.accuracy == pytest.approx(8 / 10)
    assert metrics.precision == pytest.approx(3 / 4)
    assert metrics.recall == pytest.approx(3 / 4)
    assert metrics.f1 == pytest.approx(0.75)


def test_empty_quadrants_do_not_divide_by_zero() -> None:
    """No predicted positives means precision is 0, not a crash."""
    metrics = confusion([make("bonafide", 0.1)], 0.5)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_eer_is_half_when_classes_are_indistinguishable() -> None:
    """Identical score distributions cannot be separated at any threshold."""
    rows = [make("spoof", 0.5) for _ in range(10)] + [make("bonafide", 0.5) for _ in range(10)]
    eer, _ = equal_error_rate(rows)
    assert eer == pytest.approx(0.5)


def test_eer_finds_the_balancing_threshold() -> None:
    """With one overlap per class, EER is 10% on ten samples each."""
    spoof = [make("spoof", value) for value in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.2)]
    bonafide = [make("bonafide", value) for value in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.8)]
    eer, threshold = equal_error_rate(spoof + bonafide)
    assert eer == pytest.approx(0.1, abs=0.01)
    assert 0.2 <= threshold <= 0.9


def test_eer_is_undefined_without_both_classes() -> None:
    """A single-class set cannot produce an error rate."""
    import math

    eer, threshold = equal_error_rate([make("spoof", 0.9)])
    assert math.isnan(eer)
    assert math.isnan(threshold)


@pytest.mark.skipif(not PREDICTIONS.is_file(), reason="committed predictions not present")
def test_committed_results_are_reproducible_from_committed_predictions() -> None:
    """The committed summary is derivable from the committed rows.

    This is what makes the published numbers auditable: nothing is hardcoded,
    so recomputing from the CSV must land on the same summary.
    """
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    recomputed = build_results(read_predictions(PREDICTIONS), results["threshold"], results["model"])

    assert recomputed["n"] == results["n"]
    assert recomputed["confusion_matrix"] == results["confusion_matrix"]
    for name, value in results["metrics"].items():
        assert recomputed["metrics"][name] == pytest.approx(value), name


@pytest.mark.skipif(not PREDICTIONS.is_file(), reason="committed predictions not present")
def test_committed_predictions_cover_every_manifest_sample() -> None:
    """Every sample in the manifest has exactly one prediction row."""
    manifest_ids = [s["sample_id"] for s in json.loads(MANIFEST.read_text(encoding="utf-8"))["samples"]]
    with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(manifest_ids)
    assert sorted(r["sample_id"] for r in rows) == sorted(manifest_ids)
    assert all(0.0 <= float(r["model_score"]) <= 1.0 for r in rows)
    assert all(r["ground_truth"] in {"bonafide", "spoof"} for r in rows)


@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest not present")
def test_manifest_records_enough_to_reconstruct_every_sample() -> None:
    """Each entry carries the provenance needed to rebuild it."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    assert manifest["counts"]["total"] == len(samples) == 80

    common = {"sample_id", "label", "filename", "sha256", "license", "duration_seconds"}
    for sample in samples:
        assert common <= sample.keys(), sample["sample_id"]
        assert len(sample["sha256"]) == 64
        if sample["label"] == "bonafide":
            assert {"corpus", "corpus_url", "speaker_id", "source_utterance", "source_relpath"} <= sample.keys()
        else:
            assert {"voice", "voice_url", "voice_sha256", "speaker_id", "text_id", "text", "synthesis", "generator_version"} <= sample.keys()
            # Pinned to zero, otherwise Piper's sampling makes rebuilds differ.
            assert sample["synthesis"]["noise_scale"] == 0.0
            assert sample["synthesis"]["noise_w_scale"] == 0.0

    assert len({s["sample_id"] for s in samples}) == 80
    assert len({s["filename"] for s in samples}) == 80
