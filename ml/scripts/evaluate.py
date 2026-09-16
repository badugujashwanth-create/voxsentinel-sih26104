#!/usr/bin/env python3
"""Score the evaluation set and write auditable evidence.

    # score the prepared audio, refreshing the committed artefacts
    python ml/scripts/evaluate.py

    # recompute the metrics from committed predictions, no model or audio needed
    python ml/scripts/evaluate.py --from-predictions ml/evaluation/predictions.csv

Writes one row per sample to predictions.csv and a summary to
evaluation_results.json. Every metric is computed from the recorded labels and
scores; none is hardcoded, which is why --from-predictions can reproduce the
whole summary from the committed CSV alone.

The model score is UNCALIBRATED. It is a softmax output, not a probability that
a piece of audio is fake, and it is not a VoxSentinel risk score.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "ml" / "evaluation" / "eval_manifest.json"
DEFAULT_EVAL_DIR = REPO_ROOT / "ml" / "data" / "eval"
DEFAULT_PREDICTIONS = REPO_ROOT / "ml" / "evaluation" / "predictions.csv"
DEFAULT_RESULTS = REPO_ROOT / "ml" / "evaluation" / "evaluation_results.json"

SPOOF_LABEL = "spoof"
BONAFIDE_LABEL = "bonafide"
SUBDIR = {BONAFIDE_LABEL: "genuine", SPOOF_LABEL: "synthetic"}

#: Below this many samples per class the numbers are illustrative only.
MEANINGFUL_PER_CLASS = 50

#: Provisional midpoint. NOT a tuned operating point.
DEFAULT_THRESHOLD = 0.5

CAVEATS = [
    "80 samples only (40 genuine / 40 synthetic).",
    "One genuine corpus (LibriSpeech dev-clean) and one TTS family (Piper VITS, en_US-libritts_r).",
    "Exploratory engineering evaluation, NOT a benchmark and NOT production accuracy.",
    "The model score is uncalibrated; it is not a probability that audio is fake.",
    "Threshold 0.5 is provisional, not tuned on a dev set.",
    "The reported EER is specific to this small evaluation set.",
    "No multilingual validation. No telephony-channel or codec validation. No replay/PA evaluation.",
]


@dataclass
class Prediction:
    """One scored sample."""

    sample_id: str
    ground_truth: str
    model_score: float
    predicted_label: str
    duration_seconds: float = 0.0
    inference_ms: float = 0.0
    warnings: str = ""


@dataclass
class Metrics:
    """Confusion matrix and everything derived from it."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        """Number of scored samples."""
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def accuracy(self) -> float:
        """Share classified correctly."""
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        """Of those called spoof, the share that were."""
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """Of the actual spoofs, the share caught."""
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


def confusion(predictions: list[Prediction], threshold: float) -> Metrics:
    """Builds the confusion matrix with spoof as the positive class."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in predictions:
        predicted_spoof = row.model_score >= threshold
        actually_spoof = row.ground_truth == SPOOF_LABEL
        if predicted_spoof and actually_spoof:
            counts["tp"] += 1
        elif predicted_spoof:
            counts["fp"] += 1
        elif actually_spoof:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return Metrics(counts["tp"], counts["fp"], counts["tn"], counts["fn"])


def equal_error_rate(predictions: list[Prediction]) -> tuple[float, float]:
    """Returns (EER, the threshold achieving it).

    Threshold-independent, so it separates "can the model tell these apart"
    from "is 0.5 the right cut point".
    """
    spoof = [p.model_score for p in predictions if p.ground_truth == SPOOF_LABEL]
    bonafide = [p.model_score for p in predictions if p.ground_truth == BONAFIDE_LABEL]
    if not spoof or not bonafide:
        return float("nan"), float("nan")

    best = (float("inf"), float("nan"), float("nan"))
    for threshold in sorted({*spoof, *bonafide, 0.0, 1.0}):
        miss = sum(1 for value in spoof if value < threshold) / len(spoof)
        false_alarm = sum(1 for value in bonafide if value >= threshold) / len(bonafide)
        gap = abs(miss - false_alarm)
        if gap < best[0]:
            best = (gap, (miss + false_alarm) / 2, threshold)
    return best[1], best[2]


def distribution(predictions: list[Prediction], label: str) -> dict[str, float]:
    """Summarises the score distribution for one class."""
    values = [p.model_score for p in predictions if p.ground_truth == label]
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def score_manifest(manifest: dict, eval_dir: Path, threshold: float) -> list[Prediction]:
    """Runs the detector over every sample the manifest lists."""
    from ml.audio.preprocessing import AudioValidationError
    from ml.spoof.aasist import AASISTSpoofDetector, ModelNotInstalledError

    try:
        detector = AASISTSpoofDetector()
    except ModelNotInstalledError as exc:
        raise SystemExit(f"error: {exc}") from exc

    predictions: list[Prediction] = []
    for sample in manifest["samples"]:
        path = eval_dir / SUBDIR[sample["label"]] / sample["filename"]
        if not path.is_file():
            raise SystemExit(f"missing prepared audio: {path}\nRun: python ml/scripts/prepare_eval_set.py")
        try:
            result = detector.score_file(path)
        except AudioValidationError as exc:
            raise SystemExit(f"{sample['sample_id']}: detector refused the audio: {exc}") from exc
        predictions.append(Prediction(
            sample_id=sample["sample_id"],
            ground_truth=sample["label"],
            model_score=result.synthetic_probability,
            predicted_label=SPOOF_LABEL if result.synthetic_probability >= threshold else BONAFIDE_LABEL,
            duration_seconds=round(result.audio_duration_seconds, 4),
            inference_ms=round(result.inference_seconds * 1000, 3),
            warnings="; ".join(result.warnings),
        ))
    return predictions, detector.model_id


def read_predictions(path: Path) -> list[Prediction]:
    """Loads committed predictions so metrics can be re-derived without a model."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Prediction(
                sample_id=row["sample_id"],
                ground_truth=row["ground_truth"],
                model_score=float(row["model_score"]),
                predicted_label=row["predicted_label"],
                duration_seconds=float(row.get("duration_seconds") or 0.0),
                inference_ms=float(row.get("inference_ms") or 0.0),
                warnings=row.get("warnings", ""),
            )
            for row in csv.DictReader(handle)
        ]


def write_predictions(predictions: list[Prediction], path: Path) -> None:
    """Writes one row per sample as the audit artefact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "ground_truth", "model_score", "predicted_label", "duration_seconds", "inference_ms", "warnings"])
        for row in predictions:
            writer.writerow([row.sample_id, row.ground_truth, f"{row.model_score:.10f}", row.predicted_label, f"{row.duration_seconds:.4f}", f"{row.inference_ms:.3f}", row.warnings])


def build_results(predictions: list[Prediction], threshold: float, model_id: str) -> dict:
    """Derives the whole summary from labels and scores."""
    metrics = confusion(predictions, threshold)
    eer, eer_threshold = equal_error_rate(predictions)
    per_class = {label: sum(1 for p in predictions if p.ground_truth == label) for label in (BONAFIDE_LABEL, SPOOF_LABEL)}
    return {
        "model": model_id,
        "threshold": threshold,
        "threshold_note": "Provisional midpoint, not tuned on a dev set.",
        "score_note": "Uncalibrated softmax output of the spoof class. Not a probability that audio is fake, and not a VoxSentinel risk score.",
        "n": metrics.total,
        "counts": per_class,
        "confusion_matrix": {
            "true_positive": metrics.true_positive,
            "false_positive": metrics.false_positive,
            "true_negative": metrics.true_negative,
            "false_negative": metrics.false_negative,
            "positive_class": SPOOF_LABEL,
        },
        "metrics": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "eer": eer,
            "eer_threshold": eer_threshold,
        },
        "score_distribution": {BONAFIDE_LABEL: distribution(predictions, BONAFIDE_LABEL), SPOOF_LABEL: distribution(predictions, SPOOF_LABEL)},
        "underpowered": min(per_class.values()) < MEANINGFUL_PER_CLASS,
        "caveats": CAVEATS,
    }


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--from-predictions", type=Path, help="recompute metrics from a predictions CSV, without the model or audio")
    parser.add_argument("--out-predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--per-sample", action="store_true", help="print every sample's score")
    parser.add_argument("--no-write", action="store_true", help="do not update the committed artefacts")
    args = parser.parse_args()

    if args.from_predictions:
        predictions = read_predictions(args.from_predictions)
        model_id = "(recomputed from committed predictions)"
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        predictions, model_id = score_manifest(manifest, args.eval_dir, args.threshold)

    results = build_results(predictions, args.threshold, model_id)

    if not args.no_write and not args.from_predictions:
        write_predictions(predictions, args.out_predictions)
        args.out_results.parent.mkdir(parents=True, exist_ok=True)
        args.out_results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    matrix = results["confusion_matrix"]
    metrics = results["metrics"]

    print(f"Model:      {results['model']}")
    print(f"Threshold:  {args.threshold}  (positive class = {SPOOF_LABEL}; provisional midpoint, not tuned)")
    print(f"Scored:     {results['n']} samples ({results['counts'][BONAFIDE_LABEL]} {BONAFIDE_LABEL}, {results['counts'][SPOOF_LABEL]} {SPOOF_LABEL})")

    print("\nUncalibrated model-score distribution:")
    for label in (BONAFIDE_LABEL, SPOOF_LABEL):
        d = results["score_distribution"][label]
        print(f"  {label:9} n={d['n']:<4} min={d['min']:.4f} median={d['median']:.4f} max={d['max']:.4f} stdev={d['stdev']:.4f}")

    print("\nConfusion matrix:")
    print(f"                    predicted {SPOOF_LABEL:9} predicted {BONAFIDE_LABEL}")
    print(f"  actual {SPOOF_LABEL:9}  {matrix['true_positive']:>17}  {matrix['false_negative']:>19}")
    print(f"  actual {BONAFIDE_LABEL:9}  {matrix['false_positive']:>17}  {matrix['true_negative']:>19}")

    print("\nMetrics (computed from labels + scores, never hardcoded):")
    print(f"  accuracy   {metrics['accuracy']:.4f}")
    print(f"  precision  {metrics['precision']:.4f}")
    print(f"  recall     {metrics['recall']:.4f}")
    print(f"  f1         {metrics['f1']:.4f}")
    print(f"  EER        {metrics['eer']:.4f}   at threshold {metrics['eer_threshold']:.4f}")

    if args.per_sample:
        print("\nPer-sample:")
        for row in sorted(predictions, key=lambda p: (p.ground_truth, -p.model_score)):
            flag = "" if row.predicted_label == row.ground_truth else "  <- WRONG"
            print(f"  {row.sample_id:15} {row.ground_truth:9} {row.model_score:.4f}{flag}")

    if results["underpowered"]:
        print("\nWARNING: fewer than 50 samples per class.")
        for caveat in CAVEATS:
            print(f"  - {caveat}")

    if not args.no_write and not args.from_predictions:
        print(f"\nWrote {args.out_predictions}")
        print(f"Wrote {args.out_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
