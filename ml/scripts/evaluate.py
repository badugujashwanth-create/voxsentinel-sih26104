#!/usr/bin/env python3
"""Evaluate the spoof detector against labelled audio.

    python ml/scripts/evaluate.py --manifest ml/data/manifest.csv

The manifest is CSV with a header and two columns:

    path,label
    samples/genuine/0001.flac,bonafide
    samples/synthetic/0001.wav,spoof

Paths are resolved relative to the manifest. Reports counts, accuracy,
precision, recall, F1, and the confusion matrix at the chosen threshold.

Small sample counts do not support accuracy claims, and this tool says so
rather than quietly printing a number that looks authoritative.
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

from ml.audio.preprocessing import AudioValidationError  # noqa: E402
from ml.spoof.aasist import AASISTSpoofDetector, ModelNotInstalledError  # noqa: E402
from ml.spoof.detector import SpoofDetector  # noqa: E402

SPOOF_LABEL = "spoof"
BONAFIDE_LABEL = "bonafide"
VALID_LABELS = {SPOOF_LABEL, BONAFIDE_LABEL}

#: Below this many samples per class, metrics are illustrative only.
MEANINGFUL_PER_CLASS = 50

#: Default decision threshold: at or above this, call it synthetic. This is an
#: arbitrary midpoint, NOT a tuned operating point. Tune on a real dev set.
DEFAULT_THRESHOLD = 0.5


@dataclass
class Scored:
    """One evaluated sample."""

    path: Path
    label: str
    synthetic_probability: float
    warnings: tuple[str, ...]


@dataclass
class Metrics:
    """Confusion matrix and derived metrics at one threshold."""

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
        """Share of samples classified correctly."""
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        """Of those called synthetic, the share that really were."""
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """Of the truly synthetic samples, the share we caught."""
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


def read_manifest(manifest: Path) -> list[tuple[Path, str]]:
    """Reads (path, label) pairs, resolving paths against the manifest."""
    rows: list[tuple[Path, str]] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if row.get("path") is None or row.get("label") is None:
                raise SystemExit(f"{manifest}:{line_number}: manifest needs 'path' and 'label' columns")
            label = row["label"].strip().lower()
            if label not in VALID_LABELS:
                raise SystemExit(f"{manifest}:{line_number}: label must be one of {sorted(VALID_LABELS)}, got {label!r}")
            rows.append(((manifest.parent / row["path"].strip()).resolve(), label))
    if not rows:
        raise SystemExit(f"{manifest}: no samples listed")
    return rows


def score_all(detector: SpoofDetector, rows: list[tuple[Path, str]]) -> tuple[list[Scored], list[tuple[Path, str]]]:
    """Scores every sample, separating those the detector refused."""
    scored: list[Scored] = []
    rejected: list[tuple[Path, str]] = []
    for path, label in rows:
        try:
            result = detector.score_file(path)
        except AudioValidationError as exc:
            rejected.append((path, str(exc)))
            continue
        except FileNotFoundError:
            rejected.append((path, "file not found"))
            continue
        scored.append(Scored(path=path, label=label, synthetic_probability=result.synthetic_probability, warnings=result.warnings))
    return scored, rejected


def confusion(scored: list[Scored], threshold: float) -> Metrics:
    """Builds the confusion matrix treating 'spoof' as the positive class."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for sample in scored:
        predicted_spoof = sample.synthetic_probability >= threshold
        actually_spoof = sample.label == SPOOF_LABEL
        if predicted_spoof and actually_spoof:
            counts["tp"] += 1
        elif predicted_spoof:
            counts["fp"] += 1
        elif actually_spoof:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return Metrics(counts["tp"], counts["fp"], counts["tn"], counts["fn"])


def equal_error_rate(scored: list[Scored]) -> tuple[float, float]:
    """Returns (EER, the threshold achieving it).

    Threshold-independent, so it separates "can the model tell these apart at
    all" from "is 0.5 the right cut point". Standard for anti-spoofing.
    """
    spoof = [s.synthetic_probability for s in scored if s.label == SPOOF_LABEL]
    bonafide = [s.synthetic_probability for s in scored if s.label == BONAFIDE_LABEL]
    if not spoof or not bonafide:
        return float("nan"), float("nan")

    best = (float("inf"), float("nan"), float("nan"))
    for threshold in sorted({*spoof, *bonafide, 0.0, 1.0}):
        # Miss rate: real spoofs scored below the threshold.
        miss = sum(1 for value in spoof if value < threshold) / len(spoof)
        # False-alarm rate: genuine speech scored at or above it.
        false_alarm = sum(1 for value in bonafide if value >= threshold) / len(bonafide)
        gap = abs(miss - false_alarm)
        if gap < best[0]:
            best = (gap, (miss + false_alarm) / 2, threshold)
    return best[1], best[2]


def summarise(scored: list[Scored], label: str) -> str:
    """Describes the probability distribution for one class."""
    values = [sample.synthetic_probability for sample in scored if sample.label == label]
    if not values:
        return f"  {label:9} (none)"
    spread = f" stdev={statistics.stdev(values):.4f}" if len(values) > 1 else ""
    return f"  {label:9} n={len(values):<4} min={min(values):.4f} median={statistics.median(values):.4f} max={max(values):.4f}{spread}"


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="CSV manifest of labelled samples")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"synthetic decision threshold (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--per-sample", action="store_true", help="print every sample's score")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    try:
        detector = AASISTSpoofDetector(device=args.device)
    except ModelNotInstalledError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    scored, rejected = score_all(detector, rows)
    if not scored:
        print("error: no samples could be scored", file=sys.stderr)
        return 1

    metrics = confusion(scored, args.threshold)
    eer, eer_threshold = equal_error_rate(scored)
    per_class = {label: sum(1 for sample in scored if sample.label == label) for label in VALID_LABELS}
    underpowered = min(per_class.values()) < MEANINGFUL_PER_CLASS

    if args.json:
        print(json.dumps({
            "model": detector.model_id,
            "threshold": args.threshold,
            "counts": {"scored": len(scored), "rejected": len(rejected), **per_class},
            "metrics": {"accuracy": metrics.accuracy, "precision": metrics.precision, "recall": metrics.recall, "f1": metrics.f1, "eer": eer, "eer_threshold": eer_threshold},
            "confusion_matrix": {"true_positive": metrics.true_positive, "false_positive": metrics.false_positive, "true_negative": metrics.true_negative, "false_negative": metrics.false_negative},
            "underpowered": underpowered,
            "samples": [{"path": str(s.path), "label": s.label, "synthetic_probability": s.synthetic_probability} for s in scored],
        }, indent=2))
        return 0

    print(f"Model:      {detector.model_id}")
    print(f"Threshold:  {args.threshold}  (positive class = {SPOOF_LABEL}; arbitrary midpoint, not a tuned operating point)")
    print(f"Scored:     {len(scored)} samples ({per_class[BONAFIDE_LABEL]} {BONAFIDE_LABEL}, {per_class[SPOOF_LABEL]} {SPOOF_LABEL})")
    if rejected:
        print(f"Rejected:   {len(rejected)} samples")
        for path, reason in rejected:
            print(f"  - {path.name}: {reason}")

    print("\nSynthetic-probability distribution:")
    print(summarise(scored, BONAFIDE_LABEL))
    print(summarise(scored, SPOOF_LABEL))

    print("\nConfusion matrix:")
    print(f"                    predicted {SPOOF_LABEL:9} predicted {BONAFIDE_LABEL}")
    print(f"  actual {SPOOF_LABEL:9}  {metrics.true_positive:>17}  {metrics.false_negative:>19}")
    print(f"  actual {BONAFIDE_LABEL:9}  {metrics.false_positive:>17}  {metrics.true_negative:>19}")

    print("\nMetrics:")
    print(f"  accuracy   {metrics.accuracy:.4f}")
    print(f"  precision  {metrics.precision:.4f}")
    print(f"  recall     {metrics.recall:.4f}")
    print(f"  f1         {metrics.f1:.4f}")
    print(f"  EER        {eer:.4f}   at threshold {eer_threshold:.4f}  (threshold-independent separability)")

    if args.per_sample:
        print("\nPer-sample:")
        for sample in sorted(scored, key=lambda s: (s.label, -s.synthetic_probability)):
            flag = "" if (sample.synthetic_probability >= args.threshold) == (sample.label == SPOOF_LABEL) else "  <- WRONG"
            print(f"  {sample.label:9} {sample.synthetic_probability:.4f}  {sample.path.name}{flag}")

    if underpowered:
        smallest = min(per_class, key=lambda label: per_class[label])
        print(
            f"\nWARNING: only {per_class[smallest]} {smallest} sample(s). Fewer than {MEANINGFUL_PER_CLASS} per class.\n"
            "These numbers describe this handful of files and nothing more. They are NOT an\n"
            "accuracy claim, NOT a benchmark result, and must not be quoted as either. Run\n"
            "against a proper evaluation set (e.g. ASVspoof 2019 LA eval) before claiming performance."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
