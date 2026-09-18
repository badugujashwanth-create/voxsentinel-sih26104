#!/usr/bin/env python3
"""Score the speaker-verification trial set and write auditable evidence.

    # embed the prepared audio, score every trial, refresh the committed artefacts
    python ml/scripts/evaluate_speaker.py

    # recompute every metric from committed predictions: no model, no audio
    python ml/scripts/evaluate_speaker.py --from-predictions ml/evaluation/sv_predictions.csv

Writes one row per trial to sv_predictions.csv and a summary to sv_results.json.
Every metric is computed from the recorded relations and similarities; none is
hardcoded, which is why --from-predictions reproduces the whole summary from the
committed CSV alone.

The similarity is a cosine, and it is UNCALIBRATED. It is not a probability that
two recordings share a speaker, and it is not a VoxSentinel risk score.

Where the threshold comes from
------------------------------
The default is the equal-error-rate operating point of the LibriSpeech trials
only. The CMU ARCTIC trials and every attack trial are then scored at that
threshold without contributing to it, so the clone numbers are measured against
a cut point that was not fitted to them.
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

from ml.scripts.evaluate import Metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "ml" / "evaluation" / "sv_manifest.json"
DEFAULT_EVAL_DIR = REPO_ROOT / "ml" / "data" / "sv-eval"
DEFAULT_PREDICTIONS = REPO_ROOT / "ml" / "evaluation" / "sv_predictions.csv"
DEFAULT_RESULTS = REPO_ROOT / "ml" / "evaluation" / "sv_results.json"

SUBDIR = {"librispeech": "librispeech", "cmu_arctic": "arctic", "piper_tts": "clone"}

#: Relations that are speaker-verification trials. The attack relations are
#: scored and reported, but kept out of the error rates: FAR is defined over
#: human impostors, and quietly folding clones in would change what it means.
HUMAN_RELATIONS = ("same_speaker", "different_speaker")
ATTACK_RELATIONS = ("clone", "synthetic_non_target")

#: The threshold is fitted on this corpus alone, so everything else is held out.
THRESHOLD_CORPUS = "librispeech"

#: Below this many trials per class the numbers are illustrative only.
MEANINGFUL_PER_CLASS = 100

CAVEATS = [
    "Exploratory engineering evaluation, NOT a benchmark and NOT production accuracy.",
    "37 speakers in total: 31 LibriSpeech dev-clean and 6 CMU ARCTIC.",
    "Only 6 speakers and 12 trials behind the clone result. It is a demonstration, not a rate.",
    "Two read-speech corpora, both clean and studio-derived. No telephony channel, no codec, no noise.",
    "One clone family (Piper VITS, en_US-arctic). Newer zero-shot cloning systems are untested.",
    "English only. No multilingual validation.",
    "The similarity is an uncalibrated cosine, not a probability that two recordings share a speaker.",
    "The LibriSpeech classes separate completely here. That is a statement about how easy clean read speech is, not about the system: expect overlap on real call audio.",
    "The threshold is fitted on the LibriSpeech trials and not validated on held-out data.",
    "ECAPA was trained on VoxCeleb, which overlaps neither corpus, but read speech is easier than VoxCeleb-style audio.",
]


@dataclass
class Trial:
    """One scored reference-probe pair."""

    trial_id: str
    reference_speaker: str
    probe_sample: str
    relation: str
    corpus: str
    similarity: float
    reference_seconds: float = 0.0
    probe_seconds: float = 0.0
    inference_ms: float = 0.0
    warnings: str = ""

    @property
    def is_target(self) -> bool:
        """True when the probe really is the enrolled speaker."""
        return self.relation == "same_speaker"


class TrialMetrics(Metrics):
    """The spoof evaluation's confusion matrix, plus the two rates SV reports.

    Positive class is "same speaker", so a false positive is an impostor the
    system accepted and a false negative is a genuine speaker it turned away.
    """

    @property
    def false_accept_rate(self) -> float:
        """Share of impostor trials wrongly accepted."""
        negatives = self.false_positive + self.true_negative
        return self.false_positive / negatives if negatives else 0.0

    @property
    def false_reject_rate(self) -> float:
        """Share of genuine trials wrongly turned away."""
        positives = self.true_positive + self.false_negative
        return self.false_negative / positives if positives else 0.0


def confusion(trials: list[Trial], threshold: float) -> TrialMetrics:
    """Builds the confusion matrix with "same speaker" as the positive class."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for trial in trials:
        accepted = trial.similarity >= threshold
        if accepted and trial.is_target:
            counts["tp"] += 1
        elif accepted:
            counts["fp"] += 1
        elif trial.is_target:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return TrialMetrics(counts["tp"], counts["fp"], counts["tn"], counts["fn"])


def equal_error_rate(target_scores: list[float], nontarget_scores: list[float]) -> tuple[float, float]:
    """Returns (EER, the threshold achieving it) by sweeping every observed score.

    Threshold-independent, so it separates "can the model tell these speakers
    apart" from "is this the right cut point".
    """
    if not target_scores or not nontarget_scores:
        return float("nan"), float("nan")

    best = (float("inf"), float("nan"), float("nan"))
    for threshold in sorted({*target_scores, *nontarget_scores, -1.0, 1.0}):
        false_reject = sum(1 for value in target_scores if value < threshold) / len(target_scores)
        false_accept = sum(1 for value in nontarget_scores if value >= threshold) / len(nontarget_scores)
        gap = abs(false_reject - false_accept)
        if gap < best[0]:
            best = (gap, (false_reject + false_accept) / 2, threshold)
    return best[1], best[2]


def distribution(scores: list[float]) -> dict:
    """Summarises a score distribution."""
    if not scores:
        return {}
    return {
        "n": len(scores),
        "min": min(scores),
        "median": statistics.median(scores),
        "mean": statistics.fmean(scores),
        "max": max(scores),
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
    }


def metric_block(trials: list[Trial], threshold: float) -> dict:
    """Confusion matrix, rates, and EER for one slice of the trial set."""
    matrix = confusion(trials, threshold)
    targets = [t.similarity for t in trials if t.is_target]
    nontargets = [t.similarity for t in trials if not t.is_target]
    eer, eer_threshold = equal_error_rate(targets, nontargets)
    return {
        "positive_pairs": len(targets),
        "negative_pairs": len(nontargets),
        "confusion_matrix": {
            "true_positive": matrix.true_positive,
            "false_positive": matrix.false_positive,
            "true_negative": matrix.true_negative,
            "false_negative": matrix.false_negative,
            "positive_class": "same_speaker",
        },
        "metrics": {
            "accuracy": matrix.accuracy,
            "precision": matrix.precision,
            "recall": matrix.recall,
            "f1": matrix.f1,
            "false_accept_rate": matrix.false_accept_rate,
            "false_reject_rate": matrix.false_reject_rate,
            "eer": eer,
            "eer_threshold": eer_threshold,
        },
        "score_distribution": {
            "same_speaker": distribution(targets),
            "different_speaker": distribution(nontargets),
        },
    }


def score_manifest(manifest: dict, eval_dir: Path, vendor_dir: Path | None) -> tuple[list[Trial], str, dict]:
    """Enrols every speaker and scores every trial with the real model."""
    from ml.audio.preprocessing import AudioValidationError
    from ml.speaker.ecapa import ECAPASpeakerVerifier, ModelNotInstalledError
    from ml.speaker.verifier import cosine_similarity

    try:
        verifier = ECAPASpeakerVerifier(**({"vendor_dir": vendor_dir} if vendor_dir else {}))
    except ModelNotInstalledError as exc:
        raise SystemExit(f"error: {exc}") from exc

    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}

    def path_of(sample_id: str) -> Path:
        sample = samples[sample_id]
        path = eval_dir / SUBDIR[sample["source_type"]] / sample["filename"]
        if not path.is_file():
            raise SystemExit(f"missing prepared audio: {path}\nRun: python ml/scripts/prepare_sv_eval_set.py")
        return path

    # Every file is embedded exactly once; a trial is then a dot product. With
    # 442 trials over 172 files this is the difference between one minute and
    # fifteen, and it also guarantees the same file cannot contribute two
    # different embeddings to two different trials.
    embeddings = {}
    for sample_id in samples:
        try:
            embeddings[sample_id] = verifier.embed_file(path_of(sample_id))
        except AudioValidationError as exc:
            raise SystemExit(f"{sample_id}: the verifier refused the audio: {exc}") from exc

    from ml.speaker.verifier import SpeakerEmbedding, aggregate

    references = {}
    for speaker in manifest["speakers"]:
        parts = [embeddings[sample_id] for sample_id in speaker["enrolment"]]
        references[speaker["speaker_key"]] = SpeakerEmbedding(
            vector=aggregate([part.vector for part in parts]),
            model_id=verifier.model_id,
            source_count=len(parts),
            total_duration_seconds=sum(part.total_duration_seconds for part in parts),
            sample_rate=verifier.expected_sample_rate,
            inference_seconds=sum(part.inference_seconds for part in parts),
            preprocessing_seconds=sum(part.preprocessing_seconds for part in parts),
            warnings=tuple(dict.fromkeys(w for part in parts for w in part.warnings)),
        )

    trials = []
    for entry in manifest["trials"]:
        reference = references[entry["reference_speaker"]]
        probe = embeddings[entry["probe_sample"]]
        trials.append(Trial(
            trial_id=entry["trial_id"],
            reference_speaker=entry["reference_speaker"],
            probe_sample=entry["probe_sample"],
            relation=entry["relation"],
            corpus=entry["corpus"],
            # Rounded to the precision sv_predictions.csv stores, so the
            # committed summary is exactly what the committed evidence yields.
            # Without this the JSON is derived from floats the CSV cannot
            # represent, and --from-predictions reproduces it only approximately.
            similarity=round(cosine_similarity(reference.vector, probe.vector), 10),
            reference_seconds=round(reference.total_duration_seconds, 4),
            probe_seconds=round(probe.total_duration_seconds, 4),
            inference_ms=round(probe.inference_seconds * 1000, 3),
            warnings="; ".join(dict.fromkeys([*reference.warnings, *probe.warnings])),
        ))

    latency = sorted(part.inference_seconds * 1000 for part in embeddings.values())
    latency_summary = {
        "embeddings_measured": len(latency),
        "inference_ms_median": statistics.median(latency),
        "inference_ms_mean": statistics.fmean(latency),
        "inference_ms_min": latency[0],
        "inference_ms_max": latency[-1],
        "inference_ms_p95": latency[min(len(latency) - 1, int(0.95 * len(latency)))],
        "preprocessing_ms_median": statistics.median(sorted(part.preprocessing_seconds * 1000 for part in embeddings.values())),
        "audio_seconds_median": statistics.median(sorted(part.total_duration_seconds for part in embeddings.values())),
        "ms_per_audio_second_median": statistics.median(sorted(
            part.inference_seconds * 1000 / part.total_duration_seconds for part in embeddings.values()
        )),
        "note": (
            "One forward pass per utterance, measured during this evaluation run on this machine. Excludes model "
            "load. ECAPA cost scales with utterance length, and these utterances vary from about 3 to 30 seconds, "
            "so the spread below is mostly duration, not jitter: ms_per_audio_second_median is the comparable figure."
        ),
    }
    return trials, verifier.model_id, latency_summary


def read_predictions(path: Path) -> list[Trial]:
    """Loads committed trials so metrics can be re-derived without a model."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Trial(
                trial_id=row["trial_id"],
                reference_speaker=row["reference_speaker"],
                probe_sample=row["probe_sample"],
                relation=row["relation"],
                corpus=row["corpus"],
                similarity=float(row["similarity"]),
                reference_seconds=float(row.get("reference_seconds") or 0.0),
                probe_seconds=float(row.get("probe_seconds") or 0.0),
                inference_ms=float(row.get("inference_ms") or 0.0),
                warnings=row.get("warnings", ""),
            )
            for row in csv.DictReader(handle)
        ]


def write_predictions(trials: list[Trial], threshold: float, path: Path) -> None:
    """Writes one row per trial as the audit artefact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_id", "reference_speaker", "probe_sample", "relation", "corpus", "similarity", "decision", "reference_seconds", "probe_seconds", "inference_ms", "warnings"])
        for trial in trials:
            writer.writerow([
                trial.trial_id, trial.reference_speaker, trial.probe_sample, trial.relation, trial.corpus,
                f"{trial.similarity:.10f}", "match" if trial.similarity >= threshold else "mismatch",
                f"{trial.reference_seconds:.4f}", f"{trial.probe_seconds:.4f}", f"{trial.inference_ms:.3f}", trial.warnings,
            ])


def choose_threshold(trials: list[Trial]) -> tuple[float, dict]:
    """Picks the cut point from the threshold corpus, and says which rule applied.

    Two cases, because one rule does not cover both:

    * The classes overlap. Then the equal error rate has a unique operating
      point and that is the threshold.
    * The classes separate completely, which is what happens on clean read
      speech. Then EER is zero and *every* value between the highest impostor
      score and the lowest genuine score scores identically, so "the EER
      threshold" is not a number -- the sweep just returns whichever end of the
      gap it reached first. Sitting at either edge means one more difficult
      trial tips the result. The midpoint of that gap is the only choice with a
      reason behind it: it is the furthest any cut point can be from both
      classes at once.

    Rounded to two decimals so the shipped default is a stable number rather
    than a fifteen-digit artefact of one particular trial list.
    """
    fitted = [t for t in trials if t.corpus == THRESHOLD_CORPUS and t.relation in HUMAN_RELATIONS]
    targets = [t.similarity for t in fitted if t.is_target]
    nontargets = [t.similarity for t in fitted if not t.is_target]
    eer, eer_threshold = equal_error_rate(targets, nontargets)

    note = {
        "fitted_on": THRESHOLD_CORPUS,
        "fitted_trials": len(fitted),
        "eer": eer,
        "eer_sweep_threshold": eer_threshold,
        "lowest_genuine_score": min(targets) if targets else None,
        "highest_impostor_score": max(nontargets) if nontargets else None,
        "held_out": "cmu_arctic trials and all attack trials were not used to pick it",
        "not_a_probability": "a cosine cut point, not a calibrated likelihood",
    }

    if eer > 0 or not targets or not nontargets:
        note["rule"] = f"equal error rate of the {THRESHOLD_CORPUS} same-speaker / different-speaker trials, rounded to 2 decimals"
        return round(eer_threshold, 2), note

    low, high = max(nontargets), min(targets)
    note["rule"] = (
        f"the {THRESHOLD_CORPUS} classes separate completely (EER 0), so the EER point is not unique: every cut "
        f"point in ({low:.4f}, {high:.4f}] scores the same. The midpoint of that gap is used, rounded to 2 decimals."
    )
    note["separating_interval"] = [low, high]
    note["margin"] = high - low
    return round((low + high) / 2, 2), note


def build_results(trials: list[Trial], threshold: float, threshold_note: dict, model_id: str, latency: dict) -> dict:
    """Derives the whole summary from relations and similarities."""
    human = [t for t in trials if t.relation in HUMAN_RELATIONS]
    corpora = sorted({t.corpus for t in human})

    attacks = {}
    for relation in ATTACK_RELATIONS:
        scores = [t.similarity for t in trials if t.relation == relation]
        if not scores:
            continue
        accepted = sum(1 for value in scores if value >= threshold)
        attacks[relation] = {
            "n": len(scores),
            "accepted_at_threshold": accepted,
            "acceptance_rate": accepted / len(scores),
            "score_distribution": distribution(scores),
        }

    return {
        "model": model_id,
        "similarity": "cosine of two L2-normalised ECAPA-TDNN embeddings",
        "score_note": "UNCALIBRATED. Not a probability that two recordings share a speaker, and not a VoxSentinel risk score.",
        "threshold": threshold,
        "threshold_note": threshold_note,
        "counts": {
            "trials": len(trials),
            "speaker_verification_trials": len(human),
            "by_relation": {relation: sum(1 for t in trials if t.relation == relation) for relation in (*HUMAN_RELATIONS, *ATTACK_RELATIONS)},
        },
        "overall": metric_block(human, threshold),
        "by_corpus": {corpus: metric_block([t for t in human if t.corpus == corpus], threshold) for corpus in corpora},
        "attacks": attacks,
        "attack_note": (
            "Clone and synthetic trials are NOT counted in the false accept rate above. FAR is defined over human "
            "impostors; a clone is a different attack, and the whole point of the number is that speaker verification "
            "alone does not stop it. That is what ml/spoof/ is for."
        ),
        "latency": latency,
        "underpowered": min(len([t for t in human if t.is_target]), len([t for t in human if not t.is_target])) < MEANINGFUL_PER_CLASS,
        "caveats": CAVEATS,
    }


def print_block(title: str, block: dict) -> None:
    """Prints one metric block."""
    matrix = block["confusion_matrix"]
    metrics = block["metrics"]
    print(f"\n{title}  ({block['positive_pairs']} positive pairs, {block['negative_pairs']} negative pairs)")
    print("                        predicted match   predicted mismatch")
    print(f"  actual same speaker   {matrix['true_positive']:>15}   {matrix['false_negative']:>18}")
    print(f"  actual different      {matrix['false_positive']:>15}   {matrix['true_negative']:>18}")
    print(f"  accuracy {metrics['accuracy']:.4f}   precision {metrics['precision']:.4f}   recall {metrics['recall']:.4f}   f1 {metrics['f1']:.4f}")
    print(f"  FAR      {metrics['false_accept_rate']:.4f}   FRR       {metrics['false_reject_rate']:.4f}   EER    {metrics['eer']:.4f} at {metrics['eer_threshold']:.4f}")
    for label in ("same_speaker", "different_speaker"):
        d = block["score_distribution"][label]
        if d:
            print(f"  {label:18} n={d['n']:<4} min={d['min']:+.4f} median={d['median']:+.4f} max={d['max']:+.4f} stdev={d['stdev']:.4f}")


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--vendor-dir", type=Path, default=None, help="ECAPA checkpoint directory")
    parser.add_argument("--from-predictions", type=Path, help="recompute metrics from a predictions CSV, without the model or audio")
    parser.add_argument("--out-predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--threshold", type=float, help="score at this cut point instead of the fitted one")
    parser.add_argument("--per-trial", action="store_true", help="print every trial's similarity")
    parser.add_argument("--no-write", action="store_true", help="do not update the committed artefacts")
    args = parser.parse_args()

    if args.from_predictions:
        trials = read_predictions(args.from_predictions)
        model_id = "(recomputed from committed predictions)"
        latency = {"note": "not recomputed; latency comes from a model run"}
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        trials, model_id, latency = score_manifest(manifest, args.eval_dir, args.vendor_dir)

    fitted_threshold, threshold_note = choose_threshold(trials)
    if args.threshold is not None:
        threshold_note = {**threshold_note, "overridden": f"caller passed --threshold {args.threshold}; the fitted value was {fitted_threshold}"}
    threshold = fitted_threshold if args.threshold is None else args.threshold

    results = build_results(trials, threshold, threshold_note, model_id, latency)

    if not args.no_write and not args.from_predictions:
        write_predictions(trials, threshold, args.out_predictions)
        args.out_results.parent.mkdir(parents=True, exist_ok=True)
        args.out_results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Model:      {results['model']}")
    print(f"Similarity: {results['similarity']} (UNCALIBRATED)")
    print(f"Threshold:  {threshold}  ({threshold_note['rule']})")
    print(f"Trials:     {results['counts']['trials']}  " + "  ".join(f"{k}={v}" for k, v in results["counts"]["by_relation"].items()))

    print_block("All speaker-verification trials", results["overall"])
    for corpus, block in results["by_corpus"].items():
        print_block(f"Corpus: {corpus}", block)

    if results["attacks"]:
        print("\nAttack trials (NOT counted in the rates above):")
        for relation, block in results["attacks"].items():
            d = block["score_distribution"]
            print(f"  {relation:22} n={block['n']:<4} accepted {block['accepted_at_threshold']}/{block['n']} "
                  f"({block['acceptance_rate']:.1%})  min={d['min']:+.4f} median={d['median']:+.4f} max={d['max']:+.4f}")

    if latency.get("embeddings_measured"):
        print(f"\nLatency over {latency['embeddings_measured']} utterance embeddings "
              f"(median audio {latency['audio_seconds_median']:.2f}s):")
        print(f"  preprocessing  median {latency['preprocessing_ms_median']:.2f} ms")
        print(f"  inference      median {latency['inference_ms_median']:.2f} ms   p95 {latency['inference_ms_p95']:.2f} ms   "
              f"min {latency['inference_ms_min']:.2f}   max {latency['inference_ms_max']:.2f}")
        print(f"  inference      median {latency['ms_per_audio_second_median']:.2f} ms per second of audio "
              f"({1000 / latency['ms_per_audio_second_median']:.1f}x real time)")

    if args.per_trial:
        print("\nPer-trial:")
        for trial in sorted(trials, key=lambda t: (t.relation, -t.similarity)):
            print(f"  {trial.trial_id:44} {trial.relation:22} {trial.similarity:+.4f}")

    if results["underpowered"]:
        print(f"\nWARNING: fewer than {MEANINGFUL_PER_CLASS} trials in one class. Exploratory, not production accuracy.")
    for caveat in CAVEATS:
        print(f"  - {caveat}")

    if not args.no_write and not args.from_predictions:
        print(f"\nWrote {args.out_predictions}")
        print(f"Wrote {args.out_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
