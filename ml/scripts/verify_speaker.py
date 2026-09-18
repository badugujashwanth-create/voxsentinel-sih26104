#!/usr/bin/env python3
"""Enrol a speaker and verify a probe utterance against them.

    python ml/scripts/verify_speaker.py \
        --reference ref1.flac ref2.flac \
        --probe probe.flac

    # score several probes against one enrolment
    python ml/scripts/verify_speaker.py --reference ref.flac --probe a.wav b.wav

The similarity is an UNCALIBRATED cosine. It is not a probability that the two
recordings share a speaker, it is not an anti-spoof verdict, and it is not a
VoxSentinel risk score. A voice cloned from the enrolled speaker scores like the
enrolled speaker: see ml/speaker/README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.audio.preprocessing import AudioValidationError  # noqa: E402
from ml.speaker.ecapa import DEFAULT_VENDOR_DIR, ECAPASpeakerVerifier, ModelNotInstalledError  # noqa: E402
from ml.speaker.verifier import EnrollmentError  # noqa: E402


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, nargs="+", required=True, help="one or more enrolment utterances")
    parser.add_argument("--probe", type=Path, nargs="+", required=True, help="one or more utterances to verify")
    parser.add_argument("--vendor-dir", type=Path, default=DEFAULT_VENDOR_DIR, help="ECAPA checkpoint directory")
    parser.add_argument("--threshold", type=float, default=None, help="override the fitted decision threshold")
    args = parser.parse_args()

    try:
        verifier = ECAPASpeakerVerifier(vendor_dir=args.vendor_dir)
    except ModelNotInstalledError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        reference = verifier.enroll(args.reference)
    except EnrollmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Enrolled:              {reference.source_count} utterance(s), {reference.total_duration_seconds:.2f}s total")
    print(f"Embedding:             {reference.dimensions}-d, L2-normalised")
    print(f"Model:                 {verifier.model_id}")
    print(f"Threshold:             {args.threshold if args.threshold is not None else verifier.decision_threshold}")
    for warning in reference.warnings:
        print(f"  enrolment warning:   {warning}")

    failed = False
    for probe in args.probe:
        print()
        try:
            result = verifier.verify_file(reference, probe, args.threshold)
        except AudioValidationError as exc:
            print(f"Probe:                 {probe}")
            print(f"  REFUSED:             {exc}")
            failed = True
            continue

        print(f"Probe:                 {probe}")
        print(f"Similarity:            {result.similarity_score:+.4f}   (uncalibrated cosine, -1..1)")
        print(f"Decision:              {'MATCH' if result.is_match else 'MISMATCH'}  (threshold {result.threshold})")
        print(f"Reference duration:    {result.reference_duration_seconds:.2f}s over {result.reference_utterances} utterance(s)")
        print(f"Probe duration:        {result.probe_duration_seconds:.2f}s at {verifier.expected_sample_rate} Hz")
        print(f"Preprocessing:         {result.preprocessing_seconds * 1000:.1f} ms")
        print(f"Inference latency:     {result.inference_seconds * 1000:.1f} ms")
        print(f"Total:                 {result.total_seconds * 1000:.1f} ms")
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

    print(
        "\nUNCALIBRATED cosine similarity between speaker embeddings. Not a probability that the voices match, "
        "not an anti-spoof verdict, not an identity decision, and not a VoxSentinel risk score. A voice cloned "
        "from the enrolled speaker scores like the enrolled speaker."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
