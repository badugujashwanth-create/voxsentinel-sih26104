#!/usr/bin/env python3
"""Score one or more audio files for synthetic-speech characteristics.

    python ml/scripts/detect.py --audio path/to/sample.wav

This reports a SPOOF-DETECTION probability only. It is not a fraud decision, not
an identity decision, and not a VoxSentinel risk score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.audio.preprocessing import AudioValidationError  # noqa: E402
from ml.spoof.aasist import AASISTSpoofDetector, ModelNotInstalledError  # noqa: E402
from ml.spoof.detector import SpoofResult  # noqa: E402

DISCLAIMER = "Spoof-detection score only - not an identity or fraud decision."


def render(path: Path, result: SpoofResult) -> str:
    """Formats one result for a human reader."""
    lines = [
        f"File:                  {path}",
        f"Synthetic probability: {result.synthetic_probability:.4f}",
        f"Bona-fide score:       {result.bonafide_score:+.4f}  (raw model output, higher = more human)",
        f"Model:                 {result.model_id}",
        f"Audio duration:        {result.audio_duration_seconds:.2f}s at {result.sample_rate} Hz",
        f"Preprocessing:         {result.preprocessing_seconds * 1000:.1f} ms",
        f"Inference latency:     {result.inference_seconds * 1000:.1f} ms",
        f"Total:                 {result.total_seconds * 1000:.1f} ms",
    ]
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)
    return "\n".join(lines)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, nargs="+", type=Path, help="one or more audio files to score")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    try:
        detector = AASISTSpoofDetector(device=args.device)
    except ModelNotInstalledError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payloads: list[dict] = []
    failures = 0
    for path in args.audio:
        try:
            result = detector.score_file(path)
        except AudioValidationError as exc:
            failures += 1
            if args.json:
                payloads.append({"file": str(path), "error": str(exc)})
            else:
                print(f"File:                  {path}\nRejected:              {exc}\n", file=sys.stderr)
            continue

        if args.json:
            payloads.append({"file": str(path), **result.__dict__})
        else:
            print(render(path, result))
            print()

    if args.json:
        print(json.dumps({"disclaimer": DISCLAIMER, "results": payloads}, indent=2, default=list))
    else:
        print(DISCLAIMER)

    return 1 if failures and not payloads else 0


if __name__ == "__main__":
    raise SystemExit(main())
