#!/usr/bin/env python3
"""Assemble a labelled evaluation set and write its manifest.

Genuine speech comes from LibriSpeech dev-clean (CC BY 4.0, OpenSLR resource
12), which is read speech from public-domain LibriVox audiobooks. Synthetic
speech is generated separately by ml/scripts/generate_tts_samples.py.

    python ml/scripts/prepare_eval_set.py --librispeech ml/data/librispeech

Neither the corpus nor the generated audio is committed to this repository.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = REPO_ROOT / "ml" / "data" / "eval"

#: Only take clips at least this long, so the model scores real speech rather
#: than tiling padding.
MIN_SECONDS = 4.5


def duration_of(path: Path) -> float:
    """Returns a clip's duration in seconds without decoding it fully."""
    info = sf.info(str(path))
    return info.frames / info.samplerate


def collect_genuine(corpus: Path, destination: Path, count: int, min_seconds: float) -> list[Path]:
    """Copies the first ``count`` long-enough clips, one per speaker."""
    destination.mkdir(parents=True, exist_ok=True)
    chosen: list[Path] = []
    seen_speakers: set[str] = set()

    for flac in sorted(corpus.rglob("*.flac")):
        # LibriSpeech names files <speaker>-<chapter>-<utterance>.flac.
        speaker = flac.stem.split("-")[0]
        if speaker in seen_speakers:
            continue
        try:
            if duration_of(flac) < min_seconds:
                continue
        except (sf.LibsndfileError, RuntimeError):
            continue
        target = destination / f"librispeech_{flac.stem}.flac"
        shutil.copy2(flac, target)
        chosen.append(target)
        seen_speakers.add(speaker)
        if len(chosen) >= count:
            break
    return chosen


def write_manifest(eval_dir: Path) -> tuple[Path, int, int]:
    """Writes manifest.csv covering everything under the eval directory."""
    rows: list[tuple[str, str]] = []
    for label, subdirectory in (("bonafide", "genuine"), ("spoof", "synthetic")):
        directory = eval_dir / subdirectory
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in {".wav", ".flac"}:
                rows.append((str(path.relative_to(eval_dir)), label))

    manifest = eval_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    genuine = sum(1 for _, label in rows if label == "bonafide")
    return manifest, genuine, len(rows) - genuine


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--librispeech", type=Path, help="extracted LibriSpeech directory to draw genuine samples from")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="evaluation set directory")
    parser.add_argument("--count", type=int, default=12, help="how many genuine samples to copy")
    parser.add_argument("--min-seconds", type=float, default=MIN_SECONDS, help="minimum clip duration to accept")
    args = parser.parse_args()

    if args.librispeech:
        if not args.librispeech.is_dir():
            raise SystemExit(f"not a directory: {args.librispeech}")
        chosen = collect_genuine(args.librispeech, args.eval_dir / "genuine", args.count, args.min_seconds)
        print(f"Copied {len(chosen)} genuine samples (one per speaker, >= {args.min_seconds}s)")
        for path in chosen:
            print(f"  {path.name}  {duration_of(path):.2f}s")

    manifest, genuine, spoof = write_manifest(args.eval_dir)
    print(f"\nManifest: {manifest}")
    print(f"  bonafide {genuine}")
    print(f"  spoof    {spoof}")
    if min(genuine, spoof) < 50:
        print("\nNote: this is a small set. Treat any metric from it as illustrative, not as accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
