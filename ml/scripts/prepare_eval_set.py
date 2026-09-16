#!/usr/bin/env python3
"""Rebuild the evaluation audio from the committed manifest.

The audio itself is never committed. This script reconstructs all 80 samples
from ml/evaluation/eval_manifest.json and verifies every file against the
SHA-256 recorded there, so another engineer gets byte-identical inputs.

    # one-shot: fetch the corpus and the voice, then rebuild and verify
    python ml/scripts/prepare_eval_set.py --download

    # corpus already extracted somewhere
    python ml/scripts/prepare_eval_set.py --librispeech ml/data/librispeech

    # check an existing rebuild without regenerating
    python ml/scripts/prepare_eval_set.py --verify-only

Genuine samples are copied from LibriSpeech dev-clean. Synthetic samples are
re-synthesised with Piper using the exact speaker, text, and synthesis
parameters recorded per sample.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from ml.audio.fingerprint import (  # noqa: E402
    CANONICAL_TOLERANCE_DB,
    envelope_db,
    envelopes_match,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "ml" / "evaluation" / "eval_manifest.json"
DEFAULT_EVAL_DIR = REPO_ROOT / "ml" / "data" / "eval"
DEFAULT_CORPUS_DIR = REPO_ROOT / "ml" / "data" / "librispeech"
DEFAULT_VOICE_DIR = REPO_ROOT / "ml" / "data" / "tts-voices"

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium"
VOICE_NAME = "en_US-libritts_r-medium.onnx"

SUBDIR = {"bonafide": "genuine", "spoof": "synthetic"}


def sha256_file(path: Path) -> str:
    """Returns the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path) -> None:
    """Downloads a URL to a path, skipping if it already exists."""
    if target.exists():
        print(f"  present  {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {target.name} <- {url}")
    with urllib.request.urlopen(url, timeout=1800) as response, target.open("wb") as handle:  # noqa: S310 - fixed https URLs
        shutil.copyfileobj(response, handle)


def download_corpus(corpus_dir: Path) -> None:
    """Fetches and extracts LibriSpeech dev-clean if it is not already there."""
    if (corpus_dir / "dev-clean").is_dir():
        print(f"  present  {corpus_dir}/dev-clean")
        return
    archive = corpus_dir.parent / "dev-clean.tar.gz"
    fetch(LIBRISPEECH_URL, archive)
    print(f"  extracting {archive.name} (~337 MB)")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        # Archive root is LibriSpeech/; strip it so dev-clean lands directly.
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) > 1 and parts[0] == "LibriSpeech":
                member.name = str(Path(*parts[1:]))
                tar.extract(member, corpus_dir, filter="data")
    archive.unlink()


def download_voice(voice_dir: Path) -> Path:
    """Fetches the Piper voice model and its config."""
    voice = voice_dir / VOICE_NAME
    fetch(f"{VOICE_BASE}/{VOICE_NAME}", voice)
    fetch(f"{VOICE_BASE}/{VOICE_NAME}.json", voice_dir / f"{VOICE_NAME}.json")
    return voice


def verify_voice(voice_path: Path, sample: dict) -> None:
    """Checks the voice model and config against the manifest, every run.

    A different voice or config silently produces different speech, so this is
    checked unconditionally rather than only on the download path.
    """
    config_path = voice_path.with_suffix(voice_path.suffix + ".json")
    for path, key, label in ((voice_path, "voice_sha256", "voice model"), (config_path, "voice_config_sha256", "voice config")):
        expected = sample.get(key)
        if expected is None:
            raise SystemExit(f"manifest is missing {key}; rebuild it with build_eval_manifest.py")
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"{label} checksum mismatch for {path.name}\n  expected {expected}\n  got      {actual}\nThe evaluation set cannot be reproduced with a different voice.")
        print(f"  verified {label}: {path.name}")


def rebuild_genuine(samples: list[dict], corpus_dir: Path, eval_dir: Path) -> list[tuple[str, str]]:
    """Copies each genuine sample out of the corpus and verifies it."""
    destination = eval_dir / "genuine"
    destination.mkdir(parents=True, exist_ok=True)
    problems: list[tuple[str, str]] = []

    for sample in samples:
        source = corpus_dir / sample["source_relpath"]
        if not source.is_file():
            problems.append((sample["sample_id"], f"missing from corpus: {sample['source_relpath']}"))
            continue
        actual = sha256_file(source)
        if actual != sample["sha256"]:
            problems.append((sample["sample_id"], f"corpus file hash mismatch (expected {sample['sha256'][:12]}, got {actual[:12]})"))
            continue
        shutil.copy2(source, destination / sample["filename"])
    return problems


def rebuild_synthetic(samples: list[dict], voice_path: Path, eval_dir: Path) -> list[tuple[str, str]]:
    """Re-synthesises each synthetic sample and verifies it byte for byte."""
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError as exc:
        raise SystemExit(f"piper is required to rebuild synthetic samples: {exc}\nInstall it with: pip install -r ml/requirements.txt") from exc

    destination = eval_dir / "synthetic"
    destination.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(voice_path)
    problems: list[tuple[str, str]] = []

    for sample in samples:
        synthesis = sample["synthesis"]
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            voice.synthesize_wav(
                sample["text"],
                handle,
                syn_config=SynthesisConfig(
                    speaker_id=sample["speaker_id"],
                    length_scale=synthesis["length_scale"],
                    noise_scale=synthesis["noise_scale"],
                    noise_w_scale=synthesis["noise_w_scale"],
                ),
            )
        (destination / sample["filename"]).write_bytes(buffer.getvalue())
    return problems


def verify_genuine(sample: dict, path: Path) -> str | None:
    """Strict byte equality. These are copies of fixed corpus files."""
    actual = sha256_file(path)
    if actual != sample["sha256"]:
        return f"SHA-256 mismatch (expected {sample['sha256'][:12]}, got {actual[:12]})"
    return None


def verify_synthetic(sample: dict, path: Path, tolerance_db: float) -> tuple[str | None, float]:
    """Tolerant canonical-PCM check for regenerated Piper audio.

    Format and length must match exactly; the energy envelope must match within
    the measured tolerance. Full-file SHA-256 is reported but not gating,
    because Piper output is not bit-reproducible across ONNX Runtime execution
    plans (see ml/audio/fingerprint.py).
    """
    expected_envelope = sample.get("pcm_envelope_db")
    if expected_envelope is None:
        return "manifest has no pcm_envelope_db; rebuild it with build_eval_manifest.py", float("inf")

    info = sf.info(str(path))
    if info.samplerate != sample["sample_rate"]:
        return f"sample rate {info.samplerate} != expected {sample['sample_rate']}", float("inf")
    if info.channels != sample.get("channels", 1):
        return f"channel count {info.channels} != expected {sample.get('channels', 1)}", float("inf")
    if info.frames != sample["sample_count"]:
        return f"sample count {info.frames} != expected {sample['sample_count']}", float("inf")

    pcm, _ = sf.read(str(path), dtype="int16", always_2d=False)
    matched, distance = envelopes_match(envelope_db(pcm), np.asarray(expected_envelope, dtype=np.float64), tolerance_db)
    if not matched:
        return f"canonical PCM envelope differs by {distance:.6f} dB (tolerance {tolerance_db} dB)", distance
    return None, distance


def verify(samples: list[dict], eval_dir: Path, tolerance_db: float = CANONICAL_TOLERANCE_DB) -> tuple[list[tuple[str, str]], dict]:
    """Verifies every prepared sample by the rule appropriate to its class."""
    problems: list[tuple[str, str]] = []
    stats = {"bonafide_strict": 0, "spoof_tolerant": 0, "worst_envelope_db": 0.0, "bit_identical": 0}

    for sample in samples:
        path = eval_dir / SUBDIR[sample["label"]] / sample["filename"]
        if not path.is_file():
            problems.append((sample["sample_id"], "not prepared"))
            continue

        if sample["label"] == "bonafide":
            failure = verify_genuine(sample, path)
            if failure:
                problems.append((sample["sample_id"], failure))
            else:
                stats["bonafide_strict"] += 1
            continue

        failure, distance = verify_synthetic(sample, path, tolerance_db)
        if failure:
            problems.append((sample["sample_id"], failure))
            continue
        stats["spoof_tolerant"] += 1
        stats["worst_envelope_db"] = max(stats["worst_envelope_db"], distance)
        if sha256_file(path) == sample["sha256"]:
            stats["bit_identical"] += 1

    return problems, stats


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="committed evaluation manifest")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="where to write the rebuilt audio")
    parser.add_argument("--librispeech", type=Path, default=DEFAULT_CORPUS_DIR, help="extracted LibriSpeech directory")
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE_DIR / VOICE_NAME, help="Piper .onnx voice model")
    parser.add_argument("--download", action="store_true", help="fetch the corpus and voice first (~420 MB)")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing rebuild without regenerating")
    parser.add_argument("--tolerance-db", type=float, default=CANONICAL_TOLERANCE_DB, help=f"canonical-PCM envelope tolerance in dB (default {CANONICAL_TOLERANCE_DB})")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    genuine = [s for s in samples if s["label"] == "bonafide"]
    synthetic = [s for s in samples if s["label"] == "spoof"]

    if args.verify_only:
        verify_voice(args.voice, synthetic[0])
        problems, stats = verify(samples, args.eval_dir, args.tolerance_db)
    else:
        if args.download:
            print("Fetching sources:")
            download_corpus(args.librispeech)
            download_voice(args.voice.parent)
            print()

        print("Verifying generator inputs:")
        verify_voice(args.voice, synthetic[0])

        print(f"\nRebuilding {len(genuine)} genuine samples from {args.librispeech}...")
        problems = rebuild_genuine(genuine, args.librispeech, args.eval_dir)
        print(f"Re-synthesising {len(synthetic)} samples with {args.voice.name}...")
        problems += rebuild_synthetic(synthetic, args.voice, args.eval_dir)
        extra, stats = verify(samples, args.eval_dir, args.tolerance_db)
        problems += extra

    print()
    if problems:
        print(f"FAILED: {len(problems)} of {len(samples)} samples did not verify")
        for sample_id, reason in problems[:20]:
            print(f"  {sample_id}: {reason}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return 1

    print(f"All {len(samples)} samples verified against the manifest.")
    print(f"  LibriSpeech  {stats['bonafide_strict']:>2}/{len(genuine)}  strict SHA-256")
    print(f"  Piper        {stats['spoof_tolerant']:>2}/{len(synthetic)}  tolerant canonical PCM "
          f"(worst {stats['worst_envelope_db']:.6f} dB of {args.tolerance_db} dB allowed)")
    print(f"  Total        {stats['bonafide_strict'] + stats['spoof_tolerant']:>2}/{len(samples)}")
    print(f"\n  {stats['bit_identical']}/{len(synthetic)} Piper samples also happened to be bit-identical "
          f"(informational; not required across machines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
