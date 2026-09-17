#!/usr/bin/env python3
"""Rebuild the speaker-verification evaluation audio from the committed manifest.

The audio itself is never committed. This script reconstructs every sample from
ml/evaluation/sv_manifest.json and verifies each one, so another engineer gets
the same inputs the published numbers were measured on.

    # one-shot: fetch the corpora and the voice, then rebuild and verify
    python ml/scripts/prepare_sv_eval_set.py --download

    # LibriSpeech already extracted somewhere
    python ml/scripts/prepare_sv_eval_set.py --librispeech ml/data/librispeech

    # check an existing rebuild without regenerating
    python ml/scripts/prepare_sv_eval_set.py --verify-only

Three kinds of sample, verified by the rule each one deserves:

    librispeech  byte copies of fixed corpus files      -> strict SHA-256
    cmu_arctic   byte copies of fixed corpus files      -> strict SHA-256
    piper_tts    re-synthesised on this machine         -> tolerant canonical PCM

The tolerant rule and why byte equality is the wrong gate for Piper output are
documented in ml/audio/fingerprint.py, carried over unchanged from ROHAN-002.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.error
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
DEFAULT_MANIFEST = REPO_ROOT / "ml" / "evaluation" / "sv_manifest.json"
DEFAULT_EVAL_DIR = REPO_ROOT / "ml" / "data" / "sv-eval"
DEFAULT_CORPUS_DIR = REPO_ROOT / "ml" / "data" / "librispeech"
DEFAULT_VOICE_DIR = REPO_ROOT / "ml" / "data" / "tts-voices"

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"

# festvox.org serves CMU ARCTIC over plain HTTP only; it has no HTTPS listener.
# Integrity therefore rests entirely on the per-file SHA-256 in the manifest,
# which is checked on every run and is what actually gates use of the audio.
ARCTIC_BASE = "http://festvox.org/cmu_arctic/cmu_arctic"

VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium"
VOICE_NAME = "en_US-arctic-medium.onnx"

#: Where each source type's audio lands under the evaluation directory.
SUBDIR = {"librispeech": "librispeech", "cmu_arctic": "arctic", "piper_tts": "clone"}


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
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=1800) as response, target.open("wb") as handle:  # noqa: S310 - fixed URLs, hash-gated
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, TimeoutError) as exc:
        target.unlink(missing_ok=True)
        raise SystemExit(f"download failed for {url}: {exc}") from exc


def arctic_url(speaker: str, utterance: str) -> str:
    """Returns the download URL for one CMU ARCTIC utterance."""
    return f"{ARCTIC_BASE}/cmu_us_{speaker}_arctic/wav/{utterance}.wav"


def download_corpus(corpus_dir: Path) -> None:
    """Fetches and extracts LibriSpeech dev-clean if it is not already there."""
    if (corpus_dir / "dev-clean").is_dir():
        print(f"  present  {corpus_dir}/dev-clean")
        return
    archive = corpus_dir.parent / "dev-clean.tar.gz"
    print(f"  fetching dev-clean.tar.gz <- {LIBRISPEECH_URL}")
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


#: Prompt text and licence for the ARCTIC corpus. The prompt list is identical
#: across speakers, so one copy serves all of them.
ARCTIC_METADATA = {
    "txt.done.data": "cmu_us_slt_arctic/etc/txt.done.data",
    "COPYING": "cmu_us_slt_arctic/COPYING",
}


def download_arctic(samples: list[dict], arctic_dir: Path) -> None:
    """Fetches only the CMU ARCTIC utterances the manifest actually names."""
    arctic_dir.mkdir(parents=True, exist_ok=True)
    for name, relpath in ARCTIC_METADATA.items():
        fetch(f"{ARCTIC_BASE}/{relpath}", arctic_dir / name)
    fetched = 0
    for sample in samples:
        target = arctic_dir / sample["source_relpath"]
        if target.exists():
            continue
        fetch(arctic_url(sample["speaker_id"], sample["source_utterance"]), target)
        fetched += 1
    print(f"  fetched  {fetched} CMU ARCTIC utterances ({len(samples) - fetched} already present)")


def arctic_prompts(arctic_dir: Path) -> dict[str, str]:
    """Parses the ARCTIC prompt file into {utterance id: text}.

    Reading the text from the corpus rather than transcribing it is what makes a
    clone provably say the same words as the genuine probe it is compared with.
    """
    prompts: dict[str, str] = {}
    for line in (arctic_dir / "txt.done.data").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("("):
            continue
        head, _, rest = line.partition('"')
        text, _, _ = rest.rpartition('"')
        prompts[head.strip("( ").strip()] = text
    return prompts


def download_voice(voice_dir: Path) -> Path:
    """Fetches the Piper voice model and its config."""
    voice = voice_dir / VOICE_NAME
    fetch(f"{VOICE_BASE}/{VOICE_NAME}", voice)
    # PiperVoice.load looks for <model>.json beside the model, not <stem>.json.
    fetch(f"{VOICE_BASE}/{VOICE_NAME}.json", voice_dir / f"{VOICE_NAME}.json")
    return voice


def verify_arctic_metadata(arctic_dir: Path, manifest: dict) -> None:
    """Checks the ARCTIC prompt file and licence against the manifest, every run.

    The prompt file decides what every clone says, so a different one silently
    produces a different evaluation set.
    """
    for name, expected in manifest.get("arctic_metadata_sha256", {}).items():
        path = arctic_dir / name
        if not path.is_file():
            raise SystemExit(f"CMU ARCTIC {name} not found at {path}. Re-run with --download.")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"CMU ARCTIC {name} checksum mismatch\n  expected {expected}\n  got      {actual}")
        print(f"  verified ARCTIC {name}")


def verify_voice(voice_path: Path, sample: dict) -> None:
    """Checks the voice model and config against the manifest, every run.

    A different voice or config silently produces different speech, so this is
    checked unconditionally rather than only on the download path.
    """
    config_path = voice_path.with_suffix(voice_path.suffix + ".json")
    for path, key, label in ((voice_path, "voice_sha256", "voice model"), (config_path, "voice_config_sha256", "voice config")):
        expected = sample.get(key)
        if expected is None:
            raise SystemExit(f"manifest is missing {key}; rebuild it with build_sv_manifest.py")
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"{label} checksum mismatch for {path.name}\n  expected {expected}\n  got      {actual}\nThe evaluation set cannot be reproduced with a different voice.")
        print(f"  verified {label}: {path.name}")


def synthesise(voice, sample: dict) -> bytes:
    """Re-synthesises one Piper sample exactly as the manifest records it."""
    from piper import SynthesisConfig

    synthesis = sample["synthesis"]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        voice.synthesize_wav(
            sample["text"],
            handle,
            syn_config=SynthesisConfig(
                speaker_id=sample["voice_speaker_index"],
                length_scale=synthesis["length_scale"],
                noise_scale=synthesis["noise_scale"],
                noise_w_scale=synthesis["noise_w_scale"],
            ),
        )
    return buffer.getvalue()


def load_voice(voice_path: Path):
    """Loads the Piper voice, with a usable error when piper is missing."""
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise SystemExit(f"piper is required to rebuild clone samples: {exc}\nInstall it with: pip install -r ml/requirements.txt") from exc
    return PiperVoice.load(voice_path)


def rebuild_corpus_samples(samples: list[dict], source_root: Path, eval_dir: Path, kind: str) -> list[tuple[str, str]]:
    """Copies each corpus sample into the evaluation tree and hash-checks the source."""
    destination = eval_dir / SUBDIR[kind]
    destination.mkdir(parents=True, exist_ok=True)
    problems: list[tuple[str, str]] = []

    for sample in samples:
        source = source_root / sample["source_relpath"]
        if not source.is_file():
            problems.append((sample["sample_id"], f"missing from corpus: {sample['source_relpath']}"))
            continue
        actual = sha256_file(source)
        if actual != sample["sha256"]:
            problems.append((sample["sample_id"], f"corpus file hash mismatch (expected {sample['sha256'][:12]}, got {actual[:12]})"))
            continue
        shutil.copy2(source, destination / sample["filename"])
    return problems


def rebuild_clones(samples: list[dict], voice_path: Path, eval_dir: Path) -> list[tuple[str, str]]:
    """Re-synthesises every clone sample from its recorded speaker and text."""
    destination = eval_dir / SUBDIR["piper_tts"]
    destination.mkdir(parents=True, exist_ok=True)
    voice = load_voice(voice_path)
    for sample in samples:
        (destination / sample["filename"]).write_bytes(synthesise(voice, sample))
    return []


def verify_strict(sample: dict, path: Path) -> str | None:
    """Strict byte equality. These are copies of fixed corpus files."""
    actual = sha256_file(path)
    if actual != sample["sha256"]:
        return f"SHA-256 mismatch (expected {sample['sha256'][:12]}, got {actual[:12]})"
    return None


def verify_tolerant(sample: dict, path: Path, tolerance_db: float) -> tuple[str | None, float]:
    """Tolerant canonical-PCM check for regenerated Piper audio.

    Format and length must match exactly; the energy envelope must match within
    the measured tolerance. Full-file SHA-256 is reported but not gating,
    because Piper output is not bit-reproducible across ONNX Runtime execution
    plans (see ml/audio/fingerprint.py).
    """
    expected_envelope = sample.get("pcm_envelope_db")
    if expected_envelope is None:
        return "manifest has no pcm_envelope_db; rebuild it with build_sv_manifest.py", float("inf")

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
    """Verifies every prepared sample by the rule appropriate to its source."""
    problems: list[tuple[str, str]] = []
    stats = {"librispeech": 0, "cmu_arctic": 0, "piper_tts": 0, "worst_envelope_db": 0.0, "bit_identical": 0}

    for sample in samples:
        kind = sample["source_type"]
        path = eval_dir / SUBDIR[kind] / sample["filename"]
        if not path.is_file():
            problems.append((sample["sample_id"], "not prepared"))
            continue

        if kind != "piper_tts":
            failure = verify_strict(sample, path)
            if failure:
                problems.append((sample["sample_id"], failure))
            else:
                stats[kind] += 1
            continue

        failure, distance = verify_tolerant(sample, path, tolerance_db)
        if failure:
            problems.append((sample["sample_id"], failure))
            continue
        stats[kind] += 1
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
    parser.add_argument("--arctic", type=Path, default=REPO_ROOT / "ml" / "data" / "cmu-arctic", help="CMU ARCTIC download cache")
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE_DIR / VOICE_NAME, help="Piper .onnx voice model")
    parser.add_argument("--download", action="store_true", help="fetch the corpora and voice first (~420 MB)")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing rebuild without regenerating")
    parser.add_argument("--tolerance-db", type=float, default=CANONICAL_TOLERANCE_DB, help=f"canonical-PCM envelope tolerance in dB (default {CANONICAL_TOLERANCE_DB})")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    by_kind = {kind: [s for s in samples if s["source_type"] == kind] for kind in SUBDIR}
    clones = by_kind["piper_tts"]

    if args.verify_only:
        verify_arctic_metadata(args.arctic, manifest)
        verify_voice(args.voice, clones[0])
        problems, stats = verify(samples, args.eval_dir, args.tolerance_db)
    else:
        if args.download:
            print("Fetching sources:")
            download_corpus(args.librispeech)
            download_arctic(by_kind["cmu_arctic"], args.arctic)
            download_voice(args.voice.parent)
            print()

        print("Verifying generator inputs:")
        verify_arctic_metadata(args.arctic, manifest)
        verify_voice(args.voice, clones[0])

        print(f"\nRebuilding {len(by_kind['librispeech'])} LibriSpeech samples from {args.librispeech}...")
        problems = rebuild_corpus_samples(by_kind["librispeech"], args.librispeech, args.eval_dir, "librispeech")
        print(f"Rebuilding {len(by_kind['cmu_arctic'])} CMU ARCTIC samples from {args.arctic}...")
        problems += rebuild_corpus_samples(by_kind["cmu_arctic"], args.arctic, args.eval_dir, "cmu_arctic")
        print(f"Re-synthesising {len(clones)} clone samples with {args.voice.name}...")
        problems += rebuild_clones(clones, args.voice, args.eval_dir)
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
    print(f"  LibriSpeech  {stats['librispeech']:>3}/{len(by_kind['librispeech'])}  strict SHA-256")
    print(f"  CMU ARCTIC   {stats['cmu_arctic']:>3}/{len(by_kind['cmu_arctic'])}  strict SHA-256")
    print(f"  Piper clone  {stats['piper_tts']:>3}/{len(clones)}  tolerant canonical PCM "
          f"(worst {stats['worst_envelope_db']:.6f} dB of {args.tolerance_db} dB allowed)")
    print(f"  Total        {sum(stats[k] for k in SUBDIR):>3}/{len(samples)}")
    print(f"\n  {stats['bit_identical']}/{len(clones)} clone samples also happened to be bit-identical "
          f"(informational; not required across machines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
