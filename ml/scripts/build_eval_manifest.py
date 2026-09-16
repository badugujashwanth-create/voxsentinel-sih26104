#!/usr/bin/env python3
"""Author the evaluation manifest. Run once; the output is committed.

Most people never run this. It records, in executable form, exactly how the
80-sample evaluation set was chosen, so the committed manifest is auditable
rather than hand-written. To rebuild the set on another machine, run
``prepare_eval_set.py`` against the committed manifest instead.

    python ml/scripts/build_eval_manifest.py \
        --librispeech ml/data/librispeech \
        --voice ml/data/tts-voices/en_US-libritts_r-medium.onnx

Requires both the corpus and Piper. Overwrites ml/evaluation/eval_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import soundfile as sf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "ml" / "evaluation" / "eval_manifest.json"

GENUINE_COUNT = 40
SYNTHETIC_COUNT = 40

#: Genuine selection rule, applied to LibriSpeech dev-clean and recorded in the
#: manifest so it can be checked rather than trusted.
MIN_GENUINE_SECONDS = 4.5
GENUINE_RULE = (
    "Sort every *.flac under dev-clean by POSIX path. Walking that order, take "
    f"the first utterance per speaker whose duration is >= {MIN_GENUINE_SECONDS}s. "
    f"Stop at {GENUINE_COUNT} speakers. One utterance per speaker."
)

#: Piper voice speakers cycled through for the synthetic half.
SYNTHETIC_SPEAKERS = (0, 7, 19, 42, 63, 101, 137, 188, 256, 311, 402, 513, 604, 712, 803, 877)

#: Prompts for the synthetic half. Long enough that the detector scores real
#: speech rather than tiling padding.
SENTENCES: tuple[str, ...] = (
    "Good morning, this is the finance department calling about the quarterly vendor reconciliation that we discussed last week.",
    "I need you to process an urgent transfer to our new supplier account before the close of business today, please.",
    "The board meeting has been moved to Thursday afternoon, so please reschedule the audit review accordingly and inform the team.",
    "Please confirm the account details before releasing any funds, as we have had several irregular requests this month.",
    "I am travelling at the moment and cannot access the portal, so I would like you to authorise this payment on my behalf.",
    "The compliance report needs to be submitted by the end of the week, along with the supporting transaction documentation.",
    "Could you please verify the beneficiary name and the routing number one more time before you approve the payment request.",
    "We have received the invoice from the contractor and everything appears to be in order for the scheduled disbursement.",
    "This is a routine call to confirm that the wire instructions we sent earlier this morning have reached your department.",
    "The treasury team has flagged an unusual pattern in last quarter's outgoing payments and would like a full reconciliation.",
    "Thank you for taking my call, I wanted to discuss the outstanding balance on the operational expenses account with you.",
    "Our external auditors have requested additional documentation covering the vendor onboarding process from the last year.",
)

#: Piper's defaults sample noise inside the ONNX graph, so repeated synthesis of
#: the same text produces different audio. Zeroing both noise scales makes
#: generation bit-reproducible, which is the whole point of this manifest.
SYNTHESIS_NOISE_SCALE = 0.0
SYNTHESIS_NOISE_W_SCALE = 0.0


def sha256_bytes(payload: bytes) -> str:
    """Returns the hex SHA-256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Returns the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_plan() -> list[tuple[int, int, int, float]]:
    """Returns (index, text_id, speaker_id, length_scale) for each sample.

    Text and speaker advance at different strides so no pair repeats.
    """
    plan = []
    for index in range(SYNTHETIC_COUNT):
        text_id = index % len(SENTENCES)
        speaker = SYNTHETIC_SPEAKERS[(index // len(SENTENCES) + index) % len(SYNTHETIC_SPEAKERS)]
        length_scale = round(1.0 + 0.05 * ((index % 3) - 1), 2)
        plan.append((index, text_id, speaker, length_scale))
    return plan


def build_genuine(corpus: Path) -> list[dict]:
    """Applies the selection rule and records full provenance per sample."""
    root = corpus / "dev-clean" if (corpus / "dev-clean").is_dir() else corpus
    samples: list[dict] = []
    seen: set[str] = set()

    for flac in sorted(root.rglob("*.flac")):
        speaker, chapter, utterance = flac.stem.split("-")
        if speaker in seen:
            continue
        info = sf.info(str(flac))
        duration = info.frames / info.samplerate
        if duration < MIN_GENUINE_SECONDS:
            continue
        samples.append({
            "sample_id": f"genuine_{len(samples):03d}",
            "label": "bonafide",
            "source_type": "librispeech",
            "corpus": "LibriSpeech dev-clean",
            "corpus_url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
            "license": "CC BY 4.0",
            "license_note": "Read speech from public-domain LibriVox audiobooks; corpus released under CC BY 4.0 (OpenSLR resource 12).",
            "speaker_id": speaker,
            "chapter_id": chapter,
            "utterance_id": utterance,
            "source_utterance": flac.stem,
            "source_relpath": str(flac.relative_to(root.parent) if root.name == "dev-clean" else flac.relative_to(root)),
            "filename": f"librispeech_{flac.stem}.flac",
            "sample_rate": info.samplerate,
            "duration_seconds": round(duration, 4),
            "sha256": sha256_file(flac),
        })
        seen.add(speaker)
        if len(samples) >= GENUINE_COUNT:
            break

    if len(samples) < GENUINE_COUNT:
        raise SystemExit(f"only found {len(samples)} qualifying genuine samples, need {GENUINE_COUNT}")
    return samples


def build_synthetic(voice_path: Path) -> list[dict]:
    """Synthesises each planned sample and records how to reproduce it."""
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError as exc:
        raise SystemExit(f"piper is required to author the manifest: {exc}\nSee ml/README.md for the install command.") from exc

    import importlib.metadata as metadata

    voice = PiperVoice.load(voice_path)
    piper_version = metadata.version("piper-tts")
    samples: list[dict] = []

    for index, text_id, speaker, length_scale in synthetic_plan():
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            voice.synthesize_wav(
                SENTENCES[text_id],
                handle,
                syn_config=SynthesisConfig(
                    speaker_id=speaker,
                    length_scale=length_scale,
                    noise_scale=SYNTHESIS_NOISE_SCALE,
                    noise_w_scale=SYNTHESIS_NOISE_W_SCALE,
                ),
            )
        payload = buffer.getvalue()
        with sf.SoundFile(io.BytesIO(payload)) as handle:
            duration = handle.frames / handle.samplerate
            sample_rate = handle.samplerate

        samples.append({
            "sample_id": f"synthetic_{index:03d}",
            "label": "spoof",
            "source_type": "piper_tts",
            "generator": "piper",
            "generator_version": piper_version,
            "voice": "en_US-libritts_r-medium",
            "voice_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
            "voice_sha256": sha256_file(voice_path),
            "license": "CC BY 4.0",
            "license_note": "Voice trained on LibriTTS-R (CC BY 4.0). Piper itself is GPL-3.0 and is a generation tool only, not a dependency of the detector.",
            "speaker_id": speaker,
            "text_id": text_id,
            "text": SENTENCES[text_id],
            "synthesis": {
                "length_scale": length_scale,
                "noise_scale": SYNTHESIS_NOISE_SCALE,
                "noise_w_scale": SYNTHESIS_NOISE_W_SCALE,
            },
            "filename": f"tts_{index:03d}_spk{speaker:03d}.wav",
            "sample_rate": sample_rate,
            "duration_seconds": round(duration, 4),
            "sha256": sha256_bytes(payload),
        })
    return samples


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--librispeech", required=True, type=Path, help="extracted LibriSpeech dev-clean directory")
    parser.add_argument("--voice", required=True, type=Path, help="Piper .onnx voice model")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST, help="manifest path to write")
    args = parser.parse_args()

    print("Selecting genuine samples...")
    genuine = build_genuine(args.librispeech)
    print(f"  {len(genuine)} samples, {len({s['speaker_id'] for s in genuine})} distinct speakers")

    print("Synthesising samples...")
    synthetic = build_synthetic(args.voice)
    print(f"  {len(synthetic)} samples, {len({s['speaker_id'] for s in synthetic})} distinct voices")

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": (
            "VoxSentinel ROHAN-002 spoof-detection evaluation set: 40 genuine and 40 synthetic "
            "samples. Audio is NOT committed. Rebuild it with ml/scripts/prepare_eval_set.py, "
            "which regenerates every sample from this manifest and verifies each SHA-256."
        ),
        "genuine_selection_rule": GENUINE_RULE,
        "genuine_min_seconds": MIN_GENUINE_SECONDS,
        "synthetic_generation_note": (
            "Piper's default noise_scale and noise_w_scale sample inside the ONNX graph, so the "
            "same text yields different audio on every run. Both are pinned to 0.0 here, which "
            "makes synthesis bit-reproducible. Seeding numpy does NOT help."
        ),
        "counts": {"bonafide": len(genuine), "spoof": len(synthetic), "total": len(genuine) + len(synthetic)},
        "samples": genuine + synthetic,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output} ({manifest['counts']['total']} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
