#!/usr/bin/env python3
"""Author the speaker-verification evaluation manifest.

Run once; the output is committed. This is not part of normal reproduction --
ml/scripts/prepare_sv_eval_set.py rebuilds the audio from what this writes.

It is committed so the selection rules are executable rather than asserted: the
speakers, utterances, trial pairs, and clone texts below are produced by code
anyone can re-run, not by a list someone typed.

    python ml/scripts/build_sv_manifest.py --download

The set has three parts.

LibriSpeech dev-clean  the same-speaker / different-speaker trials, and the
                       set the decision threshold is fitted on. Enrolment and
                       probe utterances come from DIFFERENT chapters of the
                       same speaker, because two utterances recorded minutes
                       apart in one session share channel and microphone
                       conditions, and a verifier can score those as similar
                       without having learned anything about the voice.

CMU ARCTIC             a second genuine corpus, used for the clone comparison
                       so the clone and its genuine counterpart come from the
                       same recording conditions.

Piper en_US-arctic     the clone attack. That voice is trained on CMU ARCTIC
                       speakers, so synthesising with speaker "slt" produces
                       speech imitating the same person the ARCTIC "slt"
                       recordings enrol. This is the only reason the ARCTIC
                       corpus is here: Piper's en_US-libritts_r voice, used by
                       ROHAN-002, shares ZERO speakers with LibriSpeech
                       dev-clean, so none of that synthetic audio is a clone of
                       anyone we can enrol.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import soundfile as sf  # noqa: E402

from ml.audio.fingerprint import (  # noqa: E402
    CANONICAL_FLOOR_DB,
    CANONICAL_FRAME_SAMPLES,
    CANONICAL_TOLERANCE_DB,
    fingerprint,
)
from ml.scripts.prepare_sv_eval_set import (  # noqa: E402
    ARCTIC_METADATA,
    DEFAULT_CORPUS_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_VOICE_DIR,
    VOICE_NAME,
    arctic_prompts,
    arctic_url,
    download_arctic,
    download_corpus,
    download_voice,
    load_voice,
    sha256_file,
    synthesise,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCTIC_DIR = REPO_ROOT / "ml" / "data" / "cmu-arctic"

# --- LibriSpeech selection rule ---------------------------------------------

#: An utterance shorter than this carries too little speech to characterise a
#: voice; the same floor ROHAN-002 used for its genuine samples.
LIBRISPEECH_MIN_SECONDS = 4.0
LIBRISPEECH_ENROLMENT_UTTERANCES = 2
LIBRISPEECH_PROBE_UTTERANCES = 2

#: How many other speakers each reference is tested against. Every speaker gets
#: the same number of impostors, taken at fixed offsets in the sorted speaker
#: list, so the negative set is balanced and does not depend on a random seed.
IMPOSTOR_OFFSETS = (1, 2, 3, 4)

# --- CMU ARCTIC selection rule ----------------------------------------------

#: Six ARCTIC speakers the Piper en_US-arctic voice can also synthesise, chosen
#: for accent and sex spread: clb and slt are female US, bdl and rms male US,
#: awb male Scottish, ksp male Indian.
ARCTIC_SPEAKERS = ("awb", "bdl", "clb", "ksp", "rms", "slt")
ARCTIC_ENROLMENT_UTTERANCES = ("arctic_a0001", "arctic_a0002", "arctic_a0003", "arctic_a0004")
ARCTIC_PROBE_UTTERANCES = ("arctic_a0011", "arctic_a0012")

# A clone says exactly what the genuine probe says: the prompt text is read out
# of the corpus's own txt.done.data, not transcribed here. ECAPA is
# text-independent, so matching the text removes content as a possible
# explanation for any difference in score.

# --- Piper synthesis --------------------------------------------------------

#: Piper's default noise_scale and noise_w_scale sample inside the ONNX graph,
#: so the same text yields different audio on every run. Both are pinned to 0.0
#: here, which makes synthesis reproducible. Seeding numpy does NOT help.
SYNTHESIS = {"length_scale": 1.0, "noise_scale": 0.0, "noise_w_scale": 0.0}


def qualifying_chapters(speaker_dir: Path, min_seconds: float, needed: int) -> list[tuple[str, list[Path]]]:
    """Returns each chapter with enough long-enough utterances, in sorted order.

    Sorting by numeric chapter id and by POSIX path within a chapter is what
    makes this selection reproducible on any filesystem.
    """
    chapters = []
    for chapter in sorted((c for c in speaker_dir.iterdir() if c.is_dir()), key=lambda c: int(c.name)):
        usable = [f for f in sorted(chapter.glob("*.flac")) if sf.info(f).duration >= min_seconds]
        if len(usable) >= needed:
            chapters.append((chapter.name, usable))
    return chapters


def build_librispeech(corpus_dir: Path) -> tuple[list[dict], list[dict]]:
    """Selects LibriSpeech speakers, utterances, and their samples.

    A speaker is included only if two different chapters each hold enough
    utterances of at least LIBRISPEECH_MIN_SECONDS. Speakers with a single
    chapter in dev-clean are skipped rather than enrolled and probed from the
    same recording session.
    """
    root = corpus_dir / "dev-clean"
    if not root.is_dir():
        raise SystemExit(f"LibriSpeech dev-clean not found at {root}. Re-run with --download.")

    speakers: list[dict] = []
    samples: list[dict] = []
    needed = max(LIBRISPEECH_ENROLMENT_UTTERANCES, LIBRISPEECH_PROBE_UTTERANCES)

    for speaker_dir in sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: int(d.name)):
        chapters = qualifying_chapters(speaker_dir, LIBRISPEECH_MIN_SECONDS, needed)
        if len(chapters) < 2:
            continue

        speaker_id = speaker_dir.name
        enrolment_chapter, enrolment_files = chapters[0]
        probe_chapter, probe_files = chapters[1]
        roles = (
            ("enrolment", enrolment_chapter, enrolment_files[:LIBRISPEECH_ENROLMENT_UTTERANCES]),
            ("probe", probe_chapter, probe_files[:LIBRISPEECH_PROBE_UTTERANCES]),
        )

        entry = {"speaker_key": f"librispeech_{speaker_id}", "corpus": "LibriSpeech dev-clean", "speaker_id": speaker_id, "enrolment": [], "probes": []}
        for role, chapter_id, files in roles:
            for path in files:
                info = sf.info(path)
                sample_id = f"ls_{path.stem}"
                samples.append({
                    "sample_id": sample_id,
                    "source_type": "librispeech",
                    "role": role,
                    "speaker_key": entry["speaker_key"],
                    "corpus": "LibriSpeech dev-clean",
                    "corpus_url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
                    "license": "CC BY 4.0",
                    "license_note": "Read speech from public-domain LibriVox audiobooks; corpus released under CC BY 4.0 (OpenSLR resource 12).",
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "source_utterance": path.stem,
                    "source_relpath": str(path.relative_to(corpus_dir).as_posix()),
                    "filename": f"{sample_id}.flac",
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "duration_seconds": round(info.duration, 4),
                    "sha256": sha256_file(path),
                })
                entry["enrolment" if role == "enrolment" else "probes"].append(sample_id)
        speakers.append(entry)

    return speakers, samples


def build_arctic(arctic_dir: Path) -> tuple[list[dict], list[dict]]:
    """Builds the CMU ARCTIC speakers and their genuine samples."""
    speakers: list[dict] = []
    samples: list[dict] = []

    for speaker in ARCTIC_SPEAKERS:
        entry = {"speaker_key": f"arctic_{speaker}", "corpus": "CMU ARCTIC", "speaker_id": speaker, "enrolment": [], "probes": []}
        for role, utterances in (("enrolment", ARCTIC_ENROLMENT_UTTERANCES), ("probe", ARCTIC_PROBE_UTTERANCES)):
            for utterance in utterances:
                relpath = f"cmu_us_{speaker}_arctic/{utterance}.wav"
                path = arctic_dir / relpath
                if not path.is_file():
                    raise SystemExit(f"CMU ARCTIC utterance not found at {path}. Re-run with --download.")
                info = sf.info(path)
                sample_id = f"arctic_{speaker}_{utterance.removeprefix('arctic_')}"
                samples.append({
                    "sample_id": sample_id,
                    "source_type": "cmu_arctic",
                    "role": role,
                    "speaker_key": entry["speaker_key"],
                    "corpus": "CMU ARCTIC",
                    "corpus_url": "http://festvox.org/cmu_arctic/",
                    "license": "Free for any purpose, commercial or otherwise (CMU ARCTIC licence, BSD-style; see COPYING in each speaker package).",
                    "license_note": "Carnegie Mellon University, (c) 2003. Served over plain HTTP; integrity rests on the SHA-256 below, which prepare_sv_eval_set.py checks on every run.",
                    "speaker_id": speaker,
                    "source_utterance": utterance,
                    "source_url": arctic_url(speaker, utterance),
                    "source_relpath": relpath,
                    "filename": f"{sample_id}.wav",
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "duration_seconds": round(info.duration, 4),
                    "sha256": sha256_file(path),
                })
                entry["enrolment" if role == "enrolment" else "probes"].append(sample_id)
        speakers.append(entry)

    return speakers, samples


def build_clones(voice_path: Path, eval_dir: Path, prompts: dict[str, str]) -> list[dict]:
    """Synthesises one clone per ARCTIC speaker per probe prompt, and records it.

    Written to the evaluation directory here so the manifest carries the exact
    sample count and envelope that prepare_sv_eval_set.py will check against.
    """
    voice = load_voice(voice_path)
    speaker_map = json.loads(voice_path.with_suffix(voice_path.suffix + ".json").read_text(encoding="utf-8"))["speaker_id_map"]
    missing = [s for s in ARCTIC_SPEAKERS if s not in speaker_map]
    if missing:
        raise SystemExit(f"voice {voice_path.name} cannot synthesise ARCTIC speakers {missing}; it is not the en_US-arctic voice")

    voice_sha = sha256_file(voice_path)
    config_sha = sha256_file(voice_path.with_suffix(voice_path.suffix + ".json"))
    destination = eval_dir / "clone"
    destination.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for speaker in ARCTIC_SPEAKERS:
        for utterance in ARCTIC_PROBE_UTTERANCES:
            text = prompts[utterance]
            sample_id = f"clone_{speaker}_{utterance.removeprefix('arctic_')}"
            sample = {
                "sample_id": sample_id,
                "source_type": "piper_tts",
                "role": "probe",
                "speaker_key": f"arctic_{speaker}",
                "clone_of": f"arctic_{speaker}",
                "generator": "piper",
                "generator_version": "1.8.0",
                "voice": "en_US-arctic-medium",
                "voice_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx",
                "voice_sha256": voice_sha,
                "voice_config_sha256": config_sha,
                "license": "Voice trained on CMU ARCTIC (free for any purpose). Piper itself is GPL-3.0 and is a generation tool only, not a dependency of the verifier.",
                "speaker_id": speaker,
                "voice_speaker_index": speaker_map[speaker],
                "source_utterance": utterance,
                "text": text,
                "synthesis": dict(SYNTHESIS),
                "filename": f"{sample_id}.wav",
            }
            payload = synthesise(voice, sample)
            path = destination / sample["filename"]
            path.write_bytes(payload)

            info = sf.info(str(path))
            pcm, _ = sf.read(str(path), dtype="int16", always_2d=False)
            sample.update({
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "sample_count": info.frames,
                "duration_seconds": round(info.duration, 4),
                "sha256": sha256_file(path),
                "pcm_envelope_db": fingerprint(pcm),
            })
            samples.append(sample)
    return samples


def build_trials(librispeech: list[dict], arctic: list[dict], clones: list[dict]) -> list[dict]:
    """Pairs every reference with its probes, deterministically.

    Four relations, kept apart on purpose:

        same_speaker           another genuine utterance from the enrolled person
        different_speaker      genuine speech from a different person
        clone                  TTS trained on the enrolled person
        synthetic_non_target   TTS trained on a DIFFERENT person

    Only the first two are speaker-verification target/non-target trials, and
    only those feed the headline error rates. A clone is not a speaker the
    system should accept, but it is also not the human-impostor case FAR is
    defined over, so folding it in would quietly change what FAR means.
    """
    trials: list[dict] = []

    def add(reference: dict, probe_id: str, relation: str, corpus: str) -> None:
        trials.append({
            "trial_id": f"{reference['speaker_key']}__{probe_id}",
            "reference_speaker": reference["speaker_key"],
            "probe_sample": probe_id,
            "relation": relation,
            "corpus": corpus,
            "is_target": relation == "same_speaker",
        })

    for group, corpus in ((librispeech, "librispeech"), (arctic, "cmu_arctic")):
        count = len(group)
        for index, reference in enumerate(group):
            for probe_id in reference["probes"]:
                add(reference, probe_id, "same_speaker", corpus)
            for offset in IMPOSTOR_OFFSETS:
                impostor = group[(index + offset) % count]
                if impostor["speaker_key"] == reference["speaker_key"]:
                    continue
                for probe_id in impostor["probes"]:
                    add(reference, probe_id, "different_speaker", corpus)

    for reference in arctic:
        for clone in clones:
            relation = "clone" if clone["clone_of"] == reference["speaker_key"] else "synthetic_non_target"
            add(reference, clone["sample_id"], relation, "cmu_arctic")

    return trials


def authoring_environment() -> dict:
    """Records what produced this manifest, so a difference is visible."""
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    # Read from installed package metadata, not module attributes: piper exposes
    # no __version__, and a silently missing version is exactly the environment
    # difference this block exists to make visible.
    for distribution in ("piper-tts", "onnxruntime", "numpy", "soundfile", "torch", "speechbrain"):
        try:
            environment[distribution.replace("-", "_")] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            environment[distribution.replace("-", "_")] = "unavailable"
    try:
        import onnxruntime

        environment["execution_providers"] = list(onnxruntime.get_available_providers())
        environment["onnxruntime_device"] = onnxruntime.get_device()
    except Exception:  # noqa: BLE001
        pass
    environment["note"] = (
        "Full-file SHA-256 values for Piper clone samples reproduce bit-for-bit only in an environment "
        "equivalent to this one. Cross-machine verification uses the tolerant canonical-PCM check."
    )
    return environment


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--librispeech", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--arctic", type=Path, default=DEFAULT_ARCTIC_DIR)
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE_DIR / VOICE_NAME)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download", action="store_true", help="fetch the corpora and voice first")
    args = parser.parse_args()

    if args.download:
        print("Fetching sources:")
        download_corpus(args.librispeech)
        # The ARCTIC fetch needs the sample list, which needs the speakers; the
        # utterance names are fixed constants, so build the list from them here.
        placeholder = [
            {"sample_id": f"{speaker}_{utterance}", "speaker_id": speaker,
             "source_utterance": utterance,
             "source_relpath": f"cmu_us_{speaker}_arctic/{utterance}.wav"}
            for speaker in ARCTIC_SPEAKERS
            for utterance in (*ARCTIC_ENROLMENT_UTTERANCES, *ARCTIC_PROBE_UTTERANCES)
        ]
        download_arctic(placeholder, args.arctic)
        download_voice(args.voice.parent)
        print()

    print("Selecting LibriSpeech speakers...")
    ls_speakers, ls_samples = build_librispeech(args.librispeech)
    print(f"  {len(ls_speakers)} speakers, {len(ls_samples)} utterances")

    print("Selecting CMU ARCTIC speakers...")
    arctic_speakers, arctic_samples = build_arctic(args.arctic)
    print(f"  {len(arctic_speakers)} speakers, {len(arctic_samples)} utterances")

    print(f"Synthesising clones with {args.voice.name}...")
    clone_samples = build_clones(args.voice, args.eval_dir, arctic_prompts(args.arctic))
    print(f"  {len(clone_samples)} clone utterances")

    speakers = ls_speakers + arctic_speakers
    samples = ls_samples + arctic_samples + clone_samples
    trials = build_trials(ls_speakers, arctic_speakers, clone_samples)
    relations = {relation: sum(1 for t in trials if t["relation"] == relation) for relation in ("same_speaker", "different_speaker", "clone", "synthetic_non_target")}

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": (
            "VoxSentinel ROHAN-003 speaker-verification evaluation set. Audio is NOT committed. "
            "Rebuild it with ml/scripts/prepare_sv_eval_set.py, which regenerates every sample from this "
            "manifest and verifies each one."
        ),
        "librispeech_selection_rule": (
            f"For each dev-clean speaker, sort chapters by numeric id and keep those holding at least "
            f"{max(LIBRISPEECH_ENROLMENT_UTTERANCES, LIBRISPEECH_PROBE_UTTERANCES)} utterances of at least "
            f"{LIBRISPEECH_MIN_SECONDS}s. A speaker qualifies only with two such chapters. Enrol on the first "
            f"{LIBRISPEECH_ENROLMENT_UTTERANCES} utterances of the first chapter (sorted by POSIX path); probe with "
            f"the first {LIBRISPEECH_PROBE_UTTERANCES} of the second. Enrolment and probe therefore never share a "
            f"recording session."
        ),
        "arctic_selection_rule": (
            f"Speakers {', '.join(ARCTIC_SPEAKERS)}, chosen because the Piper en_US-arctic voice can synthesise all of "
            f"them. Enrol on {', '.join(ARCTIC_ENROLMENT_UTTERANCES)}; probe with {', '.join(ARCTIC_PROBE_UTTERANCES)}."
        ),
        "clone_rule": (
            "For every ARCTIC speaker, synthesise both probe prompts with the Piper en_US-arctic voice using that "
            "speaker's own voice index. The voice is trained on CMU ARCTIC, so these are clones of the enrolled "
            "person rather than unrelated synthetic speech."
        ),
        "impostor_rule": f"Each reference is probed against the speakers at offsets {list(IMPOSTOR_OFFSETS)} in the sorted speaker list, wrapping around.",
        "trial_relations": {
            "same_speaker": "target: another genuine utterance from the enrolled person",
            "different_speaker": "non-target: genuine speech from a different person",
            "clone": "attack: TTS trained on the enrolled person; not counted in FAR",
            "synthetic_non_target": "attack: TTS trained on a different person; not counted in FAR",
        },
        "verification": {
            "librispeech": "strict SHA-256 of the file, which is a byte copy of a fixed corpus file",
            "cmu_arctic": "strict SHA-256 of the file, which is a byte copy of a fixed corpus file",
            "piper_tts": (
                "tolerant canonical-PCM check: sample rate, channel count, and sample count must match exactly, the "
                "Piper model and config hashes must match, and the frame-RMS dB envelope must agree within the "
                "tolerance below. Full-file SHA-256 is informational only, because Piper output is not "
                "bit-reproducible across ONNX Runtime execution plans."
            ),
            "canonical_frame_samples": CANONICAL_FRAME_SAMPLES,
            "canonical_floor_db": CANONICAL_FLOOR_DB,
            "canonical_tolerance_db": CANONICAL_TOLERANCE_DB,
            "tolerance_evidence": "Measured in ROHAN-002 and unchanged here; see ml/audio/fingerprint.py.",
        },
        "arctic_metadata_sha256": {name: sha256_file(args.arctic / name) for name in ARCTIC_METADATA},
        "authoring_environment": authoring_environment(),
        "counts": {
            "speakers": len(speakers),
            "samples": len(samples),
            "trials": len(trials),
            "by_source": {kind: sum(1 for s in samples if s["source_type"] == kind) for kind in ("librispeech", "cmu_arctic", "piper_tts")},
            "by_relation": relations,
        },
        "speakers": speakers,
        "samples": samples,
        "trials": trials,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {args.out}")
    print(f"  speakers {len(speakers)}   samples {len(samples)}   trials {len(trials)}")
    for relation, count in relations.items():
        print(f"    {relation:22} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
