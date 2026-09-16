#!/usr/bin/env python3
"""Generate synthetic speech samples for spoof-detection evaluation.

Uses Piper (https://github.com/OHF-Voice/piper1-gpl, GPL-3.0) with the
``en_US-libritts_r-medium`` voice (CC BY 4.0, trained on LibriTTS-R). Piper is a
sample-GENERATION tool only. It is deliberately not a dependency of the spoof
detector and is not listed in ml/requirements.txt.

Install it separately, outside the ML runtime:

    uv venv /tmp/ttsvenv --python 3.12
    uv pip install --python /tmp/ttsvenv/bin/python piper-tts==1.8.0
    /tmp/ttsvenv/bin/python ml/scripts/generate_tts_samples.py

The voice is multi-speaker, so samples vary by speaker as well as by text.
Nothing generated here is committed to the repository.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOICE = REPO_ROOT / "ml" / "data" / "tts-voices" / "en_US-libritts_r-medium.onnx"
DEFAULT_OUTPUT = REPO_ROOT / "ml" / "data" / "eval" / "synthetic"

#: Long enough that the detector sees real speech rather than tiling padding.
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


def build_voice(model_path: Path):
    """Loads the Piper voice, failing with an actionable message."""
    try:
        from piper import PiperVoice
    except ImportError as exc:  # pragma: no cover - tool-only path
        raise SystemExit(f"piper is not installed in this interpreter: {exc}\nSee the module docstring for the install command.") from exc
    if not model_path.is_file():
        raise SystemExit(f"voice model not found: {model_path}\nDownload en_US-libritts_r-medium from https://huggingface.co/rhasspy/piper-voices")
    return PiperVoice.load(model_path)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE, help="path to the Piper .onnx voice")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directory to write WAV files into")
    parser.add_argument("--count", type=int, default=40, help="how many samples to generate")
    parser.add_argument("--speakers", type=int, nargs="+", default=[0, 7, 19, 42, 63, 101, 137, 188, 256, 311, 402, 513, 604, 712, 803, 877], help="speaker ids to cycle through")
    args = parser.parse_args()

    from piper import SynthesisConfig

    voice = build_voice(args.voice)
    args.output.mkdir(parents=True, exist_ok=True)

    combinations = len(SENTENCES) * len(args.speakers)
    if args.count > combinations:
        raise SystemExit(f"--count {args.count} exceeds the {combinations} available sentence/speaker pairs")

    written = 0
    for index in range(args.count):
        # Walk sentences and speakers at different strides so each sample is a
        # distinct pair rather than the same text repeated.
        sentence = SENTENCES[index % len(SENTENCES)]
        speaker = args.speakers[(index // len(SENTENCES) + index) % len(args.speakers)]
        # Vary pacing slightly so samples are not uniform in rhythm.
        length_scale = 1.0 + 0.05 * ((index % 3) - 1)
        path = args.output / f"tts_{index:03d}_spk{speaker:03d}.wav"
        with wave.open(str(path), "wb") as handle:
            voice.synthesize_wav(sentence, handle, syn_config=SynthesisConfig(speaker_id=speaker, length_scale=length_scale))
        written += 1
        print(f"  {path.name}  speaker={speaker}  len_scale={length_scale:.2f}  {path.stat().st_size:,} bytes")

    print(f"\nWrote {written} synthetic samples to {args.output}")
    print("Voice: en_US-libritts_r-medium (CC BY 4.0) via Piper 1.8.0 (GPL-3.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
