#!/usr/bin/env python3
"""Feed one complete WAV/FLAC container into a live VoxSentinel call."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx

SUPPORTED_SUFFIXES = {".wav": "audio/wav", ".flac": "audio/flac"}


def content_type_for(path: Path) -> str:
    """Returns the supported media type for an audio path."""
    try:
        return SUPPORTED_SUFFIXES[path.suffix.lower()]
    except KeyError as error:
        raise ValueError("audio path must end in .wav or .flac") from error


async def feed_audio(base_url: str, call_id: str, audio_path: Path, sequence: int) -> dict[str, object]:
    """Posts one complete source container without retaining raw audio."""
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        response = await client.post(
            f"/api/v1/calls/{call_id}/audio",
            content=audio_path.read_bytes(),
            headers={"Content-Type": content_type_for(audio_path), "X-Audio-Chunk-Sequence": str(sequence)},
        )
        response.raise_for_status()
        return response.json()


def main() -> int:
    """Parses arguments and feeds the selected complete container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=1)
    args = parser.parse_args()
    print(asyncio.run(feed_audio(args.base_url, args.call_id, args.audio, args.sequence)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
