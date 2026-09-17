#!/usr/bin/env python3
"""Run the live HTTP -> audio -> ML -> WebSocket acceptance path."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import websockets

SUPPORTED_SUFFIXES = {".wav": "audio/wav", ".flac": "audio/flac"}


async def run_acceptance(base_url: str, audio_path: Path, scenario: str) -> list[dict[str, object]]:
    """Creates a call, starts it, feeds complete audio, and collects events."""
    if audio_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("audio path must end in .wav or .flac")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        created = await client.post("/api/v1/calls", json={"claimed_identity": "CEO Demo", "scenario": scenario})
        created.raise_for_status()
        call_id = created.json()["call_id"]
        started = await client.post(f"/api/v1/calls/{call_id}/start")
        started.raise_for_status()
        websocket_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + f"/api/v1/calls/{call_id}/risk-stream"
        async with websockets.connect(websocket_url) as socket:
            feed_task = asyncio.create_task(client.post(
                f"/api/v1/calls/{call_id}/audio",
                content=audio_path.read_bytes(),
                headers={"Content-Type": SUPPORTED_SUFFIXES[audio_path.suffix.lower()], "X-Audio-Chunk-Sequence": "1"},
            ))
            events: list[dict[str, object]] = []
            while True:
                try:
                    events.append(json.loads(await asyncio.wait_for(socket.recv(), timeout=30)))
                except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                    break
            response = await feed_task
            response.raise_for_status()
        await client.post(f"/api/v1/calls/{call_id}/stop")
    return events


def main() -> int:
    """Parses arguments and prints collected live events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--scenario", default="HIGH_VALUE_TRANSFER_ATTACK")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_acceptance(args.base_url, args.audio, args.scenario)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
