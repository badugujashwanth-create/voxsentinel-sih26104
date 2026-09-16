#!/usr/bin/env python3
"""Measure real spoof-detection latency on this machine.

    python ml/scripts/benchmark_latency.py --audio ml/data/eval/genuine/sample.flac

Runs warm-up passes first, then reports the distribution over repeated runs.
Numbers describe THIS machine only; they are not a portable performance claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.audio import preprocessing  # noqa: E402
from ml.spoof.aasist import AASIST_SAMPLE_RATE, AASISTSpoofDetector, ModelNotInstalledError  # noqa: E402

DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 20


def cpu_name() -> str:
    """Best-effort CPU model string."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def describe_hardware(device: str) -> dict[str, str]:
    """Collects the hardware facts a latency number is only meaningful with."""
    info = {
        "device": device,
        "cpu": cpu_name(),
        "cpu_threads_used_by_torch": str(torch.get_num_threads()),
        "logical_cores": str(len(__import__("os").sched_getaffinity(0))),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    else:
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            if gpu.returncode == 0 and gpu.stdout.strip():
                info["gpu_present_but_unused"] = gpu.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def synthetic_speech(seconds: float, sample_rate: int = AASIST_SAMPLE_RATE) -> np.ndarray:
    """Deterministic harmonic signal used when no audio file is supplied."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    f0 = 120.0 + 20.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    signal = sum(0.5**k * np.sin((k + 1) * phase) for k in range(5))
    return (0.3 * signal * (0.5 * (1 + np.sin(2 * np.pi * 3.0 * t)))).astype(np.float32)


def percentile(values: list[float], fraction: float) -> float:
    """Returns the nearest-rank percentile of a sample."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, help="audio file to benchmark (default: a generated 4.04s signal)")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"warm-up passes (default {DEFAULT_WARMUP})")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help=f"measured passes (default {DEFAULT_REPEATS})")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    try:
        detector = AASISTSpoofDetector(device=args.device)
    except ModelNotInstalledError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.audio:
        samples, source_rate = preprocessing.read_raw(args.audio)
        source = str(args.audio)
    else:
        samples, source_rate = synthetic_speech(64_600 / AASIST_SAMPLE_RATE), AASIST_SAMPLE_RATE
        source = "generated 4.04s harmonic signal"

    for _ in range(args.warmup):
        detector.score(samples, source_rate)

    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    for _ in range(args.repeats):
        result = detector.score(samples, source_rate)
        preprocess_ms.append(result.preprocessing_seconds * 1000)
        inference_ms.append(result.inference_seconds * 1000)

    total_ms = [p + i for p, i in zip(preprocess_ms, inference_ms, strict=True)]
    duration = len(preprocessing.prepare(samples, source_rate, AASIST_SAMPLE_RATE).samples) / AASIST_SAMPLE_RATE
    hardware = describe_hardware(args.device)

    def stats(values: list[float]) -> dict[str, float]:
        """Summarises one timing series in milliseconds."""
        return {
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "p95_ms": percentile(values, 0.95),
        }

    payload = {
        "model": detector.model_id,
        "source": source,
        "audio_duration_seconds": duration,
        "model_window_seconds": 64_600 / AASIST_SAMPLE_RATE,
        "warmup_passes": args.warmup,
        "measured_passes": args.repeats,
        "hardware": hardware,
        "preprocessing": stats(preprocess_ms),
        "inference": stats(inference_ms),
        "total": stats(total_ms),
        "realtime_factor": (64_600 / AASIST_SAMPLE_RATE) / (statistics.median(total_ms) / 1000),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Model:            {payload['model']}")
    print(f"Source:           {source}")
    print(f"Audio duration:   {duration:.2f}s (model window {payload['model_window_seconds']:.2f}s)")
    print(f"Passes:           {args.warmup} warm-up + {args.repeats} measured")
    print("\nHardware:")
    for key, value in hardware.items():
        print(f"  {key:28} {value}")
    print("\nLatency (milliseconds):")
    print(f"  {'stage':14} {'mean':>9} {'median':>9} {'min':>9} {'max':>9} {'p95':>9}")
    for stage in ("preprocessing", "inference", "total"):
        s = payload[stage]
        print(f"  {stage:14} {s['mean_ms']:>9.2f} {s['median_ms']:>9.2f} {s['min_ms']:>9.2f} {s['max_ms']:>9.2f} {s['p95_ms']:>9.2f}")
    print(f"\nReal-time factor: {payload['realtime_factor']:.1f}x  (model window / median total)")
    print("\nMeasured on this machine only. Not a portable performance claim, and not")
    print("a statement about end-to-end latency in a live call pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
