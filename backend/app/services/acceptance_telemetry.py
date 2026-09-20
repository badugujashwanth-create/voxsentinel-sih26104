"""Bounded, ephemeral metadata snapshots for local acceptance verification."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any

DEFAULT_MAX_INFERENCES = 50
DEFAULT_MAX_SNAPSHOTS = 16


@dataclass
class AcceptanceTelemetry:
    """Stores non-audio lifecycle and inference metadata for one call."""

    call_id: str
    max_inferences: int = DEFAULT_MAX_INFERENCES
    session_started_at: float = field(default_factory=time.time)
    microphone_state: str = "IDLE"
    audio_context_sample_rate: int | None = None
    channels: int | None = None
    canonical_samples: int = 0
    aasist_windows_generated: int = 0
    backend_windows_dropped: int = 0
    source_gap_count: int = 0
    warmup_latency_ms: float | None = None
    _stream_started_at: float | None = field(default=None, repr=False)
    _inferences: deque[dict[str, Any]] = field(init=False, repr=False)
    _cleanup: dict[str, bool] = field(default_factory=lambda: {"producer_active": False, "canonicalizer_active": False, "session_active": True})

    def __post_init__(self) -> None:
        """Validates the bounded history configuration."""
        if self.max_inferences < 1:
            raise ValueError("max_inferences must be positive")
        self._inferences = deque(maxlen=self.max_inferences)

    def mark_stream_started(self) -> None:
        """Starts warm-up timing without storing audio."""
        self._stream_started_at = time.perf_counter()

    def mark_first_event(self) -> None:
        """Captures warm-up once, on the first observed risk event."""
        if self.warmup_latency_ms is None and self._stream_started_at is not None:
            self.warmup_latency_ms = (time.perf_counter() - self._stream_started_at) * 1000

    def update_audio(self, *, canonical_samples: int, windows_generated: int, windows_dropped: int, source_gap_count: int) -> None:
        """Accumulates bounded audio-session counters without retaining audio."""
        self.canonical_samples += canonical_samples
        self.aasist_windows_generated += windows_generated
        self.backend_windows_dropped = windows_dropped
        self.source_gap_count = source_gap_count

    def record_inference(self, *, audio_window_sequence: int | None, raw_spoof_score: float, aggregate_score: float, policy_state: str, overall_risk_score: int, risk_level: str, recommended_action: str, ml_inference_latency_ms: float | None, steady_state_latency_ms: float | None) -> None:
        """Adds one bounded uncalibrated model-evidence observation."""
        self.mark_first_event()
        self._inferences.append({
            "audio_window_sequence": audio_window_sequence,
            "raw_spoof_score": raw_spoof_score,
            "aggregate_score": aggregate_score,
            "policy_state": policy_state,
            "overall_risk_score": overall_risk_score,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "ml_inference_latency_ms": ml_inference_latency_ms,
            "steady_state_latency_ms": steady_state_latency_ms,
            "score_semantics": "uncalibrated",
        })

    def mark_cleanup(self) -> None:
        """Marks all runtime resources inactive after disposal."""
        self._cleanup = {"producer_active": False, "canonicalizer_active": False, "session_active": False}

    def snapshot(self) -> dict[str, Any]:
        """Returns a detached JSON-safe snapshot without audio data."""
        return {
            "call_id": self.call_id,
            "session_started_at": self.session_started_at,
            "microphone_state": self.microphone_state,
            "audio_context_sample_rate": self.audio_context_sample_rate,
            "channels": self.channels,
            "canonical_samples": self.canonical_samples,
            "aasist_windows_generated": self.aasist_windows_generated,
            "backend_windows_dropped": self.backend_windows_dropped,
            "source_gap_count": self.source_gap_count,
            "warmup_latency_ms": self.warmup_latency_ms,
            "inferences": list(self._inferences),
            "cleanup": dict(self._cleanup),
        }


class AcceptanceTelemetryRegistry:
    """Owns a bounded, ephemeral set of per-call acceptance snapshots."""

    def __init__(self, max_snapshots: int = DEFAULT_MAX_SNAPSHOTS) -> None:
        """Creates a registry that retains only a bounded completed-call history."""
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be positive")
        self.max_snapshots = max_snapshots
        self._items: dict[str, AcceptanceTelemetry] = {}

    def create(self, call_id: str) -> AcceptanceTelemetry:
        """Creates a fresh snapshot after evicting completed snapshots as needed."""
        self._evict_completed()
        if call_id not in self._items and len(self._items) >= self.max_snapshots:
            raise RuntimeError("acceptance telemetry registry is at active-session capacity")
        telemetry = AcceptanceTelemetry(call_id)
        self._items[call_id] = telemetry
        return telemetry

    def get(self, call_id: str) -> AcceptanceTelemetry | None:
        """Returns the current snapshot for a call."""
        return self._items.get(call_id)

    def remove(self, call_id: str) -> None:
        """Discards a disposed call snapshot."""
        self._items.pop(call_id, None)

    def _evict_completed(self) -> None:
        """Drops the oldest inactive snapshots before accepting another call."""
        for call_id, telemetry in tuple(self._items.items()):
            if len(self._items) < self.max_snapshots:
                return
            if not telemetry.snapshot()["cleanup"]["session_active"]:
                self._items.pop(call_id, None)