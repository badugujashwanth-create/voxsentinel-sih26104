"""Shared asynchronous session primitives for streaming audio inference."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from secrets import token_urlsafe
from typing import Any

from app.services.audio_source_ledger import AudioWindowMetadata


class SessionClosedError(RuntimeError):
    """Raised when a caller uses a closed asynchronous session."""


@dataclass(frozen=True)
class AudioWindow:
    """One complete model window awaiting inference."""

    sequence: int
    samples: Any
    metadata: AudioWindowMetadata | None = None


_CLOSED_SENTINEL = object()
DEFAULT_MAX_PENDING_WINDOWS = 4


class AsyncCallSession:
    """Owns one call's bounded inference queue and cancellation state."""

    def __init__(self, call_id: str, max_pending_windows: int = DEFAULT_MAX_PENDING_WINDOWS, speaker_profile_id: str | None = None) -> None:
        """Creates an open session with a bounded pending-window queue."""
        if not call_id.strip():
            raise ValueError("call_id must not be blank")
        if max_pending_windows < 1:
            raise ValueError("max_pending_windows must be positive")
        self.call_id = call_id
        self.speaker_profile_id = speaker_profile_id
        self.queue: asyncio.Queue[AudioWindow | object] = asyncio.Queue(maxsize=max_pending_windows)
        self.cancellation = asyncio.Event()
        self.generation_token = token_urlsafe(18)
        self.dropped_window_count = 0
        self._next_window_sequence = 0
        self.last_chunk_sequence = 0
        self.window_scheduler: Any | None = None
        self.spoof_aggregator: Any | None = None
        self.risk_policy: Any | None = None
        self.event_sequence = 0
        self._stream_claimed = False
        self._audio_claimed = False
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """Reports whether this runtime session has been closed."""
        return self._closed

    def enqueue_window(self, window: AudioWindow | object) -> None:
        """Keeps the newest complete window, evicting the oldest when full."""
        if self._closed or self.cancellation.is_set():
            raise SessionClosedError(f"session {self.call_id} is closed")
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped_window_count += 1
        self.queue.put_nowait(window)

    def allocate_window_sequence(self) -> int:
        """Allocates the next monotonic sequence number for a model window."""
        if self._closed or self.cancellation.is_set():
            raise SessionClosedError(f"session {self.call_id} is closed")
        self._next_window_sequence += 1
        return self._next_window_sequence

    async def next_window(self) -> AudioWindow:
        """Waits for the next window or raises when the session closes."""
        item = await self.queue.get()
        if item is _CLOSED_SENTINEL:
            raise SessionClosedError(f"session {self.call_id} is closed")
        return item

    async def close(self) -> None:
        """Cancels the session and releases consumers idempotently."""
        if self._closed:
            return
        self._closed = True
        self.cancellation.set()
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(_CLOSED_SENTINEL)


class AudioSessionRegistry:
    """Maps each call id to exactly one active asynchronous call session."""

    def __init__(self) -> None:
        """Creates an empty runtime-session registry."""
        self._sessions: dict[str, AsyncCallSession] = {}

    def register(self, call_id: str, max_pending_windows: int = DEFAULT_MAX_PENDING_WINDOWS, speaker_profile_id: str | None = None) -> AsyncCallSession:
        """Registers one runtime session and rejects duplicate call ids."""
        if call_id in self._sessions:
            raise ValueError(f"session for {call_id} is already registered")
        session = AsyncCallSession(call_id, max_pending_windows, speaker_profile_id)
        self._sessions[call_id] = session
        return session

    def get(self, call_id: str) -> AsyncCallSession | None:
        """Returns the registered session for a call, if one exists."""
        return self._sessions.get(call_id)

    def is_current(self, call_id: str, generation_token: str) -> bool:
        """Checks that a generation token still identifies the active session."""
        session = self.get(call_id)
        return session is not None and not session.is_closed and session.generation_token == generation_token

    def claim_stream(self, call_id: str, generation_token: str) -> bool:
        """Claims the sole stream consumer for an active runtime session."""
        session = self.get(call_id)
        if session is None or session.is_closed or session.generation_token != generation_token or session._stream_claimed:
            return False
        session._stream_claimed = True
        return True

    def release_stream(self, call_id: str, generation_token: str) -> None:
        """Releases a stream claim without destroying the live call session."""
        session = self.get(call_id)
        if session is not None and session.generation_token == generation_token:
            session._stream_claimed = False

    def claim_audio(self, call_id: str, generation_token: str) -> bool:
        """Claims the sole microphone producer for an active runtime session."""
        session = self.get(call_id)
        if session is None or session.is_closed or session.generation_token != generation_token or session._audio_claimed:
            return False
        session._audio_claimed = True
        return True

    def release_audio(self, call_id: str, generation_token: str) -> None:
        """Releases the microphone producer claim without closing the call."""
        session = self.get(call_id)
        if session is not None and session.generation_token == generation_token:
            session._audio_claimed = False

    async def close(self, call_id: str, generation_token: str | None = None) -> None:
        """Closes the current session when its optional generation matches."""
        session = self.get(call_id)
        if session is None or (generation_token is not None and session.generation_token != generation_token):
            return
        await session.close()

    def remove(self, call_id: str, generation_token: str | None = None) -> None:
        """Removes a session only when its optional generation still matches."""
        session = self.get(call_id)
        if session is None or (generation_token is not None and session.generation_token != generation_token):
            return
        self._sessions.pop(call_id, None)

    def ingest(self, call_id: str, container: bytes, content_type: str, chunk_sequence: int) -> Any:
        """Canonicalizes one complete container and enqueues its model windows."""
        from app.services.audio_conversion import decode_and_canonicalize
        canonical = decode_and_canonicalize(container, content_type)
        return self.ingest_canonical(call_id, canonical, chunk_sequence)

    def ingest_canonical(self, call_id: str, canonical: Any, chunk_sequence: int) -> Any:
        """Enqueues already canonicalized audio on the event-loop-owned session."""
        from app.services.audio_ingestion import AudioIngestAcknowledgement
        from app.services.audio_windowing import AASISTWindowScheduler

        session = self.get(call_id)
        if session is None or session.is_closed:
            raise SessionClosedError(f"session {call_id} is closed or not registered")
        if chunk_sequence <= session.last_chunk_sequence:
            raise ValueError("audio chunk sequence must increase")
        if session.window_scheduler is None:
            session.window_scheduler = AASISTWindowScheduler()
        source_segment = getattr(canonical, "source_segment", None)
        windows = session.window_scheduler.push_with_metadata(canonical.samples, source_segment)
        for samples, metadata in windows:
            session.enqueue_window(AudioWindow(session.allocate_window_sequence(), samples, metadata))
        session.last_chunk_sequence = chunk_sequence
        return AudioIngestAcknowledgement(
            call_id=call_id,
            chunk_sequence=chunk_sequence,
            accepted_sample_count=int(canonical.samples.size),
            windows_enqueued=len(windows),
            dropped_window_count=session.dropped_window_count,
        )
