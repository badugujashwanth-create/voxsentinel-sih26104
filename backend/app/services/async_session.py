"""Shared asynchronous session primitives for streaming audio inference."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from secrets import token_urlsafe
from typing import Any


class SessionClosedError(RuntimeError):
    """Raised when a caller uses a closed asynchronous session."""


@dataclass(frozen=True)
class AudioWindow:
    """One complete model window awaiting inference."""

    sequence: int
    samples: Any


_CLOSED_SENTINEL = object()
DEFAULT_MAX_PENDING_WINDOWS = 4


class AsyncCallSession:
    """Owns one call's bounded inference queue and cancellation state."""

    def __init__(self, call_id: str, max_pending_windows: int = DEFAULT_MAX_PENDING_WINDOWS) -> None:
        """Creates an open session with a bounded pending-window queue."""
        if not call_id.strip():
            raise ValueError("call_id must not be blank")
        if max_pending_windows < 1:
            raise ValueError("max_pending_windows must be positive")
        self.call_id = call_id
        self.queue: asyncio.Queue[AudioWindow | object] = asyncio.Queue(maxsize=max_pending_windows)
        self.cancellation = asyncio.Event()
        self.generation_token = token_urlsafe(18)
        self.dropped_window_count = 0
        self._next_window_sequence = 0
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

    def register(self, call_id: str, max_pending_windows: int = DEFAULT_MAX_PENDING_WINDOWS) -> AsyncCallSession:
        """Registers one runtime session and rejects duplicate call ids."""
        if call_id in self._sessions:
            raise ValueError(f"session for {call_id} is already registered")
        session = AsyncCallSession(call_id, max_pending_windows)
        self._sessions[call_id] = session
        return session

    def get(self, call_id: str) -> AsyncCallSession | None:
        """Returns the registered session for a call, if one exists."""
        return self._sessions.get(call_id)

    def is_current(self, call_id: str, generation_token: str) -> bool:
        """Checks that a generation token still identifies the active session."""
        session = self.get(call_id)
        return session is not None and not session.is_closed and session.generation_token == generation_token

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
