"""In-memory call session storage."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.models.call import CallSession, CallStatus, CreateCallRequest


class SessionStore:
    """Holds call sessions for the lifetime of the process."""

    def __init__(self) -> None:
        """Creates an empty in-memory store."""
        self._sessions: dict[str, CallSession] = {}

    def create(self, request: CreateCallRequest) -> CallSession:
        """Stores a new session in CREATED state."""
        session = CallSession(call_id=uuid4().hex[:12], status=CallStatus.CREATED, claimed_identity=request.claimed_identity, scenario=request.scenario, transaction_value=request.transaction_value, currency=request.currency, speaker_profile_id=request.speaker_profile_id, created_at=datetime.now(UTC))
        self._sessions[session.call_id] = session
        return session

    def get(self, call_id: str) -> CallSession | None:
        """Returns a session when present."""
        return self._sessions.get(call_id)

    def save(self, session: CallSession) -> CallSession:
        """Writes a mutated session back to the store."""
        self._sessions[session.call_id] = session
        return session

    def clear(self) -> None:
        """Drops every stored session."""
        self._sessions.clear()
