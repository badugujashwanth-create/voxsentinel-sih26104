"""In-memory call session storage.

No database by design: ROHAN-001 is a demo foundation and every session is
discarded when the process exits.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.models.call import CallSession, CallStatus, CreateCallRequest


class SessionStore:
    """Holds call sessions for the lifetime of the process.

    ponytail: plain dict, no locking. FastAPI serves these handlers on one
    event loop, so there is no concurrent mutation. Swap in a real store (or a
    lock) only if the backend ever runs multi-worker.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}

    def create(self, request: CreateCallRequest) -> CallSession:
        """Stores a new session in ``CREATED`` and returns it."""
        session = CallSession(
            call_id=uuid4().hex[:12],
            status=CallStatus.CREATED,
            claimed_identity=request.claimed_identity,
            scenario=request.scenario,
            transaction_value=request.transaction_value,
            currency=request.currency,
            created_at=datetime.now(UTC),
        )
        self._sessions[session.call_id] = session
        return session

    def get(self, call_id: str) -> CallSession | None:
        """Returns the session for ``call_id`` when one exists."""
        return self._sessions.get(call_id)

    def save(self, session: CallSession) -> CallSession:
        """Writes a mutated session back to the store."""
        self._sessions[session.call_id] = session
        return session

    def clear(self) -> None:
        """Drops every stored session."""
        self._sessions.clear()
