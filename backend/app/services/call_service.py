"""Call session lifecycle rules."""

from __future__ import annotations

from datetime import UTC, datetime

from app.models.call import CallSession, CallStatus, CreateCallRequest
from app.state.session_store import SessionStore

ALLOWED_TRANSITIONS: dict[CallStatus, frozenset[CallStatus]] = {
    CallStatus.CREATED: frozenset({CallStatus.LIVE, CallStatus.FAILED}),
    CallStatus.LIVE: frozenset({CallStatus.VERIFYING, CallStatus.BLOCKED, CallStatus.COMPLETED, CallStatus.FAILED}),
    CallStatus.VERIFYING: frozenset({CallStatus.LIVE, CallStatus.BLOCKED, CallStatus.COMPLETED, CallStatus.FAILED}),
    CallStatus.BLOCKED: frozenset({CallStatus.COMPLETED, CallStatus.FAILED}),
    CallStatus.COMPLETED: frozenset(),
    CallStatus.FAILED: frozenset(),
}


class CallNotFoundError(LookupError):
    """Raised when a call id has no stored session."""


class InvalidTransitionError(ValueError):
    """Raised when a lifecycle transition is not permitted."""

    def __init__(self, current: CallStatus, target: CallStatus) -> None:
        super().__init__(f"Cannot move call from {current} to {target}")
        self.current = current
        self.target = target


class CallService:
    """Creates call sessions and moves them between lifecycle states."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    def create(self, request: CreateCallRequest) -> CallSession:
        """Opens a new call session in ``CREATED``."""
        return self._store.create(request)

    def get(self, call_id: str) -> CallSession:
        """Returns a session or raises ``CallNotFoundError``."""
        session = self._store.get(call_id)
        if session is None:
            raise CallNotFoundError(call_id)
        return session

    def start(self, call_id: str) -> CallSession:
        """Moves a call from ``CREATED`` to ``LIVE``."""
        session = self._transition(call_id, CallStatus.LIVE)
        session.started_at = datetime.now(UTC)
        return self._store.save(session)

    def stop(self, call_id: str) -> CallSession:
        """Moves a live call to ``COMPLETED``."""
        session = self._transition(call_id, CallStatus.COMPLETED)
        session.completed_at = datetime.now(UTC)
        return self._store.save(session)

    def _transition(self, call_id: str, target: CallStatus) -> CallSession:
        """Applies a guarded status change to a stored session."""
        session = self.get(call_id)
        if target not in ALLOWED_TRANSITIONS[session.status]:
            raise InvalidTransitionError(session.status, target)
        session.status = target
        return session
