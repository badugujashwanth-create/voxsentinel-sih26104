"""Call session REST endpoints."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, status

from app.models.call import CallSession, CreateCallRequest, CreateCallResponse
from app.services.call_service import CallService
from app.state.session_store import SessionStore

router = APIRouter(prefix="/api/v1/calls", tags=["calls"])


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """Returns the process-wide session store."""
    return SessionStore()


def get_call_service(store: SessionStore = Depends(get_session_store)) -> CallService:
    """Builds the call service over the shared store."""
    return CallService(store)


@router.post("", response_model=CreateCallResponse, status_code=status.HTTP_201_CREATED)
def create_call(request: CreateCallRequest, service: CallService = Depends(get_call_service)) -> CallSession:
    """Opens a call session in ``CREATED``."""
    return service.create(request)


@router.get("/{call_id}", response_model=CallSession)
def read_call(call_id: str, service: CallService = Depends(get_call_service)) -> CallSession:
    """Returns the full current state of one call session."""
    return service.get(call_id)


@router.post("/{call_id}/start", response_model=CallSession)
def start_call(call_id: str, service: CallService = Depends(get_call_service)) -> CallSession:
    """Moves a call from ``CREATED`` to ``LIVE``."""
    return service.start(call_id)


@router.post("/{call_id}/stop", response_model=CallSession)
def stop_call(call_id: str, service: CallService = Depends(get_call_service)) -> CallSession:
    """Moves a live call to ``COMPLETED``."""
    return service.stop(call_id)
