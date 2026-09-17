"""Call session REST endpoints and the live risk WebSocket."""

from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status

from app.config import get_settings
from app.models.call import CallSession, CallStatus, CreateCallRequest, CreateCallResponse
from app.services.call_service import CallNotFoundError, CallService
from app.services.risk_provider import MockRiskProvider, RiskProvider
from app.state.session_store import SessionStore

#: Application-specific WebSocket close codes (4000-4999 is the private range).
WS_UNKNOWN_CALL = 4404
WS_CALL_NOT_LIVE = 4409

router = APIRouter(prefix="/api/v1/calls", tags=["calls"])


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """Returns the process-wide session store."""
    return SessionStore()


@lru_cache(maxsize=1)
def get_risk_provider() -> RiskProvider:
    """Returns the configured risk provider.

    Swapping the demo engine for real inference happens here and nowhere else.
    """
    return MockRiskProvider(emit_interval_ms=get_settings().risk_emit_interval_ms)


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


@router.websocket("/{call_id}/risk-stream")
async def stream_risk(websocket: WebSocket, call_id: str) -> None:
    """Emits live risk events for a call that is already ``LIVE``."""
    await websocket.accept()
    service = CallService(get_session_store())

    try:
        session = service.get(call_id)
    except CallNotFoundError:
        await websocket.close(code=WS_UNKNOWN_CALL, reason="Unknown call_id")
        return

    if session.status is not CallStatus.LIVE:
        await websocket.close(code=WS_CALL_NOT_LIVE, reason=f"Call is {session.status}, not LIVE")
        return

    cancellation = asyncio.Event()
    try:
        async for event in get_risk_provider().stream(session, cancellation):
            await websocket.send_json(event.model_dump(mode="json"))
        await websocket.close()
    except (WebSocketDisconnect, RuntimeError):
        # Client hung up mid-stream; nothing to clean up beyond stopping.
        return
    finally:
        cancellation.set()
