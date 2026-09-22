"""Call session REST endpoints and the live risk WebSocket."""

from __future__ import annotations

import asyncio
import base64
import binascii
import numpy as np
import json
import os
import time
from dataclasses import replace
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status

from app.config import get_settings
from app.models.call import CallSession, CallStatus, CreateCallRequest, CreateCallResponse, CreateSpeakerProfileRequest
from app.services.call_service import CallNotFoundError, CallService
from app.services.async_session import AudioSessionRegistry, SessionClosedError
from app.services.audio_conversion import AudioConversionError
from app.services.audio_conversion import decode_and_canonicalize
from app.services.audio_protocol import AudioProtocolError, AudioStartMetadata
from app.services.audio_stream import AudioStreamHandler
from app.services.ml_service_client import MLServiceClient, MLServiceError
from app.services.acceptance_telemetry import AcceptanceTelemetryRegistry
from app.services.speaker_profiles import SpeakerProfileRegistry
from app.services.risk_provider import MLRiskProvider, MockRiskProvider, RiskProvider
from app.state.session_store import SessionStore

#: Application-specific WebSocket close codes (4000-4999 is the private range).
WS_UNKNOWN_CALL = 4404
WS_CALL_NOT_LIVE = 4409
WS_AUDIO_INVALID = 4400
WS_AUDIO_DUPLICATE = 4410
WS_AUDIO_NOT_READY = 4411
MAX_AUDIO_CONTAINER_BYTES = 25 * 1024 * 1024
MAX_PROFILE_SAMPLES = 480_000

router = APIRouter(prefix="/api/v1/calls", tags=["calls"])


async def read_limited_body(request: Request, maximum_bytes: int = MAX_AUDIO_CONTAINER_BYTES) -> bytes:
    """Reads a request body with a bounded container size."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content-Length must be an integer") from error
        if declared_length < 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content-Length must not be negative")
        if declared_length > maximum_bytes:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="audio container is too large")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > maximum_bytes:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="audio container is too large")
    return bytes(body)


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """Returns the process-wide session store."""
    return SessionStore()


@lru_cache(maxsize=1)
def get_audio_session_registry() -> AudioSessionRegistry:
    """Returns the process-wide audio runtime-session registry."""
    return AudioSessionRegistry()


@lru_cache(maxsize=1)
def get_speaker_profile_registry() -> SpeakerProfileRegistry:
    """Returns the bounded process-local profile registry."""
    return SpeakerProfileRegistry()

@lru_cache(maxsize=1)
def get_acceptance_telemetry_registry() -> AcceptanceTelemetryRegistry:
    """Returns ephemeral acceptance snapshots for local verification."""
    return AcceptanceTelemetryRegistry()

@lru_cache(maxsize=1)
def get_risk_provider() -> RiskProvider:
    """Returns the configured risk provider.

    Swapping the demo engine for real inference happens here and nowhere else.
    """
    settings = get_settings()
    if settings.risk_provider_mode.value == "ml":
        return MLRiskProvider(get_audio_session_registry(), MLServiceClient(settings.ml_service_url), get_acceptance_telemetry_registry(), get_speaker_profile_registry())
    return MockRiskProvider(emit_interval_ms=settings.risk_emit_interval_ms)


def get_call_service(store: SessionStore = Depends(get_session_store)) -> CallService:
    """Builds the call service over the shared store."""
    return CallService(store)


@router.post("", response_model=CreateCallResponse, status_code=status.HTTP_201_CREATED)
def create_call(request: CreateCallRequest, service: CallService = Depends(get_call_service)) -> CallSession:
    """Opens a call session in ``CREATED``."""
    return service.create(request)


@router.get("/{call_id}/acceptance-telemetry")
async def read_acceptance_telemetry(call_id: str) -> object:
    """Returns opt-in local acceptance metadata without audio payloads."""
    import os
    if os.getenv("VOXSENTINEL_ACCEPTANCE_TELEMETRY", "") != "1":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Acceptance telemetry is disabled")
    telemetry = get_acceptance_telemetry_registry().get(call_id)
    if telemetry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No acceptance telemetry for call")
    return telemetry.snapshot()

@router.post("/speaker-profiles", status_code=status.HTTP_201_CREATED)
async def create_speaker_profile(request: CreateSpeakerProfileRequest) -> object:
    """Creates a bounded profile from ephemeral canonical float32 audio."""
    provider = get_risk_provider()
    if not isinstance(provider, MLRiskProvider):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Speaker profiles require ML mode")
    try:
        samples = np.frombuffer(base64.b64decode(request.samples_base64, validate=True), dtype="<f4").copy()
        if samples.ndim != 1 or samples.size < 48_000 or samples.size > MAX_PROFILE_SAMPLES or not np.isfinite(samples).all():
            raise ValueError("profile audio must be finite mono float32 with at least 3 seconds")
        result = await provider.client.embed_speaker(samples.astype(np.float32))
        profile = get_speaker_profile_registry().add(request.expected_speaker_id, result, request.provenance)
        return {"profile_id": profile.profile_id, "expected_speaker_id": profile.expected_speaker_id, "model_id": profile.model_id, "model_revision": profile.model_revision, "embedding_dimensions": profile.embedding_dimensions, "threshold": profile.threshold, "score_semantics": "uncalibrated_embedding"}
    except (binascii.Error, ValueError, MLServiceError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

@router.get("/{call_id}", response_model=CallSession)
def read_call(call_id: str, service: CallService = Depends(get_call_service)) -> CallSession:
    """Returns the full current state of one call session."""
    return service.get(call_id)


@router.post("/{call_id}/start", response_model=CallSession)
async def start_call(call_id: str, service: CallService = Depends(get_call_service)) -> CallSession:
    """Moves a call from ``CREATED`` to ``LIVE``."""
    provider = get_risk_provider()
    runtime_session = None
    if isinstance(provider, MLRiskProvider):
        try:
            call = service.get(call_id)
            runtime_session = await provider.prepare_session(call_id, call.speaker_profile_id)
            if os.getenv("VOXSENTINEL_ACCEPTANCE_TELEMETRY", "") == "1":
                get_acceptance_telemetry_registry().create(call_id)
        except MLServiceError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"ML start unavailable: {error}") from error
        except Exception as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"ML start failed: {error}") from error
    try:
        return service.start(call_id)
    except Exception:
        if isinstance(provider, MLRiskProvider) and runtime_session is not None:
            await provider.close_session(call_id, runtime_session.generation_token)
        raise


@router.post("/{call_id}/stop", response_model=CallSession)
async def stop_call(call_id: str, service: CallService = Depends(get_call_service)) -> CallSession:
    """Moves a live call to ``COMPLETED``."""
    result = service.stop(call_id)
    provider = get_risk_provider()
    if isinstance(provider, MLRiskProvider):
        await provider.close_session(call_id)
        telemetry = get_acceptance_telemetry_registry().get(call_id)
        if telemetry is not None: telemetry.mark_cleanup()
    return result


@router.post("/{call_id}/audio", status_code=status.HTTP_202_ACCEPTED)
async def ingest_audio(call_id: str, request: Request, service: CallService = Depends(get_call_service), registry: AudioSessionRegistry = Depends(get_audio_session_registry)) -> object:
    """Accepts one complete WAV/FLAC container for a live runtime session."""
    session = service.get(call_id)
    if session.status is not CallStatus.LIVE:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Call is {session.status}, not LIVE")
    sequence_header = request.headers.get("x-audio-chunk-sequence")
    try:
        chunk_sequence = int(sequence_header or "")
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="X-Audio-Chunk-Sequence must be an integer") from error
    try:
        container = await read_limited_body(request)
        preprocessing_started = time.perf_counter()
        canonical = await asyncio.to_thread(decode_and_canonicalize, container, request.headers.get("content-type", ""))
        acknowledgement = registry.ingest_canonical(call_id, canonical, chunk_sequence)
        return replace(acknowledgement, preprocessing_ms=(time.perf_counter() - preprocessing_started) * 1000)
    except AudioConversionError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except SessionClosedError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


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

    provider = get_risk_provider()
    runtime_session = None
    if isinstance(provider, MLRiskProvider):
        runtime_session = get_audio_session_registry().get(call_id)
        telemetry = get_acceptance_telemetry_registry().get(call_id)
        if runtime_session is None:
            await websocket.close(code=WS_CALL_NOT_LIVE, reason="ML runtime session is not registered")
            return
        if not get_audio_session_registry().claim_stream(call_id, runtime_session.generation_token):
            await websocket.close(code=WS_CALL_NOT_LIVE, reason="ML risk stream already has an active consumer")
            return
    else:
        from app.services.async_session import AsyncCallSession

        runtime_session = AsyncCallSession(call_id)
    stream_session = runtime_session if isinstance(provider, MLRiskProvider) else session
    cancellation = asyncio.Event()
    try:
        async for event in provider.stream(stream_session, cancellation):
            await websocket.send_json(event.model_dump(mode="json", exclude_none=True))
        await websocket.close()
    except WebSocketDisconnect:
        # Client hung up mid-stream; nothing to clean up beyond stopping.
        return
    except Exception as error:
        await websocket.close(code=1011, reason=f"ML risk stream failed: {error}")
    finally:
        cancellation.set()
        if isinstance(provider, MLRiskProvider) and runtime_session is not None:
            get_audio_session_registry().release_stream(call_id, runtime_session.generation_token)
            await get_audio_session_registry().close(call_id, runtime_session.generation_token)


@router.websocket("/{call_id}/audio-stream")
async def stream_audio(websocket: WebSocket, call_id: str) -> None:
    """Receives one browser microphone producer for a live ML call."""
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
    runtime_session = get_audio_session_registry().get(call_id)
    telemetry = get_acceptance_telemetry_registry().get(call_id)
    if runtime_session is None:
        await websocket.close(code=WS_CALL_NOT_LIVE, reason="ML runtime session is not registered")
        return
    handler = AudioStreamHandler(get_audio_session_registry(), runtime_session, telemetry)
    normal = False
    closed = False
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                    if control.get("type") == "audio_start":
                        metadata = AudioStartMetadata(
                            sample_rate=control.get("sample_rate"),
                            channels=control.get("channels"),
                            sample_format=control.get("sample_format", ""),
                            protocol_version=control.get("protocol_version"),
                        )
                        await handler.start(metadata)
                        await websocket.send_json({"type": "audio_ready", "protocol_version": 1, "call_id": call_id})
                    elif control.get("type") == "audio_stop":
                        normal = True
                        break
                    else:
                        raise AudioProtocolError("unknown audio control message")
                except RuntimeError as error:
                    await websocket.close(code=WS_AUDIO_DUPLICATE, reason=str(error))
                    return
                except (AudioProtocolError, TypeError, ValueError, KeyError) as error:
                    await websocket.close(code=WS_AUDIO_INVALID, reason=str(error))
                    return
            elif message.get("bytes") is not None:
                try:
                    await handler.receive_binary(message["bytes"])
                except (AudioProtocolError, SessionClosedError, ValueError) as error:
                    await websocket.close(code=WS_AUDIO_INVALID, reason=str(error))
                    return
            else:
                await websocket.close(code=WS_AUDIO_INVALID, reason="audio message is empty")
                return
        await handler.close(normal=normal)
        closed = True
        if normal:
            await websocket.send_json({"type": "audio_stopped", "call_id": call_id})
            await websocket.close()
    except WebSocketDisconnect:
        await handler.close(normal=False)
        closed = True
    except Exception as error:
        await handler.close(normal=False)
        closed = True
        await websocket.close(code=1011, reason=f"audio stream failed: {error}")
    finally:
        if not closed:
            await handler.close(normal=False)
