"""FastAPI application for loopback AASIST and ECAPA inference."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, status

from ml.runtime.contracts import HealthResponse, SpeakerEmbedRequest, SpeakerEmbedResponse, SpeakerVerifyRequest, SpeakerVerifyResponse, SpoofInferenceRequest, SpoofInferenceResponse
from ml.runtime.inference import AASISTInferenceRuntime
from ml.speaker.ecapa import ECAPASpeakerVerifier, ModelNotInstalledError as SpeakerModelNotInstalledError
from ml.spoof.aasist import AASISTSpoofDetector, ModelNotInstalledError


def load_verified_runtime() -> AASISTInferenceRuntime:
    """Loads verified model assets, keeping unavailable dimensions explicit."""
    try:
        detector = AASISTSpoofDetector()
    except (ModelNotInstalledError, OSError, RuntimeError):
        return AASISTInferenceRuntime()
    try:
        speaker_verifier = ECAPASpeakerVerifier()
    except (SpeakerModelNotInstalledError, OSError, RuntimeError):
        speaker_verifier = None
    return AASISTInferenceRuntime(detector=detector, speaker_verifier=speaker_verifier)


def create_app(runtime: AASISTInferenceRuntime | None = None) -> FastAPI:
    """Creates the local ML service with optional injected runtime."""
    active_runtime = runtime or load_verified_runtime()
    app = FastAPI(title="VoxSentinel ML Runtime", version="0.2.0")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Returns model readiness without fabricating unavailable evidence."""
        result = active_runtime.health()
        requires_speaker = os.getenv("VOXSENTINEL_REQUIRE_SPEAKER_MODEL", "false").strip().lower() == "true"
        if not result.ready or (requires_speaker and not result.speaker_ready):
            if requires_speaker and result.ready and not result.speaker_ready:
                result = result.model_copy(update={"ready": False, "reason": "ECAPA speaker model is not initialized"})
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=result.model_dump(mode="json"))
        return result

    @app.post("/v1/spoof/infer", response_model=SpoofInferenceResponse)
    async def infer(request: SpoofInferenceRequest) -> SpoofInferenceResponse:
        """Runs one exact canonical window through AASIST."""
        try:
            return await active_runtime.infer(request)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    @app.post("/v1/speaker/embed", response_model=SpeakerEmbedResponse)
    async def embed_speaker(request: SpeakerEmbedRequest) -> SpeakerEmbedResponse:
        """Creates a reference embedding and immediately discards source samples."""
        try:
            return await active_runtime.embed_speaker(request)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    @app.post("/v1/speaker/verify", response_model=SpeakerVerifyResponse)
    async def verify_speaker(request: SpeakerVerifyRequest) -> SpeakerVerifyResponse:
        """Returns uncalibrated ECAPA cosine similarity for a probe."""
        try:
            return await active_runtime.verify_speaker(request)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    return app


app = create_app()
