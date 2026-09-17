"""FastAPI application for the loopback AASIST inference service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from ml.runtime.contracts import HealthResponse, SpoofInferenceRequest, SpoofInferenceResponse
from ml.runtime.inference import AASISTInferenceRuntime


def create_app(runtime: AASISTInferenceRuntime | None = None) -> FastAPI:
    """Creates the local service with an optionally injected runtime."""
    active_runtime = runtime or AASISTInferenceRuntime()
    app = FastAPI(title="VoxSentinel ML Runtime", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Returns model readiness without loading or fabricating inference."""
        result = active_runtime.health()
        if not result.ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=result.model_dump(mode="json"))
        return result

    @app.post("/v1/spoof/infer", response_model=SpoofInferenceResponse)
    async def infer(request: SpoofInferenceRequest) -> SpoofInferenceResponse:
        """Runs one exact canonical window through the loaded detector."""
        try:
            return await active_runtime.infer(request)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    return app


app = create_app()
