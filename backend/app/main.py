"""VoxSentinel demo backend application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import calls, health
from app.config import get_settings
from app.services.call_service import CallNotFoundError, InvalidTransitionError


def create_app() -> FastAPI:
    """Builds the FastAPI application with routes and error mapping."""
    settings = get_settings()
    app = FastAPI(title="VoxSentinel Backend", version="0.1.0", summary="Demo backend for the VoxSentinel live-call console. Risk output is mock data.")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(calls.router)

    @app.exception_handler(CallNotFoundError)
    async def _handle_call_not_found(_: Request, exc: CallNotFoundError) -> JSONResponse:
        """Maps an unknown call id to 404."""
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": f"Call {exc.args[0]} not found"})

    @app.exception_handler(InvalidTransitionError)
    async def _handle_invalid_transition(_: Request, exc: InvalidTransitionError) -> JSONResponse:
        """Maps an illegal lifecycle transition to 409."""
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    return app


app = create_app()
