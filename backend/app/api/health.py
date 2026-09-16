"""Service health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def read_health() -> dict[str, str]:
    """Reports that the service process is up."""
    return {"status": "ok", "service": get_settings().service_name}
