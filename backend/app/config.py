"""Runtime settings read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

SERVICE_NAME = "voxsentinel-backend"
DEFAULT_EMIT_INTERVAL_MS = 700
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


@dataclass(frozen=True)
class Settings:
    """Process configuration for the demo backend."""

    service_name: str
    risk_emit_interval_ms: int
    cors_origins: tuple[str, ...]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Builds cached settings; call ``get_settings.cache_clear()`` in tests."""
    return Settings(
        service_name=SERVICE_NAME,
        risk_emit_interval_ms=int(os.getenv("VOXSENTINEL_RISK_EMIT_INTERVAL_MS", str(DEFAULT_EMIT_INTERVAL_MS))),
        cors_origins=tuple(origin.strip() for origin in os.getenv("VOXSENTINEL_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if origin.strip()),
    )
