"""Runtime settings read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

SERVICE_NAME = "voxsentinel-backend"
DEFAULT_EMIT_INTERVAL_MS = 700
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


class RiskProviderMode(StrEnum):
    """Supported backend risk-provider modes."""

    MOCK = "mock"
    ML = "ml"


@dataclass(frozen=True)
class Settings:
    """Process configuration for the demo backend."""

    service_name: str
    risk_emit_interval_ms: int
    cors_origins: tuple[str, ...]
    risk_provider_mode: RiskProviderMode


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Builds cached settings; call ``get_settings.cache_clear()`` in tests."""
    provider_value = os.getenv("VOXSENTINEL_RISK_PROVIDER", RiskProviderMode.MOCK.value).strip().lower()
    try:
        provider_mode = RiskProviderMode(provider_value)
    except ValueError as error:
        raise ValueError("VOXSENTINEL_RISK_PROVIDER must be one of: mock, ml") from error
    return Settings(
        service_name=SERVICE_NAME,
        risk_emit_interval_ms=int(os.getenv("VOXSENTINEL_RISK_EMIT_INTERVAL_MS", str(DEFAULT_EMIT_INTERVAL_MS))),
        cors_origins=tuple(origin.strip() for origin in os.getenv("VOXSENTINEL_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if origin.strip()),
        risk_provider_mode=provider_mode,
    )
