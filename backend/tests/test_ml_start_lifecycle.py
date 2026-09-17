"""Tests for transactional ML start lifecycle behavior."""

from __future__ import annotations

import asyncio

import pytest

from app.services.async_session import AudioSessionRegistry
from app.services.risk_provider import MLRiskProvider


class UnavailableClient:
    """ML client double that fails readiness."""

    async def health(self) -> None:
        """Raises when the service is unavailable."""
        raise RuntimeError("ML service unavailable")


def test_failed_ml_start_creates_no_runtime_session() -> None:
    """Readiness failure occurs before registration and leaves no orphan."""
    async def exercise() -> None:
        """Runs the asynchronous transactional-start assertion."""
        registry = AudioSessionRegistry()
        provider = MLRiskProvider(registry=registry, client=UnavailableClient())
        with pytest.raises(RuntimeError, match="unavailable"):
            await provider.prepare_session("call-1")
        assert registry.get("call-1") is None

    asyncio.run(exercise())
