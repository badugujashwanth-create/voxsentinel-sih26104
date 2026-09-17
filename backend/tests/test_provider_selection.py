"""Tests for risk-provider configuration validation."""

from __future__ import annotations

import pytest

from app.config import RiskProviderMode, get_settings


@pytest.mark.parametrize("value, expected", [("mock", RiskProviderMode.MOCK), ("ml", RiskProviderMode.ML)])
def test_provider_mode_accepts_supported_values(monkeypatch: pytest.MonkeyPatch, value: str, expected: RiskProviderMode) -> None:
    """Supported provider names are parsed without importing the ML runtime."""
    monkeypatch.setenv("VOXSENTINEL_RISK_PROVIDER", value)
    get_settings.cache_clear()
    assert get_settings().risk_provider_mode is expected


def test_provider_mode_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown provider values fail instead of silently selecting mock mode."""
    monkeypatch.setenv("VOXSENTINEL_RISK_PROVIDER", "unknown")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="VOXSENTINEL_RISK_PROVIDER"):
        get_settings()
