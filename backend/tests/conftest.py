"""Shared fixtures for backend tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.calls import get_risk_provider, get_session_store
from app.config import get_settings
from app.main import create_app

TRANSFER_ATTACK_CALL = {
    "claimed_identity": "CEO Demo",
    "scenario": "HIGH_VALUE_TRANSFER_ATTACK",
    "transaction_value": 2500000,
    "currency": "INR",
}


def _reset_process_state() -> None:
    """Clears cached singletons so each test starts from a clean process."""
    get_settings.cache_clear()
    get_session_store.cache_clear()
    get_risk_provider.cache_clear()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Yields a test client with risk events emitted without pacing delay."""
    monkeypatch.setenv("VOXSENTINEL_RISK_EMIT_INTERVAL_MS", "0")
    _reset_process_state()
    with TestClient(create_app()) as test_client:
        yield test_client
    _reset_process_state()


@pytest.fixture
def live_call_id(client: TestClient) -> str:
    """Creates and starts a HIGH_VALUE_TRANSFER_ATTACK call, returning its id."""
    call_id = client.post("/api/v1/calls", json=TRANSFER_ATTACK_CALL).json()["call_id"]
    client.post(f"/api/v1/calls/{call_id}/start")
    return call_id
