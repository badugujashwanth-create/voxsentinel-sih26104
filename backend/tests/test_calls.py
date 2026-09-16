"""Call session lifecycle tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


GENUINE_CALL = {"claimed_identity": "Arjun Mehta", "scenario": "GENUINE"}
TRANSFER_CALL = {"claimed_identity": "CEO Demo", "scenario": "HIGH_VALUE_TRANSFER_ATTACK", "transaction_value": 2500000, "currency": "INR"}


def test_create_call_returns_created_session(client: TestClient) -> None:
    """Creating a call echoes the identity and scenario in CREATED."""
    response = client.post("/api/v1/calls", json=TRANSFER_CALL)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "CREATED"
    assert body["claimed_identity"] == "CEO Demo"
    assert body["scenario"] == "HIGH_VALUE_TRANSFER_ATTACK"
    assert body["call_id"]


def test_get_call_returns_full_state(client: TestClient) -> None:
    """Fetching a call exposes the stored transaction context."""
    call_id = client.post("/api/v1/calls", json=TRANSFER_CALL).json()["call_id"]
    body = client.get(f"/api/v1/calls/{call_id}").json()
    assert body["call_id"] == call_id
    assert body["status"] == "CREATED"
    assert body["transaction_value"] == 2500000
    assert body["currency"] == "INR"
    assert body["created_at"]
    assert body["started_at"] is None


def test_unknown_call_returns_404(client: TestClient) -> None:
    """An unknown call id is a 404 on every read path."""
    assert client.get("/api/v1/calls/does-not-exist").status_code == 404
    assert client.post("/api/v1/calls/does-not-exist/start").status_code == 404
    assert client.post("/api/v1/calls/does-not-exist/stop").status_code == 404


def test_start_moves_call_to_live(client: TestClient) -> None:
    """Starting a created call makes it LIVE and stamps started_at."""
    call_id = client.post("/api/v1/calls", json=GENUINE_CALL).json()["call_id"]
    body = client.post(f"/api/v1/calls/{call_id}/start").json()
    assert body["status"] == "LIVE"
    assert body["started_at"]


def test_stop_moves_live_call_to_completed(client: TestClient) -> None:
    """Stopping a live call completes it and stamps completed_at."""
    call_id = client.post("/api/v1/calls", json=GENUINE_CALL).json()["call_id"]
    client.post(f"/api/v1/calls/{call_id}/start")
    body = client.post(f"/api/v1/calls/{call_id}/stop").json()
    assert body["status"] == "COMPLETED"
    assert body["completed_at"]


def test_stopping_a_call_that_never_started_is_rejected(client: TestClient) -> None:
    """CREATED cannot jump straight to COMPLETED."""
    call_id = client.post("/api/v1/calls", json=GENUINE_CALL).json()["call_id"]
    response = client.post(f"/api/v1/calls/{call_id}/stop")
    assert response.status_code == 409
    assert "CREATED" in response.json()["detail"]


def test_completed_call_cannot_restart(client: TestClient) -> None:
    """COMPLETED is terminal, so restarting is a conflict."""
    call_id = client.post("/api/v1/calls", json=GENUINE_CALL).json()["call_id"]
    client.post(f"/api/v1/calls/{call_id}/start")
    client.post(f"/api/v1/calls/{call_id}/stop")
    response = client.post(f"/api/v1/calls/{call_id}/start")
    assert response.status_code == 409


def test_double_start_is_rejected(client: TestClient) -> None:
    """A call already LIVE cannot transition to LIVE again."""
    call_id = client.post("/api/v1/calls", json=GENUINE_CALL).json()["call_id"]
    client.post(f"/api/v1/calls/{call_id}/start")
    assert client.post(f"/api/v1/calls/{call_id}/start").status_code == 409


def test_unknown_scenario_is_rejected(client: TestClient) -> None:
    """An unsupported scenario fails validation."""
    response = client.post("/api/v1/calls", json={"claimed_identity": "X", "scenario": "NOT_A_SCENARIO"})
    assert response.status_code == 422


def test_blank_claimed_identity_is_rejected(client: TestClient) -> None:
    """Claimed identity is required at the trust boundary."""
    response = client.post("/api/v1/calls", json={"claimed_identity": "", "scenario": "GENUINE"})
    assert response.status_code == 422
