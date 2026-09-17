"""Live risk WebSocket tests."""

from __future__ import annotations

from itertools import pairwise

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from app.api.calls import WS_CALL_NOT_LIVE, WS_UNKNOWN_CALL
from app.models.risk import LiveRiskEvent, RecommendedAction, RiskLevel

TRANSFER_PROGRESSION = [18, 27, 43, 61, 79, 92]


def drain(client: TestClient, call_id: str) -> list[dict]:
    """Reads every risk event a call emits until the server closes."""
    payloads: list[dict] = []
    with client.websocket_connect(f"/api/v1/calls/{call_id}/risk-stream") as websocket:
        try:
            while True:
                payloads.append(websocket.receive_json())
        except WebSocketDisconnect:
            pass
    return payloads


def test_stream_emits_the_full_transfer_attack_progression(client: TestClient, live_call_id: str) -> None:
    """The primary demo scenario streams its agreed progression."""
    payloads = drain(client, live_call_id)
    assert [payload["overall_risk_score"] for payload in payloads] == TRANSFER_PROGRESSION


def test_stream_payloads_satisfy_the_frontend_contract(client: TestClient, live_call_id: str) -> None:
    """Every payload re-validates against the shared event model."""
    for payload in drain(client, live_call_id):
        event = LiveRiskEvent.model_validate(payload)
        assert event.call_id == live_call_id
        assert set(payload) == set(LiveRiskEvent.model_fields) - {"evidence_availability", "synthetic_score_semantics"}


def test_sequence_and_timestamps_increase_monotonically(client: TestClient, live_call_id: str) -> None:
    """Sequence starts at 1 and both ordering fields strictly increase."""
    payloads = drain(client, live_call_id)
    sequences = [payload["sequence"] for payload in payloads]
    timestamps = [payload["timestamp_ms"] for payload in payloads]
    assert sequences == list(range(1, len(payloads) + 1))
    assert all(later > earlier for earlier, later in pairwise(timestamps))


def test_final_event_blocks_the_transfer(client: TestClient, live_call_id: str) -> None:
    """The stream ends CRITICAL with BLOCK_ACTION."""
    final = LiveRiskEvent.model_validate(drain(client, live_call_id)[-1])
    assert final.risk_level is RiskLevel.CRITICAL
    assert final.recommended_action is RecommendedAction.BLOCK_ACTION


def test_unknown_call_is_closed_cleanly(client: TestClient) -> None:
    """An unknown call id closes with the unknown-call code."""
    with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect("/api/v1/calls/nope/risk-stream") as websocket:
        websocket.receive_json()
    assert excinfo.value.code == WS_UNKNOWN_CALL


def test_call_that_never_started_is_not_streamed(client: TestClient) -> None:
    """A CREATED call must not behave as if it were LIVE."""
    call_id = client.post("/api/v1/calls", json={"claimed_identity": "CEO Demo", "scenario": "GENUINE"}).json()["call_id"]
    with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect(f"/api/v1/calls/{call_id}/risk-stream") as websocket:
        websocket.receive_json()
    assert excinfo.value.code == WS_CALL_NOT_LIVE


def test_completed_call_is_not_streamed(client: TestClient, live_call_id: str) -> None:
    """A completed call stops being streamable."""
    client.post(f"/api/v1/calls/{live_call_id}/stop")
    with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect(f"/api/v1/calls/{live_call_id}/risk-stream") as websocket:
        websocket.receive_json()
    assert excinfo.value.code == WS_CALL_NOT_LIVE


def test_early_client_disconnect_does_not_break_the_server(client: TestClient, live_call_id: str) -> None:
    """Hanging up mid-stream leaves the service healthy."""
    with client.websocket_connect(f"/api/v1/calls/{live_call_id}/risk-stream") as websocket:
        websocket.receive_json()

    assert client.get("/health").status_code == 200
    assert drain(client, live_call_id)[-1]["overall_risk_score"] == TRANSFER_PROGRESSION[-1]


def test_each_scenario_streams_its_own_progression(client: TestClient) -> None:
    """Every demo scenario is reachable over the socket."""
    expected = {
        "GENUINE": [12, 10, 14, 11, 13],
        "HUMAN_IMPOSTOR": [18, 26, 39, 54, 67, 76],
        "AI_CLONE": [20, 31, 46, 63, 78, 88],
        "HIGH_VALUE_TRANSFER_ATTACK": TRANSFER_PROGRESSION,
    }
    for scenario, progression in expected.items():
        call_id = client.post("/api/v1/calls", json={"claimed_identity": "CEO Demo", "scenario": scenario}).json()["call_id"]
        client.post(f"/api/v1/calls/{call_id}/start")
        assert [payload["overall_risk_score"] for payload in drain(client, call_id)] == progression
