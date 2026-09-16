"""Health endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_ok(client: TestClient) -> None:
    """The health endpoint returns the documented body."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "voxsentinel-backend"}
