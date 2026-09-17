"""Tests for the asynchronous loopback ML client."""

from __future__ import annotations

import asyncio
import base64

import httpx
import numpy as np
import pytest

from app.services.ml_service_client import MLServiceClient, MLServiceMalformedResponseError, MLServiceUnavailableError


def request_client(handler) -> MLServiceClient:
    """Builds a client around an in-memory async HTTP transport."""
    return MLServiceClient("http://ml.test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def valid_response() -> dict[str, object]:
    """Returns a complete raw-evidence service response."""
    return {"model_id": "AASIST/test", "raw_spoof_score": 0.8, "score_semantics": "uncalibrated", "threshold": 0.5, "prediction": "spoof", "audio_duration_ms": 4037.5, "inference_ms": 11.0, "warnings": []}


def test_infer_sends_exact_canonical_window_json() -> None:
    """The client sends a single exact-size float32 model window."""
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        """Captures the request and returns valid evidence."""
        seen["body"] = request.read().decode()
        return httpx.Response(200, json=valid_response())

    async def exercise() -> None:
        """Runs the asynchronous request assertion."""
        client = request_client(handler)
        samples = np.zeros(64_600, dtype=np.float32)
        evidence = await client.infer(samples)
        await client.close()
        assert evidence.raw_spoof_score == 0.8
        assert base64.b64encode(samples.tobytes()).decode() in str(seen["body"])

    asyncio.run(exercise())


def test_health_and_failures_are_typed() -> None:
    """Unavailable and malformed service responses become explicit errors."""
    async def unavailable(_: httpx.Request) -> httpx.Response:
        """Returns a service-unavailable response."""
        return httpx.Response(503)

    async def malformed(_: httpx.Request) -> httpx.Response:
        """Returns an invalid evidence response."""
        return httpx.Response(200, json={"raw_spoof_score": 0.8})

    async def exercise() -> None:
        """Runs typed failure assertions."""
        unavailable_client = request_client(unavailable)
        with pytest.raises(MLServiceUnavailableError):
            await unavailable_client.health()
        await unavailable_client.close()
        malformed_client = request_client(malformed)
        with pytest.raises(MLServiceMalformedResponseError):
            await malformed_client.health()
        await malformed_client.close()

    asyncio.run(exercise())
