"""Async client for the loopback AASIST and ECAPA inference service."""

from __future__ import annotations

import base64
from typing import Any

import httpx
import numpy as np

from app.models.ml_evidence import RawSpoofEvidence

EXPECTED_SAMPLE_COUNT = 64_600
EXPECTED_SAMPLE_RATE = 16_000
SPEAKER_MINIMUM_SAMPLES = 32_000
# Free-tier hosted model cold starts can exceed five seconds before returning
# the first real embedding; keep the timeout bounded but deployment-safe.
REQUEST_TIMEOUT_SECONDS = 30.0


class MLServiceError(RuntimeError):
    """Base error for ML service communication failures."""


class MLServiceUnavailableError(MLServiceError):
    """Raised when the ML service cannot be reached or is not ready."""


class MLServiceMalformedResponseError(MLServiceError):
    """Raised when the ML service returns an invalid contract payload."""


class MLServiceClient:
    """Sends canonical audio to a configured loopback ML service."""

    def __init__(self, base_url: str, http_client: httpx.AsyncClient | None = None, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> None:
        """Creates a reusable asynchronous client with explicit timeouts."""
        if not base_url.strip():
            raise ValueError("ML service URL must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = base_url.rstrip("/")
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    async def health(self) -> None:
        """Verifies AASIST model readiness."""
        try:
            response = await self._http_client.get(f"{self._base_url}/health")
            if response.status_code != 200:
                raise MLServiceUnavailableError(f"ML service health returned HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict) or "ready" not in payload:
                raise MLServiceMalformedResponseError("ML service health response is malformed")
            if payload.get("ready") is not True:
                raise MLServiceUnavailableError("ML service model is not ready")
        except MLServiceError:
            raise
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            raise MLServiceUnavailableError("ML service health check failed") from error

    async def infer(self, window: np.ndarray) -> RawSpoofEvidence:
        """Sends one exact float32 AASIST window and validates raw evidence."""
        samples = np.asarray(window)
        if samples.shape != (EXPECTED_SAMPLE_COUNT,) or samples.dtype != np.dtype("float32") or not np.isfinite(samples).all():
            raise ValueError("window must be finite float32 with exactly 64600 samples")
        payload = {"samples_base64": base64.b64encode(np.ascontiguousarray(samples, dtype="<f4").tobytes()).decode("ascii"), "sample_rate": EXPECTED_SAMPLE_RATE, "channels": 1, "sample_format": "float32le", "sample_count": EXPECTED_SAMPLE_COUNT}
        try:
            response = await self._http_client.post(f"{self._base_url}/v1/spoof/infer", json=payload)
            if response.status_code != 200:
                raise MLServiceUnavailableError(f"ML service inference returned HTTP {response.status_code}")
            return RawSpoofEvidence.model_validate(response.json())
        except MLServiceError:
            raise
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as error:
            raise MLServiceMalformedResponseError("ML service returned malformed evidence") from error

    async def embed_speaker(self, samples: np.ndarray) -> dict[str, Any]:
        """Creates an ECAPA reference embedding from canonical mono audio."""
        payload = self._speaker_payload(samples)
        try:
            response = await self._http_client.post(f"{self._base_url}/v1/speaker/embed", json=payload)
            if response.status_code != 200:
                raise MLServiceUnavailableError(f"speaker embedding returned HTTP {response.status_code}")
            result = response.json()
            if not isinstance(result, dict) or not isinstance(result.get("embedding"), list):
                raise MLServiceMalformedResponseError("speaker embedding response is malformed")
            return result
        except MLServiceError:
            raise
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as error:
            raise MLServiceMalformedResponseError("speaker embedding response is malformed") from error

    async def verify_speaker(self, samples: np.ndarray, reference_embedding: list[float], threshold: float) -> dict[str, Any]:
        """Returns one ECAPA cosine comparison against an enrolled profile."""
        payload = self._speaker_payload(samples)
        payload["reference_embedding"] = reference_embedding
        payload["threshold"] = threshold
        try:
            response = await self._http_client.post(f"{self._base_url}/v1/speaker/verify", json=payload)
            if response.status_code != 200:
                raise MLServiceUnavailableError(f"speaker verification returned HTTP {response.status_code}")
            result = response.json()
            if not isinstance(result, dict) or not isinstance(result.get("model_id"), str) or not isinstance(result.get("model_revision"), str) or not isinstance(result.get("cosine_similarity"), (int, float)):
                raise MLServiceMalformedResponseError("speaker verification response is malformed")
            return result
        except MLServiceError:
            raise
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as error:
            raise MLServiceMalformedResponseError("speaker verification response is malformed") from error

    @staticmethod
    def _speaker_payload(samples: np.ndarray) -> dict[str, Any]:
        """Builds a bounded canonical speaker payload without retaining it."""
        array = np.asarray(samples)
        if array.ndim != 1 or array.size < SPEAKER_MINIMUM_SAMPLES or array.dtype != np.dtype("float32") or not np.isfinite(array).all():
            raise ValueError("speaker audio must be finite mono float32 with at least 32000 samples")
        return {"samples_base64": base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii"), "sample_rate": EXPECTED_SAMPLE_RATE, "channels": 1, "sample_format": "float32le"}

    async def close(self) -> None:
        """Closes the underlying HTTP client when this instance owns it."""
        if self._owns_http_client:
            await self._http_client.aclose()
