"""Asynchronous, stateless execution around the verified AASIST adapter."""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
from typing import Protocol

import numpy as np

from ml.runtime.contracts import AASIST_SAMPLE_COUNT, AASIST_SAMPLE_RATE, HealthResponse, SpoofInferenceRequest, SpoofInferenceResponse

MAX_CONCURRENT_MODEL_FORWARDS = 1
SPOOF_DECISION_THRESHOLD = 0.5


class ExactWindowDetector(Protocol):
    """Protocol required by the runtime without importing model internals."""

    model_id: str

    def score_model_window(self, samples: np.ndarray):
        """Scores one exact canonical model window."""


class AASISTInferenceRuntime:
    """Runs one loaded detector with bounded asynchronous model execution."""

    def __init__(self, detector: ExactWindowDetector | None = None) -> None:
        """Creates a ready runtime only when a detector is injected."""
        self._detector = detector
        self._forward_gate = asyncio.Semaphore(MAX_CONCURRENT_MODEL_FORWARDS)

    def health(self) -> HealthResponse:
        """Reports whether a verified detector is loaded and ready."""
        if self._detector is None:
            return HealthResponse(ready=False, reason="AASIST model is not initialized")
        return HealthResponse(ready=True, model_id=self._detector.model_id)

    async def infer(self, request: SpoofInferenceRequest) -> SpoofInferenceResponse:
        """Validates transport bytes and asynchronously executes exact-window inference."""
        preprocessing_start = time.perf_counter()
        health = self.health()
        if not health.ready or self._detector is None:
            raise RuntimeError("AASIST model is unavailable")
        try:
            raw_bytes = base64.b64decode(request.samples_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("samples_base64 is invalid") from error
        expected_bytes = AASIST_SAMPLE_COUNT * np.dtype("<f4").itemsize
        if len(raw_bytes) != expected_bytes:
            raise ValueError("sample payload does not contain exactly 64600 float32 samples")
        samples = np.frombuffer(raw_bytes, dtype="<f4").copy()
        if samples.shape != (AASIST_SAMPLE_COUNT,) or not np.isfinite(samples).all():
            raise ValueError("samples must be finite exact-size float32 values")
        preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000
        async with self._forward_gate:
            inference_start = time.perf_counter()
            result = await asyncio.to_thread(self._detector.score_model_window, samples)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        prediction = "spoof" if result.synthetic_probability >= SPOOF_DECISION_THRESHOLD else "bonafide"
        return SpoofInferenceResponse(
            model_id=result.model_id,
            raw_spoof_score=result.synthetic_probability,
            threshold=SPOOF_DECISION_THRESHOLD,
            prediction=prediction,
            audio_duration_ms=AASIST_SAMPLE_COUNT / AASIST_SAMPLE_RATE * 1000,
            inference_ms=inference_ms,
            preprocessing_ms=preprocessing_ms,
            warnings=list(result.warnings),
        )
