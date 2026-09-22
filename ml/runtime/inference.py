"""Asynchronous, bounded execution around verified AASIST and ECAPA adapters."""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
from typing import Protocol

import numpy as np

from ml.runtime.contracts import AASIST_SAMPLE_COUNT, AASIST_SAMPLE_RATE, HealthResponse, SpeakerEmbedRequest, SpeakerEmbedResponse, SpeakerVerifyRequest, SpeakerVerifyResponse, SpoofInferenceRequest, SpoofInferenceResponse
from ml.speaker.verifier import SpeakerEmbedding, SpeakerVerifier

MAX_CONCURRENT_MODEL_FORWARDS = 1
SPOOF_DECISION_THRESHOLD = 0.5


class ExactWindowDetector(Protocol):
    """Protocol required by the runtime without importing model internals."""
    model_id: str

    def score_model_window(self, samples: np.ndarray):
        """Scores one exact model window."""


class AASISTInferenceRuntime:
    """Runs one loaded detector and optional speaker verifier."""

    def __init__(self, detector: ExactWindowDetector | None = None, speaker_verifier: SpeakerVerifier | None = None) -> None:
        """Creates a runtime with bounded independent model gates."""
        self._detector = detector
        self._speaker_verifier = speaker_verifier
        self._forward_gate = asyncio.Semaphore(MAX_CONCURRENT_MODEL_FORWARDS)
        self._speaker_forward_gate = asyncio.Semaphore(1)

    def health(self) -> HealthResponse:
        """Reports readiness of both independently optional models."""
        if self._detector is None:
            return HealthResponse(ready=False, reason="AASIST model is not initialized", speaker_ready=self._speaker_verifier is not None, speaker_model_id=self._speaker_verifier.model_id if self._speaker_verifier else None)
        return HealthResponse(ready=True, model_id=self._detector.model_id, speaker_ready=self._speaker_verifier is not None, speaker_model_id=self._speaker_verifier.model_id if self._speaker_verifier else None)

    @staticmethod
    def _decode_audio(samples_base64: str, minimum_samples: int = 1) -> np.ndarray:
        """Decodes finite little-endian float32 audio with a bounded shape contract."""
        try:
            raw_bytes = base64.b64decode(samples_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("samples_base64 is invalid") from error
        if len(raw_bytes) % np.dtype("<f4").itemsize != 0:
            raise ValueError("audio payload is not float32 aligned")
        samples = np.frombuffer(raw_bytes, dtype="<f4").copy()
        if samples.size < minimum_samples or not np.isfinite(samples).all():
            raise ValueError("audio must contain enough finite samples")
        return samples

    async def infer(self, request: SpoofInferenceRequest) -> SpoofInferenceResponse:
        """Validates transport bytes and asynchronously executes exact-window AASIST inference."""
        preprocessing_start = time.perf_counter()
        health = self.health()
        if not health.ready or self._detector is None:
            raise RuntimeError("AASIST model is unavailable")
        samples = self._decode_audio(request.samples_base64, AASIST_SAMPLE_COUNT)
        if samples.size != AASIST_SAMPLE_COUNT:
            raise ValueError("sample payload does not contain exactly 64600 float32 samples")
        preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000
        async with self._forward_gate:
            inference_start = time.perf_counter()
            result = await asyncio.to_thread(self._detector.score_model_window, samples)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        prediction = "spoof" if result.synthetic_probability >= SPOOF_DECISION_THRESHOLD else "bonafide"
        return SpoofInferenceResponse(model_id=result.model_id, raw_spoof_score=result.synthetic_probability, threshold=SPOOF_DECISION_THRESHOLD, prediction=prediction, audio_duration_ms=AASIST_SAMPLE_COUNT / AASIST_SAMPLE_RATE * 1000, inference_ms=inference_ms, preprocessing_ms=preprocessing_ms, warnings=list(result.warnings))

    async def embed_speaker(self, request: SpeakerEmbedRequest) -> SpeakerEmbedResponse:
        """Creates one normalized ECAPA enrollment embedding."""
        verifier = self._speaker_verifier
        if verifier is None:
            raise RuntimeError("speaker model is unavailable")
        samples = self._decode_audio(request.samples_base64, 32_000)
        async with self._speaker_forward_gate:
            embedding = await asyncio.to_thread(verifier.embed, samples, AASIST_SAMPLE_RATE)
        return SpeakerEmbedResponse(model_id=embedding.model_id, model_revision=getattr(verifier, "model_revision", embedding.model_id.rsplit("@", 1)[-1]), embedding=embedding.vector.astype(np.float32).tolist(), embedding_dimensions=embedding.dimensions, audio_duration_ms=embedding.total_duration_seconds * 1000, inference_ms=embedding.inference_seconds * 1000, preprocessing_ms=embedding.preprocessing_seconds * 1000)

    async def verify_speaker(self, request: SpeakerVerifyRequest) -> SpeakerVerifyResponse:
        """Scores one canonical probe against a stored normalized reference."""
        verifier = self._speaker_verifier
        if verifier is None:
            raise RuntimeError("speaker model is unavailable")
        samples = self._decode_audio(request.samples_base64, 32_000)
        reference_vector = np.asarray(request.reference_embedding, dtype=np.float32)
        if reference_vector.ndim != 1 or reference_vector.size != verifier.embedding_dimensions or not np.isfinite(reference_vector).all():
            raise ValueError("reference embedding dimensions or values are invalid")
        reference = SpeakerEmbedding(vector=reference_vector, model_id=verifier.model_id, source_count=1, total_duration_seconds=1.0, sample_rate=AASIST_SAMPLE_RATE)
        async with self._speaker_forward_gate:
            result = await asyncio.to_thread(verifier.verify, reference, samples, AASIST_SAMPLE_RATE, request.threshold)
        return SpeakerVerifyResponse(model_id=result.model_id, model_revision=getattr(verifier, "model_revision", result.model_id.rsplit("@", 1)[-1]), cosine_similarity=result.similarity_score, threshold=result.threshold, is_match=result.is_match, embedding_dimensions=reference_vector.size, audio_duration_ms=result.probe_duration_seconds * 1000, inference_ms=result.inference_seconds * 1000, preprocessing_ms=result.preprocessing_seconds * 1000)
