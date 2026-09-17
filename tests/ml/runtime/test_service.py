"""Tests for the asynchronous stateless ML inference runtime."""

from __future__ import annotations

import asyncio

import numpy as np

from ml.runtime.contracts import SpoofInferenceRequest
from ml.runtime.inference import AASISTInferenceRuntime


class DetectorDouble:
    """Ready detector double used without loading Torch in unit tests."""

    model_id = "test-detector"

    def score_model_window(self, samples: np.ndarray):
        """Returns a minimal detector result for the exact input window."""
        from ml.spoof.detector import SpoofResult

        assert samples.shape == (64_600,)
        return SpoofResult(
            synthetic_probability=0.75,
            bonafide_score=-1.0,
            raw_scores=(1.0, -1.0),
            model_id=self.model_id,
            inference_seconds=0.01,
            preprocessing_seconds=0.0,
            audio_duration_seconds=4.0375,
            sample_rate=16_000,
            warnings=(),
        )


def test_runtime_health_requires_a_ready_detector() -> None:
    """Uninitialized runtime reports model unavailability."""
    assert AASISTInferenceRuntime().health().ready is False
    assert AASISTInferenceRuntime(detector=DetectorDouble()).health().ready is True


def test_runtime_calls_exact_window_detector() -> None:
    """Runtime decodes canonical bytes and returns measured model evidence."""
    import base64

    request = SpoofInferenceRequest(samples_base64=base64.b64encode(np.zeros(64_600, dtype="<f4").tobytes()).decode("ascii"))
    response = asyncio.run(AASISTInferenceRuntime(detector=DetectorDouble()).infer(request))
    assert response.raw_spoof_score == 0.75
    assert response.score_semantics == "uncalibrated"
    assert response.model_id == "test-detector"
