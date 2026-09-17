"""Stateless local runtime for exact-window AASIST inference."""

from ml.runtime.contracts import SpoofInferenceRequest, SpoofInferenceResponse
from ml.runtime.inference import AASISTInferenceRuntime

__all__ = ["AASISTInferenceRuntime", "SpoofInferenceRequest", "SpoofInferenceResponse"]
