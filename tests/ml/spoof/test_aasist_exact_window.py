"""Unit tests for the exact-window AASIST adapter boundary."""

from __future__ import annotations

import numpy as np
import pytest

from ml.spoof.aasist import AASISTSpoofDetector


def test_exact_window_rejects_wrong_shape_before_model_access() -> None:
    """The service-facing method accepts only one exact canonical window."""
    detector = object.__new__(AASISTSpoofDetector)
    with pytest.raises(ValueError, match="64600"):
        detector.score_model_window(np.zeros(64_599, dtype=np.float32))
