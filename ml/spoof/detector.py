"""Spoof detector interface.

This subsystem answers exactly one question:

    how likely is it that this audio is synthetic or replayed?

It deliberately does NOT produce a VoxSentinel 0-100 risk score. Combining this
probability with speaker verification, prosody, and transaction context into a
final risk decision belongs to a later task.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.audio import preprocessing


@dataclass(frozen=True)
class SpoofResult:
    """One spoof-detection verdict for one piece of audio."""

    #: UNCALIBRATED model score in 0..1, higher meaning more synthetic-like.
    #: This is a softmax output, not a calibrated probability: 0.90 means "well
    #: above this model's spoof decision boundary", NOT "90% likely to be fake".
    #: Named to match the frontend's LiveRiskEvent field for eventual wiring.
    synthetic_probability: float
    #: Model's raw bona-fide score (higher means more human). Useful for
    #: threshold tuning and for comparing against published EER operating points.
    bonafide_score: float
    #: Every raw output score the model produced, in model order.
    raw_scores: tuple[float, ...]
    #: Identifier of the model and checkpoint that produced this result.
    model_id: str
    #: Seconds spent in the model forward pass.
    inference_seconds: float
    #: Seconds spent decoding, downmixing, resampling, and padding.
    preprocessing_seconds: float
    #: Duration of the audio actually scored, after resampling.
    audio_duration_seconds: float
    #: Sample rate the model saw.
    sample_rate: int
    #: Anything done to the audio to make it usable, or any reason to distrust
    #: this result. Empty means the audio arrived in the expected shape.
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Rejects results downstream code could not safely consume."""
        probability = self.synthetic_probability
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"synthetic_probability must be between 0 and 1, got {probability}")
        for name in ("inference_seconds", "preprocessing_seconds", "audio_duration_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative number, got {value}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if not self.model_id.strip():
            raise ValueError("model_id is required")

    @property
    def total_seconds(self) -> float:
        """Wall-clock cost of preprocessing plus inference."""
        return self.preprocessing_seconds + self.inference_seconds


class SpoofDetector(ABC):
    """Scores audio for synthetic-speech characteristics."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifies the model and checkpoint behind this detector."""

    @property
    @abstractmethod
    def expected_sample_rate(self) -> int:
        """Sample rate the underlying model was trained on."""

    @abstractmethod
    def score(self, audio: np.ndarray, sample_rate: int) -> SpoofResult:
        """Scores in-memory audio samples."""

    def score_file(self, path: str | Path) -> SpoofResult:
        """Scores an audio file from disk.

        Only decodes here; ``score`` performs the single preprocessing pass, so
        every conversion appears once in the result's warnings.

        Raises ``AudioValidationError`` for input that cannot be judged.
        """
        samples, sample_rate = preprocessing.read_raw(path)
        return self.score(samples, sample_rate)
