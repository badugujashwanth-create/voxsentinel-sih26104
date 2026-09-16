"""Spoof detector boundary tests.

Fast, model-free checks of the interface, the result contract, and the padding
logic. Real inference is covered separately in ``test_aasist_integration.py``;
nothing here pretends the model works.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="ML runtime not installed in this environment")
pytest.importorskip("scipy", reason="ML runtime not installed in this environment")
pytest.importorskip("soundfile", reason="ML runtime not installed in this environment")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from ml.audio import preprocessing  # noqa: E402
from ml.audio.preprocessing import TARGET_SAMPLE_RATE, AudioTooShortError, SilentAudioError  # noqa: E402
from ml.spoof.detector import SpoofDetector, SpoofResult  # noqa: E402

def tone(seconds: float, sample_rate: int = TARGET_SAMPLE_RATE, amplitude: float = 0.4) -> np.ndarray:
    """Builds a mono sine wave."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def make_result(**overrides) -> SpoofResult:
    """Builds a valid result, with overrides for the field under test."""
    fields = {
        "synthetic_probability": 0.5,
        "bonafide_score": 0.0,
        "raw_scores": (0.0, 0.0),
        "model_id": "test-model",
        "inference_seconds": 0.01,
        "preprocessing_seconds": 0.002,
        "audio_duration_seconds": 4.0,
        "sample_rate": TARGET_SAMPLE_RATE,
    }
    return SpoofResult(**(fields | overrides))


class StubDetector(SpoofDetector):
    """Records what the base class hands it, without running a model."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    @property
    def model_id(self) -> str:
        """Identifies this stub."""
        return "stub/v0"

    @property
    def expected_sample_rate(self) -> int:
        """Stubs operate at the standard rate."""
        return TARGET_SAMPLE_RATE

    def score(self, audio: np.ndarray, sample_rate: int) -> SpoofResult:
        """Prepares audio like a real detector, then records what it saw."""
        prepared = preprocessing.prepare(audio, sample_rate, self.expected_sample_rate)
        self.seen.append((len(prepared.samples), prepared.sample_rate))
        return make_result(audio_duration_seconds=prepared.duration_seconds)


def test_the_interface_cannot_be_instantiated_directly() -> None:
    """SpoofDetector is abstract."""
    with pytest.raises(TypeError):
        SpoofDetector()  # type: ignore[abstract]


def test_total_seconds_sums_preprocessing_and_inference() -> None:
    """Reported total latency is the sum of its parts."""
    result = make_result(preprocessing_seconds=0.004, inference_seconds=0.02)
    assert result.total_seconds == pytest.approx(0.024)


@pytest.mark.parametrize("probability", [-0.001, 1.001, float("nan"), float("inf")])
def test_probability_outside_zero_to_one_is_rejected(probability: float) -> None:
    """A probability the contract cannot honour fails at construction."""
    with pytest.raises(ValueError):
        make_result(synthetic_probability=probability)


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_probability_inside_the_range_is_accepted(probability: float) -> None:
    """The inclusive bounds are valid."""
    assert make_result(synthetic_probability=probability).synthetic_probability == probability


def test_negative_timings_are_rejected() -> None:
    """Latency cannot be negative."""
    with pytest.raises(ValueError):
        make_result(inference_seconds=-0.1)


def test_score_file_resamples_before_reaching_the_model(tmp_path) -> None:
    """The base class hands the model audio at its expected rate."""
    path = tmp_path / "sample.wav"
    sf.write(path, tone(2.0, sample_rate=44_100), 44_100)

    detector = StubDetector()
    detector.score_file(path)

    samples_seen, rate_seen = detector.seen[0]
    assert rate_seen == TARGET_SAMPLE_RATE
    assert abs(samples_seen - 2 * TARGET_SAMPLE_RATE) < 0.02 * TARGET_SAMPLE_RATE


def test_score_file_downmixes_stereo(tmp_path) -> None:
    """Stereo input reaches the model as one channel."""
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([tone(2.0), tone(2.0)], axis=1), TARGET_SAMPLE_RATE)

    detector = StubDetector()
    detector.score_file(path)
    assert detector.seen[0] == (2 * TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE)


def test_score_file_refuses_silence_without_calling_the_model(tmp_path) -> None:
    """Unusable audio never reaches the model."""
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.float32), TARGET_SAMPLE_RATE)

    detector = StubDetector()
    with pytest.raises(SilentAudioError):
        detector.score_file(path)
    assert detector.seen == []


def test_score_file_refuses_too_short_audio_without_calling_the_model(tmp_path) -> None:
    """Short clips are rejected before inference, not scored confidently."""
    path = tmp_path / "short.wav"
    sf.write(path, tone(0.3), TARGET_SAMPLE_RATE)

    detector = StubDetector()
    with pytest.raises(AudioTooShortError):
        detector.score_file(path)
    assert detector.seen == []