"""Speaker verifier boundary tests.

Fast, model-free checks of the interface, the result contract, enrolment, and
the similarity and aggregation maths. Real inference is covered separately in
``test_ecapa_integration.py``; nothing here pretends the model works.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("numpy", reason="ML runtime not installed in this environment")
pytest.importorskip("scipy", reason="ML runtime not installed in this environment")
pytest.importorskip("soundfile", reason="ML runtime not installed in this environment")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from ml.audio import preprocessing  # noqa: E402
from ml.audio.preprocessing import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    AudioTooShortError,
    MalformedAudioError,
    SilentAudioError,
)
from ml.speaker.verifier import (  # noqa: E402
    MIN_ENROLLMENT_SECONDS,
    EnrollmentError,
    SpeakerEmbedding,
    SpeakerVerificationResult,
    SpeakerVerifier,
    aggregate,
    cosine_similarity,
)


def tone(seconds: float, frequency: float = 220.0, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Builds a mono sine wave. Plumbing only: this is not speech."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.4 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def unit(values: list[float]) -> np.ndarray:
    """Builds an L2-normalised float32 vector."""
    vector = np.asarray(values, dtype=np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def make_embedding(**overrides) -> SpeakerEmbedding:
    """Builds a valid embedding, with overrides for the field under test."""
    fields = {
        "vector": unit([1.0, 0.0, 0.0, 0.0]),
        "model_id": "stub/v0",
        "source_count": 1,
        "total_duration_seconds": 5.0,
        "sample_rate": TARGET_SAMPLE_RATE,
    }
    return SpeakerEmbedding(**(fields | overrides))


def make_result(**overrides) -> SpeakerVerificationResult:
    """Builds a valid result, with overrides for the field under test."""
    fields = {
        "similarity_score": 0.8,
        "is_match": True,
        "threshold": 0.5,
        "model_id": "stub/v0",
        "reference_duration_seconds": 10.0,
        "probe_duration_seconds": 5.0,
        "reference_utterances": 2,
        "inference_seconds": 0.3,
        "preprocessing_seconds": 0.001,
    }
    return SpeakerVerificationResult(**(fields | overrides))


class StubVerifier(SpeakerVerifier):
    """Maps audio to a deterministic vector without running a model.

    The vector is derived from the audio's own samples, so two different
    waveforms embed differently and the same waveform always embeds the same.
    That is enough to exercise enrolment, aggregation, and the decision logic
    while keeping these tests model-free.
    """

    dimensions = 8

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.seen: list[tuple[int, int]] = []

    @property
    def model_id(self) -> str:
        """Identifies this stub."""
        return "stub/v0"

    @property
    def expected_sample_rate(self) -> int:
        """Stubs operate at the standard rate."""
        return TARGET_SAMPLE_RATE

    @property
    def embedding_dimensions(self) -> int:
        """Length of the stub's vectors."""
        return self.dimensions

    @property
    def decision_threshold(self) -> float:
        """Cut point this stub decides with."""
        return self.threshold

    def embed(self, audio: np.ndarray, sample_rate: int) -> SpeakerEmbedding:
        """Prepares audio like a real verifier, then folds it into a vector."""
        prepared = preprocessing.prepare(audio, sample_rate, TARGET_SAMPLE_RATE)
        self.seen.append((len(prepared.samples), prepared.sample_rate))
        usable = len(prepared.samples) // self.dimensions * self.dimensions
        folded = prepared.samples[:usable].reshape(self.dimensions, -1).sum(axis=1).astype(np.float64)
        folded = folded + 1e-6  # never exactly zero, so the vector is orientable
        return SpeakerEmbedding(
            vector=(folded / np.linalg.norm(folded)).astype(np.float32),
            model_id=self.model_id,
            source_count=1,
            total_duration_seconds=prepared.duration_seconds,
            sample_rate=TARGET_SAMPLE_RATE,
            inference_seconds=0.001,
            preprocessing_seconds=0.001,
            warnings=prepared.warnings,
        )


@pytest.fixture
def verifier() -> StubVerifier:
    """A model-free verifier for interface tests."""
    return StubVerifier()


# --- SpeakerEmbedding contract ----------------------------------------------


def test_embedding_reports_its_dimensions() -> None:
    assert make_embedding().dimensions == 4


@pytest.mark.parametrize("vector", [np.zeros(4, dtype=np.float32), np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)])
def test_embedding_requires_unit_norm(vector: np.ndarray) -> None:
    with pytest.raises(ValueError, match="L2-normalised"):
        make_embedding(vector=vector)


def test_embedding_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        make_embedding(vector=np.array([np.nan, 0.0, 0.0, 1.0], dtype=np.float32))


def test_embedding_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="1-D"):
        make_embedding(vector=np.eye(2, dtype=np.float32))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_count", 0),
        ("total_duration_seconds", 0.0),
        ("total_duration_seconds", -1.0),
        ("sample_rate", 0),
        ("model_id", "  "),
        ("inference_seconds", -0.1),
    ],
)
def test_embedding_rejects_unusable_metadata(field: str, value) -> None:
    with pytest.raises(ValueError):
        make_embedding(**{field: value})


# --- SpeakerVerificationResult contract -------------------------------------


@pytest.mark.parametrize("score", [1.5, -1.5, math.nan, math.inf])
def test_result_requires_a_cosine(score: float) -> None:
    with pytest.raises(ValueError, match="similarity_score"):
        make_result(similarity_score=score, is_match=False, threshold=0.5)


def test_result_rejects_a_decision_that_contradicts_the_score() -> None:
    with pytest.raises(ValueError, match="contradicts"):
        make_result(similarity_score=0.9, threshold=0.5, is_match=False)


def test_result_accepts_a_score_exactly_on_the_threshold_as_a_match() -> None:
    assert make_result(similarity_score=0.5, threshold=0.5, is_match=True).is_match


def test_result_total_seconds_is_the_sum_of_its_stages() -> None:
    result = make_result(preprocessing_seconds=0.25, inference_seconds=0.75)
    assert result.total_seconds == pytest.approx(1.0)


# --- Similarity --------------------------------------------------------------


def test_identical_embeddings_score_one() -> None:
    vector = unit([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_opposite_embeddings_score_minus_one() -> None:
    vector = unit([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(vector, -vector) == pytest.approx(-1.0)


def test_orthogonal_embeddings_score_zero() -> None:
    assert cosine_similarity(unit([1.0, 0.0]), unit([0.0, 1.0])) == pytest.approx(0.0)


def test_similarity_never_escapes_the_cosine_range() -> None:
    # float32 dot products can land a few ULP outside -1..1, which would fail
    # every downstream range check for no real reason.
    rng = np.random.default_rng(0)
    for _ in range(200):
        vector = rng.normal(size=192).astype(np.float32)
        vector = (vector / np.linalg.norm(vector)).astype(np.float32)
        assert -1.0 <= cosine_similarity(vector, vector.copy()) <= 1.0


def test_similarity_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        cosine_similarity(unit([1.0, 0.0]), unit([1.0, 0.0, 0.0]))


def test_similarity_rejects_a_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        cosine_similarity(np.zeros(4), unit([1.0, 0.0, 0.0, 0.0]))


# --- Deterministic aggregation ----------------------------------------------


def test_aggregate_returns_a_unit_vector() -> None:
    result = aggregate([unit([1.0, 1.0, 0.0]), unit([1.0, 0.0, 1.0])])
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)
    assert result.dtype == np.float32


def test_aggregate_of_one_vector_is_that_vector() -> None:
    vector = unit([0.3, -0.5, 0.8])
    assert aggregate([vector]) == pytest.approx(vector, abs=1e-6)


def test_aggregate_is_independent_of_input_order() -> None:
    # float32 addition is not associative, so summing in a different order can
    # give a different reference. Aggregation accumulates in float64 precisely
    # so that two callers enrolling the same utterances cannot disagree.
    rng = np.random.default_rng(7)
    vectors = []
    for _ in range(6):
        vector = rng.normal(size=192).astype(np.float32)
        vectors.append((vector / np.linalg.norm(vector)).astype(np.float32))

    reference = aggregate(vectors)
    for permutation in ([5, 0, 3, 1, 4, 2], [3, 2, 1, 0, 5, 4], list(reversed(range(6)))):
        assert np.array_equal(aggregate([vectors[i] for i in permutation]), reference)


def test_aggregate_is_repeatable() -> None:
    vectors = [unit([1.0, 2.0, 3.0]), unit([3.0, 2.0, 1.0])]
    assert np.array_equal(aggregate(vectors), aggregate(vectors))


def test_aggregate_weights_utterances_equally_regardless_of_scale() -> None:
    # Inputs are normalised before averaging, so a long or loud utterance does
    # not dominate the reference just because its vector is larger.
    small, large = np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1000.0], dtype=np.float32)
    assert aggregate([small, large]) == pytest.approx(unit([1.0, 1.0]), abs=1e-6)


def test_aggregate_refuses_an_empty_reference() -> None:
    with pytest.raises(EnrollmentError, match="no embeddings"):
        aggregate([])


def test_aggregate_refuses_vectors_that_cancel_out() -> None:
    vector = unit([1.0, 0.0])
    with pytest.raises(EnrollmentError, match="cancelled out"):
        aggregate([vector, -vector])


def test_aggregate_refuses_non_finite_vectors() -> None:
    with pytest.raises(EnrollmentError, match="NaN or infinite"):
        aggregate([unit([1.0, 0.0]), np.array([np.nan, 1.0], dtype=np.float32)])


# --- Enrolment ---------------------------------------------------------------


def test_enrol_from_one_utterance(verifier: StubVerifier) -> None:
    reference = verifier.enroll([(tone(5.0), TARGET_SAMPLE_RATE)])
    assert reference.source_count == 1
    assert reference.dimensions == verifier.embedding_dimensions
    assert reference.total_duration_seconds == pytest.approx(5.0)


def test_enrol_from_several_utterances_sums_their_duration(verifier: StubVerifier) -> None:
    reference = verifier.enroll([(tone(4.0), TARGET_SAMPLE_RATE), (tone(6.0, 330.0), TARGET_SAMPLE_RATE)])
    assert reference.source_count == 2
    assert reference.total_duration_seconds == pytest.approx(10.0)


def test_enrol_from_files(verifier: StubVerifier, tmp_path) -> None:
    paths = []
    for index, frequency in enumerate((220.0, 330.0)):
        path = tmp_path / f"ref{index}.wav"
        sf.write(path, tone(4.0, frequency), TARGET_SAMPLE_RATE)
        paths.append(path)
    assert verifier.enroll(paths).source_count == 2


def test_enrol_refuses_an_empty_reference_set(verifier: StubVerifier) -> None:
    with pytest.raises(EnrollmentError, match="at least one"):
        verifier.enroll([])


def test_enrol_fails_loudly_on_silence_rather_than_dropping_it(verifier: StubVerifier) -> None:
    # A reference quietly built from half the audio the caller supplied is a
    # worse outcome than an error.
    silence = np.zeros(TARGET_SAMPLE_RATE * 4, dtype=np.float32)
    with pytest.raises(EnrollmentError, match="unusable"):
        verifier.enroll([(tone(4.0), TARGET_SAMPLE_RATE), (silence, TARGET_SAMPLE_RATE)])


def test_enrol_fails_loudly_on_short_audio(verifier: StubVerifier) -> None:
    with pytest.raises(EnrollmentError, match="unusable"):
        verifier.enroll([(tone(0.4), TARGET_SAMPLE_RATE)])


def test_enrol_warns_when_there_is_too_little_speech(verifier: StubVerifier) -> None:
    reference = verifier.enroll([(tone(1.2), TARGET_SAMPLE_RATE)])
    assert reference.total_duration_seconds < MIN_ENROLLMENT_SECONDS
    assert any("enrolment speech" in warning for warning in reference.warnings)


def test_enrol_carries_preprocessing_warnings_into_the_reference(verifier: StubVerifier) -> None:
    stereo = np.stack([tone(4.0), tone(4.0, 330.0)], axis=1)
    reference = verifier.enroll([(stereo, TARGET_SAMPLE_RATE)])
    assert any("downmixed" in warning for warning in reference.warnings)


# --- Verification decision ---------------------------------------------------


def test_same_audio_matches_itself(verifier: StubVerifier) -> None:
    audio = tone(5.0)
    reference = verifier.enroll([(audio, TARGET_SAMPLE_RATE)])
    result = verifier.verify(reference, audio, TARGET_SAMPLE_RATE)
    assert result.similarity_score == pytest.approx(1.0, abs=1e-5)
    assert result.is_match


def test_different_audio_scores_lower_than_the_same_audio(verifier: StubVerifier) -> None:
    reference = verifier.enroll([(tone(5.0, 220.0), TARGET_SAMPLE_RATE)])
    same = verifier.verify(reference, tone(5.0, 220.0), TARGET_SAMPLE_RATE)
    other = verifier.verify(reference, tone(5.0, 523.0), TARGET_SAMPLE_RATE)
    assert other.similarity_score < same.similarity_score


def test_threshold_decides_the_verdict_and_is_reported(verifier: StubVerifier) -> None:
    reference = verifier.enroll([(tone(5.0, 220.0), TARGET_SAMPLE_RATE)])
    probe = tone(5.0, 523.0)
    strict = verifier.verify(reference, probe, TARGET_SAMPLE_RATE, threshold=0.99)
    lenient = verifier.verify(reference, probe, TARGET_SAMPLE_RATE, threshold=-1.0)
    assert strict.threshold == 0.99 and not strict.is_match
    assert lenient.is_match
    assert strict.similarity_score == pytest.approx(lenient.similarity_score)


@pytest.mark.parametrize("threshold", [1.5, -1.5, math.nan])
def test_an_out_of_range_threshold_is_refused(verifier: StubVerifier, threshold: float) -> None:
    reference = verifier.enroll([(tone(5.0), TARGET_SAMPLE_RATE)])
    with pytest.raises(ValueError, match="threshold"):
        verifier.verify(reference, tone(5.0), TARGET_SAMPLE_RATE, threshold=threshold)


def test_verifier_default_threshold_is_used_when_none_is_given(verifier: StubVerifier) -> None:
    reference = verifier.enroll([(tone(5.0), TARGET_SAMPLE_RATE)])
    assert verifier.verify(reference, tone(5.0), TARGET_SAMPLE_RATE).threshold == verifier.decision_threshold


def test_result_reports_both_durations(verifier: StubVerifier) -> None:
    reference = verifier.enroll([(tone(4.0), TARGET_SAMPLE_RATE), (tone(6.0), TARGET_SAMPLE_RATE)])
    result = verifier.verify(reference, tone(3.0), TARGET_SAMPLE_RATE)
    assert result.reference_duration_seconds == pytest.approx(10.0)
    assert result.probe_duration_seconds == pytest.approx(3.0)
    assert result.reference_utterances == 2


def test_verify_refuses_a_reference_from_another_model(verifier: StubVerifier) -> None:
    foreign = make_embedding(vector=unit([1.0] * 8), model_id="other/v9", sample_rate=TARGET_SAMPLE_RATE)
    with pytest.raises(ValueError, match="not comparable"):
        verifier.verify(foreign, tone(4.0), TARGET_SAMPLE_RATE)


def test_verify_refuses_a_reference_of_the_wrong_size(verifier: StubVerifier) -> None:
    wrong = make_embedding(vector=unit([1.0, 0.0, 0.0]), model_id=verifier.model_id)
    with pytest.raises(ValueError, match="dimensions"):
        verifier.verify(wrong, tone(4.0), TARGET_SAMPLE_RATE)


@pytest.mark.parametrize(
    ("audio", "error"),
    [
        (np.zeros(TARGET_SAMPLE_RATE * 4, dtype=np.float32), SilentAudioError),
        (tone(0.3), AudioTooShortError),
        (np.array([], dtype=np.float32), MalformedAudioError),
        (np.full(TARGET_SAMPLE_RATE * 4, np.nan, dtype=np.float32), MalformedAudioError),
    ],
)
def test_unusable_probes_are_refused_not_scored(verifier: StubVerifier, audio: np.ndarray, error: type) -> None:
    # The dangerous failure here is a confident MISMATCH on audio that never
    # had a chance, so the probe must raise rather than return a verdict.
    reference = verifier.enroll([(tone(5.0), TARGET_SAMPLE_RATE)])
    with pytest.raises(error):
        verifier.verify(reference, audio, TARGET_SAMPLE_RATE)


def test_verify_file_refuses_an_undecodable_probe(verifier: StubVerifier, tmp_path) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF....WAVEjunk")
    reference = verifier.enroll([(tone(5.0), TARGET_SAMPLE_RATE)])
    with pytest.raises(MalformedAudioError):
        verifier.verify_file(reference, broken)


def test_stereo_and_resampled_probes_are_converted_once_and_flagged(verifier: StubVerifier) -> None:
    reference = verifier.enroll([(tone(5.0), TARGET_SAMPLE_RATE)])
    stereo_44k = np.stack([tone(5.0, sample_rate=44_100), tone(5.0, 330.0, sample_rate=44_100)], axis=1)
    result = verifier.verify(reference, stereo_44k, 44_100)
    assert any("downmixed" in w for w in result.warnings)
    assert any("resampled 44100Hz to 16000Hz" in w for w in result.warnings)
    assert verifier.seen[-1][1] == TARGET_SAMPLE_RATE


def test_warnings_are_not_repeated_between_reference_and_probe(verifier: StubVerifier) -> None:
    stereo = np.stack([tone(5.0), tone(5.0, 330.0)], axis=1)
    reference = verifier.enroll([(stereo, TARGET_SAMPLE_RATE)])
    result = verifier.verify(reference, stereo, TARGET_SAMPLE_RATE)
    assert len(result.warnings) == len(set(result.warnings))
