"""Speaker verification interface.

This subsystem answers exactly one question:

    does this audio come from the speaker who was enrolled?

That is a different question from the one ``ml/spoof/`` answers. The spoof
detector asks "is this audio synthetic?"; this asks "is this the same voice?".
A perfect answer here says nothing about whether the voice is real, and the
demonstration in ``ml/speaker/README.md`` shows why: a text-to-speech model
trained on the enrolled speaker scores like the enrolled speaker.

The output is a **cosine similarity**, not a probability. Nothing here is
calibrated, and nothing here is a VoxSentinel risk score. Fusing this with the
spoof score and transaction context belongs to a later task.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.audio import preprocessing

#: Total enrolled speech below this is too little to characterise a voice, so
#: the embedding is produced but flagged. Chosen to match the shortest single
#: utterance the evaluation set uses; it is a warning line, not a measured
#: operating point.
MIN_ENROLLMENT_SECONDS = 3.0

#: A probe shorter than this still produces an embedding, but a short window
#: captures little speaker information and the similarity moves around a lot.
MIN_RELIABLE_PROBE_SECONDS = 2.0

#: Tolerance on the L2 norm of a unit vector after float32 rounding.
_UNIT_NORM_TOLERANCE = 1e-3


class EnrollmentError(ValueError):
    """Raised when enrolment audio cannot produce a usable reference."""


@dataclass(frozen=True)
class SpeakerEmbedding:
    """A speaker reference: one L2-normalised vector plus its provenance.

    Normalised on construction so that cosine similarity is a dot product and
    so that averaging several utterances weights them equally rather than by
    loudness.
    """

    #: Unit-norm float32 vector of shape ``(dimensions,)``.
    vector: np.ndarray
    #: Identifier of the model and checkpoint that produced this vector.
    model_id: str
    #: How many utterances were aggregated into it.
    source_count: int
    #: Total seconds of speech behind it, after resampling.
    total_duration_seconds: float
    #: Sample rate the model saw.
    sample_rate: int
    #: Seconds spent in the forward pass, summed over the source utterances.
    inference_seconds: float = 0.0
    #: Seconds spent decoding, downmixing, and resampling, summed likewise.
    preprocessing_seconds: float = 0.0
    #: Anything done to the audio, or any reason to distrust this reference.
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Rejects references downstream code could not safely compare."""
        vector = np.asarray(self.vector)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"embedding must be a non-empty 1-D vector, got shape {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("embedding contains NaN or infinite values")
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ValueError(f"embedding must be L2-normalised, got norm {norm:.6f}")
        if self.source_count < 1:
            raise ValueError(f"source_count must be at least 1, got {self.source_count}")
        if not math.isfinite(self.total_duration_seconds) or self.total_duration_seconds <= 0:
            raise ValueError(f"total_duration_seconds must be positive, got {self.total_duration_seconds}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        for name in ("inference_seconds", "preprocessing_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative number, got {value}")
        if not self.model_id.strip():
            raise ValueError("model_id is required")

    @property
    def dimensions(self) -> int:
        """Length of the embedding vector."""
        return int(np.asarray(self.vector).shape[0])


@dataclass(frozen=True)
class SpeakerVerificationResult:
    """One match/mismatch verdict for one reference against one probe."""

    #: Cosine similarity in -1..1, higher meaning more like the reference.
    #: UNCALIBRATED. 0.7 does not mean "70% likely the same person"; it means
    #: "this far above the threshold below". It carries no probability meaning.
    similarity_score: float
    #: ``similarity_score >= threshold``. A decision, not a confidence.
    is_match: bool
    #: The cut point this decision used.
    threshold: float
    #: Identifier of the model and checkpoint behind both embeddings.
    model_id: str
    #: Seconds of enrolled speech behind the reference.
    reference_duration_seconds: float
    #: Seconds of probe speech scored, after resampling.
    probe_duration_seconds: float
    #: How many utterances the reference aggregates.
    reference_utterances: int
    #: Seconds spent in the probe's forward pass.
    inference_seconds: float
    #: Seconds spent decoding, downmixing, and resampling the probe.
    preprocessing_seconds: float
    #: Anything done to the audio, or any reason to distrust this verdict.
    #: Includes the reference's own warnings.
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Rejects verdicts that are internally inconsistent or out of range."""
        score = self.similarity_score
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise ValueError(f"similarity_score must be a cosine in -1..1, got {score}")
        if not math.isfinite(self.threshold) or not -1.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in -1..1, got {self.threshold}")
        if self.is_match != (score >= self.threshold):
            raise ValueError(f"is_match={self.is_match} contradicts score {score} against threshold {self.threshold}")
        for name in ("reference_duration_seconds", "probe_duration_seconds", "inference_seconds", "preprocessing_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative number, got {value}")
        if self.reference_utterances < 1:
            raise ValueError(f"reference_utterances must be at least 1, got {self.reference_utterances}")
        if not self.model_id.strip():
            raise ValueError("model_id is required")

    @property
    def total_seconds(self) -> float:
        """Wall-clock cost of preprocessing plus inference for the probe."""
        return self.preprocessing_seconds + self.inference_seconds


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Returns the cosine of the angle between two embeddings.

    Clipped to -1..1: the dot product of two unit float32 vectors can land a
    few ULP outside that range, and a similarity of 1.0000000002 would fail
    every downstream range check for no real reason.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"cannot compare embeddings of shape {a.shape} and {b.shape}")
    norms = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if norms == 0.0:
        raise ValueError("cannot compare a zero-length embedding")
    return float(np.clip(float(a @ b) / norms, -1.0, 1.0))


def aggregate(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Averages unit embeddings into one unit embedding, deterministically.

    Each input is normalised first, so a loud utterance does not outweigh a
    quiet one, and the mean is accumulated in float64. That last detail is what
    makes the result independent of the order the utterances arrive in:
    float32 addition is not associative, so summing in float32 would let the
    same two utterances produce two different references depending on which was
    read first.
    """
    if not vectors:
        raise EnrollmentError("no embeddings to aggregate")
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vectors])
    if not np.all(np.isfinite(stacked)):
        raise EnrollmentError("embedding contains NaN or infinite values")
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise EnrollmentError("cannot aggregate a zero-length embedding")
    centroid = np.mean(stacked / norms, axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm == 0.0:
        raise EnrollmentError("enrolment utterances cancelled out; references are not from one speaker")
    return (centroid / centroid_norm).astype(np.float32)


class SpeakerVerifier(ABC):
    """Compares a probe utterance against an enrolled speaker reference."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifies the model and checkpoint behind this verifier."""

    @property
    @abstractmethod
    def expected_sample_rate(self) -> int:
        """Sample rate the underlying model was trained on."""

    @property
    @abstractmethod
    def embedding_dimensions(self) -> int:
        """Length of the vectors this verifier produces."""

    @property
    @abstractmethod
    def decision_threshold(self) -> float:
        """Default cosine cut point separating match from mismatch."""

    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int) -> SpeakerEmbedding:
        """Embeds one utterance of in-memory audio.

        Raises ``AudioValidationError`` for input that cannot be embedded.
        """

    def embed_file(self, path: str | Path) -> SpeakerEmbedding:
        """Embeds one utterance from disk."""
        samples, sample_rate = preprocessing.read_raw(path)
        return self.embed(samples, sample_rate)

    def enroll(self, references: Iterable[str | Path | tuple[np.ndarray, int]]) -> SpeakerEmbedding:
        """Builds a speaker reference from one or more enrolment utterances.

        Accepts file paths or ``(samples, sample_rate)`` pairs. Utterances the
        preprocessor refuses are not silently dropped: enrolment fails, because
        a reference quietly built from half the audio the caller supplied is a
        worse outcome than an error.
        """
        items = list(references)
        if not items:
            raise EnrollmentError("enrolment needs at least one reference utterance")

        embeddings: list[SpeakerEmbedding] = []
        for index, item in enumerate(items):
            try:
                if isinstance(item, tuple):
                    samples, sample_rate = item
                    embeddings.append(self.embed(samples, sample_rate))
                else:
                    embeddings.append(self.embed_file(item))
            except preprocessing.AudioValidationError as exc:
                label = item if not isinstance(item, tuple) else f"utterance {index}"
                raise EnrollmentError(f"enrolment utterance {label} is unusable: {exc}") from exc

        total_duration = sum(e.total_duration_seconds for e in embeddings)
        warnings = [w for e in embeddings for w in e.warnings]
        if total_duration < MIN_ENROLLMENT_SECONDS:
            warnings.append(
                f"only {total_duration:.2f}s of enrolment speech (below {MIN_ENROLLMENT_SECONDS:.1f}s); "
                f"this reference is weak and its similarities should not be trusted"
            )
        return SpeakerEmbedding(
            vector=aggregate([e.vector for e in embeddings]),
            model_id=self.model_id,
            source_count=len(embeddings),
            total_duration_seconds=total_duration,
            sample_rate=self.expected_sample_rate,
            inference_seconds=sum(e.inference_seconds for e in embeddings),
            preprocessing_seconds=sum(e.preprocessing_seconds for e in embeddings),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def verify(
        self,
        reference: SpeakerEmbedding,
        probe_audio: np.ndarray,
        sample_rate: int,
        threshold: float | None = None,
    ) -> SpeakerVerificationResult:
        """Scores a probe utterance against an enrolled reference.

        Raises ``AudioValidationError`` for probe audio that cannot be judged,
        rather than returning a confident mismatch for unusable input.
        """
        if reference.model_id != self.model_id:
            raise ValueError(
                f"reference was enrolled with {reference.model_id!r} but this verifier is {self.model_id!r}; "
                f"embeddings from different models are not comparable"
            )
        if reference.dimensions != self.embedding_dimensions:
            raise ValueError(f"reference has {reference.dimensions} dimensions, this verifier produces {self.embedding_dimensions}")

        cut = self.decision_threshold if threshold is None else threshold
        probe = self.embed(probe_audio, sample_rate)
        score = cosine_similarity(reference.vector, probe.vector)
        return SpeakerVerificationResult(
            similarity_score=score,
            is_match=score >= cut,
            threshold=cut,
            model_id=self.model_id,
            reference_duration_seconds=reference.total_duration_seconds,
            probe_duration_seconds=probe.total_duration_seconds,
            reference_utterances=reference.source_count,
            inference_seconds=probe.inference_seconds,
            preprocessing_seconds=probe.preprocessing_seconds,
            warnings=tuple(dict.fromkeys([*reference.warnings, *probe.warnings])),
        )

    def verify_file(
        self,
        reference: SpeakerEmbedding,
        probe_path: str | Path,
        threshold: float | None = None,
    ) -> SpeakerVerificationResult:
        """Scores a probe utterance from disk against an enrolled reference."""
        samples, sample_rate = preprocessing.read_raw(probe_path)
        return self.verify(reference, samples, sample_rate, threshold)
