import asyncio
import base64

import numpy as np
import pytest

from ml.runtime.contracts import SpeakerEmbedRequest, SpeakerVerifyRequest
from ml.runtime.inference import AASISTInferenceRuntime
from ml.speaker.verifier import SpeakerEmbedding, SpeakerVerificationResult, SpeakerVerifier


class FakeSpeakerVerifier(SpeakerVerifier):
    @property
    def model_id(self):
        """Identifies the deterministic test model."""
        return "fake-ecapa@revision"

    @property
    def expected_sample_rate(self):
        """Returns the canonical sample rate."""
        return 16_000

    @property
    def embedding_dimensions(self):
        """Returns the test vector dimensions."""
        return 3

    @property
    def decision_threshold(self):
        """Returns the exploratory test threshold."""
        return 0.55

    def embed(self, audio, sample_rate):
        """Returns a normalized deterministic embedding."""
        return SpeakerEmbedding(np.array([1.0, 0.0, 0.0], dtype=np.float32), self.model_id, 1, len(audio) / sample_rate, sample_rate)

    def verify(self, reference, probe_audio, sample_rate, threshold=None):
        """Returns a deterministic uncalibrated similarity."""
        return SpeakerVerificationResult(0.8, True, threshold or self.decision_threshold, self.model_id, 3.0, len(probe_audio) / sample_rate, 1, 0.01, 0.01)


def encoded(samples):
    """Encodes canonical samples for the transport contract."""
    return base64.b64encode(np.asarray(samples, dtype="<f4").tobytes()).decode("ascii")


def test_speaker_runtime_returns_embedding_and_uncalibrated_similarity():
    """The runtime exposes real contract semantics without probability claims."""
    async def exercise():
        """Runs the asynchronous contract assertions."""
        runtime = AASISTInferenceRuntime(speaker_verifier=FakeSpeakerVerifier())
        audio = np.zeros(32_000, dtype=np.float32)
        embedded = await runtime.embed_speaker(SpeakerEmbedRequest(samples_base64=encoded(audio)))
        assert embedded.embedding_dimensions == 3
        assert embedded.score_semantics == "uncalibrated_embedding"
        verified = await runtime.verify_speaker(SpeakerVerifyRequest(samples_base64=encoded(audio), reference_embedding=embedded.embedding, threshold=0.55))
        assert verified.cosine_similarity == 0.8
        assert verified.score_semantics == "uncalibrated_similarity"
    asyncio.run(exercise())


def test_speaker_runtime_rejects_short_probe():
    """The runtime refuses insufficient speaker evidence."""
    async def exercise():
        """Runs the short-audio assertion."""
        runtime = AASISTInferenceRuntime(speaker_verifier=FakeSpeakerVerifier())
        with pytest.raises(ValueError, match="enough"):
            await runtime.embed_speaker(SpeakerEmbedRequest(samples_base64=encoded(np.zeros(1, dtype=np.float32))))
    asyncio.run(exercise())
