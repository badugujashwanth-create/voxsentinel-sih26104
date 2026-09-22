import asyncio

import numpy as np

from app.models.ml_evidence import RawSpoofEvidence
from app.services.async_session import AudioSessionRegistry, AudioWindow
from app.services.risk_provider import MLRiskProvider
from app.services.speaker_profiles import SpeakerProfileRegistry


class ClientDouble:
    async def health(self):
        """Accepts the deterministic test runtime."""

    async def infer(self, window):
        """Returns real-contract-shaped spoof evidence."""
        return RawSpoofEvidence(model_id="aasist-test", raw_spoof_score=0.1, score_semantics="uncalibrated", threshold=0.5, prediction="bonafide", audio_duration_ms=4000, inference_ms=3, preprocessing_ms=1)

    async def verify_speaker(self, samples, reference_embedding, threshold):
        """Returns deterministic uncalibrated speaker similarity."""
        return {"model_id": "ecapa-test", "cosine_similarity": 0.8}


def test_profile_selected_call_emits_speaker_and_fusion_evidence():
    """One existing AASIST window produces explicit ECAPA/fusion fields."""
    async def exercise():
        """Runs the single-window provider flow."""
        registry = AudioSessionRegistry()
        profiles = SpeakerProfileRegistry()
        profile = profiles.add("speaker-1", {"embedding": [1.0, 0.0], "model_id": "ecapa-test", "model_revision": "revision", "embedding_dimensions": 2}, "test fixture")
        provider = MLRiskProvider(registry, ClientDouble(), profile_registry=profiles)
        session = await provider.prepare_session("call-1", profile.profile_id)
        session.enqueue_speaker_window(np.zeros(32_000, dtype=np.float32))
        session.enqueue_window(AudioWindow(1, np.zeros(64_600, dtype=np.float32)))
        event = await anext(provider.stream(session, asyncio.Event()))
        assert event.speaker_state == "CONSISTENT"
        assert event.speaker_score_semantics == "uncalibrated_similarity"
        assert event.fusion_state == "NORMAL"
        await session.close()
    asyncio.run(exercise())
