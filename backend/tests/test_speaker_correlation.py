import asyncio

import numpy as np

from app.services.async_session import AsyncCallSession, AudioWindow
from app.services.audio_source_ledger import AudioWindowMetadata
from app.services.speaker_windowing import SpeakerProbe, SpeakerProbeMetadata


def _metadata(sequence: int, start: int, end: int) -> SpeakerProbeMetadata:
    """Creates deterministic probe metadata for correlation tests."""
    return SpeakerProbeMetadata(sequence, start, end, 10 + start, 10 + end, sequence, sequence, 0)


def _aasist_metadata(end: int) -> AudioWindowMetadata:
    """Creates deterministic AASIST metadata for correlation tests."""
    return AudioWindowMetadata(0, end, 1, 2, 0, 1, 0, end)


def test_aligned_probe_is_selected_for_aasist_window():
    """The newest causally aligned speaker probe is selected."""
    session = AsyncCallSession("aligned")
    session.enqueue_speaker_window(SpeakerProbe(np.zeros(32_000, dtype=np.float32), _metadata(1, 0, 32_000)))
    session.enqueue_speaker_window(SpeakerProbe(np.zeros(32_000, dtype=np.float32), _metadata(2, 16_000, 48_000)))
    selected = session.speaker_probe_for_window(_aasist_metadata(64_600))
    assert selected is not None
    assert selected.metadata.speaker_window_sequence == 2


def test_stale_probe_is_not_selected():
    """A probe outside the bounded causal age is unavailable to fusion."""
    session = AsyncCallSession("stale")
    session.enqueue_speaker_window(SpeakerProbe(np.zeros(32_000, dtype=np.float32), _metadata(1, 0, 32_000)))
    assert session.speaker_probe_for_window(_aasist_metadata(100_000)) is None


def test_out_of_order_probe_cannot_replace_newer_probe():
    """Older probe sequences are rejected after a newer sequence is accepted."""
    session = AsyncCallSession("order")
    newest = SpeakerProbe(np.zeros(32_000, dtype=np.float32), _metadata(2, 16_000, 48_000))
    older = SpeakerProbe(np.zeros(32_000, dtype=np.float32), _metadata(1, 0, 32_000))
    session.enqueue_speaker_window(newest)
    session.enqueue_speaker_window(older)
    selected = session.speaker_probe_for_window(_aasist_metadata(64_600))
    assert selected is newest


def test_probe_history_is_bounded():
    """Correlation history remains bounded under continuous probe production."""
    session = AsyncCallSession("bounded")
    for sequence in range(1, 20):
        start = sequence * 16_000
        session.enqueue_speaker_window(SpeakerProbe(np.zeros(32_000, dtype=np.float32), _metadata(sequence, start, start + 32_000)))
    assert len(session._speaker_probe_history) == 8


def test_new_session_has_no_previous_probe_correlation():
    """A new call cannot observe a previous call's speaker evidence."""
    previous = AsyncCallSession("previous")
    previous.enqueue_speaker_window(SpeakerProbe(np.zeros(32_000, dtype=np.float32), _metadata(1, 0, 32_000)))
    current = AsyncCallSession("current")
    assert current.speaker_probe_for_window(_aasist_metadata(64_600)) is None
class CorrelationClient:
    async def health(self):
        """Reports test-service readiness."""
        return None

    """Provides deterministic spoof and speaker results for correlation tests."""

    async def infer(self, window):
        """Returns bounded spoof evidence."""
        from app.models.ml_evidence import RawSpoofEvidence
        return RawSpoofEvidence(model_id="aasist-test", raw_spoof_score=0.1, score_semantics="uncalibrated", threshold=0.5, prediction="bonafide", audio_duration_ms=4000, inference_ms=1, preprocessing_ms=1)

    async def verify_speaker(self, samples, reference_embedding, threshold):
        """Returns one deterministic speaker similarity."""
        return {"model_id": "ecapa-test", "model_revision": "revision", "cosine_similarity": 0.8}


async def _correlated_event():
    """Runs one provider event with aligned speaker and AASIST metadata."""
    from app.services.async_session import AudioSessionRegistry
    from app.services.risk_provider import MLRiskProvider
    from app.services.speaker_profiles import SpeakerProfileRegistry
    profiles = SpeakerProfileRegistry()
    profile = profiles.add("expected", {"embedding": [1.0, 0.0], "model_id": "ecapa-test", "model_revision": "revision", "embedding_dimensions": 2}, "test")
    provider = MLRiskProvider(AudioSessionRegistry(), CorrelationClient(), profile_registry=profiles)
    session = await provider.prepare_session("correlated", profile.profile_id)
    session.enqueue_speaker_window(SpeakerProbe(np.zeros(32_000, dtype=np.float32), _metadata(3, 16_000, 48_000)))
    session.enqueue_window(AudioWindow(1, np.zeros(64_600, dtype=np.float32), _aasist_metadata(64_600)))
    return await anext(provider.stream(session, asyncio.Event()))


def test_speaker_result_preserves_probe_correlation_metadata():
    """ECAPA result remains tied to the probe interval it actually evaluated."""
    event = asyncio.run(_correlated_event())
    assert event.speaker_window_sequence == 3
    assert event.speaker_canonical_start_sample == 16_000
    assert event.speaker_canonical_end_sample == 48_000
    assert event.speaker_source_frame_start == 16_010
    assert event.speaker_source_frame_end == 48_010