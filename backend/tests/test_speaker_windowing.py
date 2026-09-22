import numpy as np

from app.services.audio_source_ledger import AudioSourceSegment
from app.services.speaker_windowing import SPEAKER_WINDOW_HOP_SAMPLES, SPEAKER_WINDOW_SAMPLES, SpeakerProbeScheduler


def test_speaker_scheduler_is_continuous_across_arbitrary_chunks():
    """The 32k/16k probe schedule is independent of transport boundaries."""
    scheduler = SpeakerProbeScheduler()
    source = np.arange(SPEAKER_WINDOW_SAMPLES + SPEAKER_WINDOW_HOP_SAMPLES, dtype=np.float32)
    windows = []
    for chunk in (source[:777], source[777:12_345], source[12_345:]):
        windows.extend(scheduler.push(chunk))
    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0], source[:SPEAKER_WINDOW_SAMPLES])
    np.testing.assert_array_equal(windows[1], source[SPEAKER_WINDOW_HOP_SAMPLES:])


def test_speaker_scheduler_never_zero_pads_short_audio():
    """Insufficient audio produces no fabricated speaker evidence."""
    scheduler = SpeakerProbeScheduler()
    assert scheduler.push(np.zeros(SPEAKER_WINDOW_SAMPLES - 1, dtype=np.float32)) == []


def test_speaker_probe_carries_canonical_and_source_correlation():
    """Speaker probes retain the source interval that produced them."""
    scheduler = SpeakerProbeScheduler()
    source = AudioSourceSegment(4_000, 35_999, 7, 7, 0)
    probes = scheduler.push(np.zeros(SPEAKER_WINDOW_SAMPLES, dtype=np.float32), source)
    assert len(probes) == 1
    metadata = probes[0].metadata
    assert metadata.canonical_start_sample == 0
    assert metadata.canonical_end_sample == SPEAKER_WINDOW_SAMPLES
    assert metadata.source_frame_start == 4_000
    assert metadata.source_frame_end == 35_999
