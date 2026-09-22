import numpy as np

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
