"""Audio preprocessing tests.

These cover the trust boundary in front of the model: what gets converted, what
gets flagged, and what gets refused outright.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="ML runtime not installed in this environment")
pytest.importorskip("scipy", reason="ML runtime not installed in this environment")
pytest.importorskip("soundfile", reason="ML runtime not installed in this environment")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from ml.audio.preprocessing import (  # noqa: E402

    TARGET_SAMPLE_RATE,
    AudioTooShortError,
    MalformedAudioError,
    SilentAudioError,
    load,
    prepare,
    resample,
    to_mono,
)


def tone(seconds: float, sample_rate: int = TARGET_SAMPLE_RATE, frequency: float = 220.0, amplitude: float = 0.4) -> np.ndarray:
    """Builds a mono sine wave, a stand-in for real speech."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_mono_audio_passes_through_unchanged() -> None:
    """Mono input is not touched by the downmix step."""
    samples = tone(2.0)
    mono, warnings = to_mono(samples)
    assert warnings == []
    assert np.array_equal(mono, samples)


def test_stereo_is_downmixed_and_reported() -> None:
    """Two channels are averaged and the change is recorded."""
    left, right = tone(2.0), tone(2.0, frequency=440.0)
    mono, warnings = to_mono(np.stack([left, right], axis=1))
    assert mono.shape == left.shape
    assert np.allclose(mono, (left + right) / 2, atol=1e-6)
    assert warnings == ["downmixed 2 channels to mono"]


def test_single_channel_2d_is_flattened_without_warning() -> None:
    """A (n, 1) array is already mono."""
    samples = tone(2.0)
    mono, warnings = to_mono(samples.reshape(-1, 1))
    assert warnings == []
    assert np.array_equal(mono, samples)


def test_three_dimensional_audio_is_rejected() -> None:
    """Unexpected array shapes fail loudly."""
    with pytest.raises(MalformedAudioError):
        to_mono(np.zeros((2, 2, 2), dtype=np.float32))


def test_resample_is_a_noop_at_the_target_rate() -> None:
    """No resampling happens when the rate already matches."""
    samples = tone(2.0)
    resampled, warnings = resample(samples, TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    assert warnings == []
    assert np.array_equal(resampled, samples)


@pytest.mark.parametrize("source_rate", [8_000, 22_050, 44_100, 48_000])
def test_resample_converts_to_the_target_length(source_rate: int) -> None:
    """Resampling produces roughly duration * target_rate samples."""
    samples = tone(2.0, sample_rate=source_rate)
    resampled, warnings = resample(samples, source_rate, TARGET_SAMPLE_RATE)
    assert warnings == [f"resampled {source_rate}Hz to {TARGET_SAMPLE_RATE}Hz"]
    assert abs(len(resampled) - 2.0 * TARGET_SAMPLE_RATE) < 0.02 * TARGET_SAMPLE_RATE


def test_resample_rejects_a_non_positive_rate() -> None:
    """A zero or negative sample rate is malformed, not a default."""
    with pytest.raises(MalformedAudioError):
        resample(tone(2.0), 0, TARGET_SAMPLE_RATE)


def test_prepare_returns_target_rate_mono_float32() -> None:
    """A clean 2s clip comes back model-ready with no warnings."""
    prepared = prepare(tone(2.0), TARGET_SAMPLE_RATE)
    assert prepared.sample_rate == TARGET_SAMPLE_RATE
    assert prepared.samples.dtype == np.float32
    assert prepared.samples.ndim == 1
    assert prepared.duration_seconds == pytest.approx(2.0, abs=0.01)
    assert prepared.warnings == ()


def test_prepare_records_every_conversion_it_made() -> None:
    """Downmix and resample are both surfaced to the caller."""
    stereo = np.stack([tone(2.0, sample_rate=44_100), tone(2.0, sample_rate=44_100)], axis=1)
    prepared = prepare(stereo, 44_100)
    assert "downmixed 2 channels to mono" in prepared.warnings
    assert f"resampled 44100Hz to {TARGET_SAMPLE_RATE}Hz" in prepared.warnings


def test_silence_is_refused_rather_than_scored() -> None:
    """Digital silence cannot produce a meaningful verdict."""
    with pytest.raises(SilentAudioError):
        prepare(np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.float32), TARGET_SAMPLE_RATE)


def test_near_silence_is_refused() -> None:
    """Amplitude below the silence floor is still silence."""
    with pytest.raises(SilentAudioError):
        prepare(tone(2.0, amplitude=1e-6), TARGET_SAMPLE_RATE)


def test_audio_shorter_than_the_minimum_is_refused() -> None:
    """Very short clips would score the padding, not the speaker."""
    with pytest.raises(AudioTooShortError):
        prepare(tone(0.4), TARGET_SAMPLE_RATE)


def test_short_audio_is_measured_after_resampling() -> None:
    """Duration is judged at the target rate, not the source rate."""
    with pytest.raises(AudioTooShortError):
        prepare(tone(0.5, sample_rate=48_000), 48_000)


def test_low_level_audio_is_scored_but_flagged() -> None:
    """Quiet-but-present audio is a warning, not a refusal."""
    prepared = prepare(tone(2.0, amplitude=5e-4), TARGET_SAMPLE_RATE)
    assert any("low signal level" in warning for warning in prepared.warnings)


def test_clipped_audio_is_flagged() -> None:
    """Samples beyond unit scale suggest clipping."""
    prepared = prepare(tone(2.0, amplitude=2.5), TARGET_SAMPLE_RATE)
    assert any("clipped" in warning for warning in prepared.warnings)


def test_empty_audio_is_rejected() -> None:
    """No samples means nothing to score."""
    with pytest.raises(MalformedAudioError):
        prepare(np.array([], dtype=np.float32), TARGET_SAMPLE_RATE)


def test_non_finite_samples_are_rejected() -> None:
    """NaN or inf would propagate silently into the model."""
    samples = tone(2.0).copy()
    samples[100] = np.nan
    with pytest.raises(MalformedAudioError):
        prepare(samples, TARGET_SAMPLE_RATE)


def test_integer_samples_are_rejected() -> None:
    """Integer PCM must be converted to float before scoring."""
    with pytest.raises(MalformedAudioError):
        prepare(np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.int16), TARGET_SAMPLE_RATE)


def test_load_reads_a_real_wav_file(tmp_path) -> None:
    """A written WAV round-trips through the loader."""
    path = tmp_path / "sample.wav"
    sf.write(path, tone(2.0), TARGET_SAMPLE_RATE)
    prepared = load(path)
    assert prepared.sample_rate == TARGET_SAMPLE_RATE
    assert prepared.duration_seconds == pytest.approx(2.0, abs=0.01)


def test_load_resamples_a_non_target_rate_file(tmp_path) -> None:
    """A 44.1kHz file is converted on the way in."""
    path = tmp_path / "sample.wav"
    sf.write(path, tone(2.0, sample_rate=44_100), 44_100)
    prepared = load(path)
    assert prepared.sample_rate == TARGET_SAMPLE_RATE
    assert any("resampled" in warning for warning in prepared.warnings)


def test_load_rejects_a_file_that_is_not_audio(tmp_path) -> None:
    """Garbage bytes fail as malformed rather than crashing."""
    path = tmp_path / "not-audio.wav"
    path.write_bytes(b"this is definitely not a RIFF header")
    with pytest.raises(MalformedAudioError):
        load(path)


def test_load_rejects_a_truncated_wav_header(tmp_path) -> None:
    """A partial header is malformed input."""
    path = tmp_path / "truncated.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    with pytest.raises(MalformedAudioError):
        load(path)