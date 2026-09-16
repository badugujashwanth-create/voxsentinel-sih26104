"""Canonical PCM fingerprint tests.

The canonical check has to do two things at once: accept the floating-point
variation that real hardware produces, and reject audio that is actually wrong.
These tests pin both sides of that line.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="ML runtime not installed in this environment")
pytest.importorskip("scipy", reason="ML runtime not installed in this environment")

import numpy as np  # noqa: E402

from ml.audio.fingerprint import (  # noqa: E402
    CANONICAL_FLOOR_DB,
    CANONICAL_FRAME_SAMPLES,
    CANONICAL_TOLERANCE_DB,
    envelope_db,
    envelope_distance,
    envelopes_match,
    fingerprint,
)

SAMPLE_RATE = 22_050
SECONDS = 5.0


def speech_like(seconds: float = SECONDS, seed: int = 0) -> np.ndarray:
    """Builds a deterministic int16 waveform with varying loudness.

    Amplitude is modulated so the envelope has structure to compare, rather
    than a single flat value that would match trivially.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * 180.0 * t) + 0.4 * np.sin(2 * np.pi * 430.0 * t)
    envelope = 0.25 + 0.7 * np.abs(np.sin(2 * np.pi * 0.7 * t))
    noise = 0.02 * rng.standard_normal(n)
    return np.clip((carrier * envelope + noise) * 12000, -32767, 32767).astype(np.int16)


def dither(pcm: np.ndarray, lsb: int, seed: int = 1) -> np.ndarray:
    """Adds bounded integer noise, standing in for float-kernel differences."""
    rng = np.random.default_rng(seed)
    return np.clip(pcm.astype(int) + rng.integers(-lsb, lsb + 1, len(pcm)), -32767, 32767).astype(np.int16)


# --- the canonical form -------------------------------------------------------


def test_envelope_has_one_value_per_frame() -> None:
    """A 5s clip at 22.05kHz yields the expected frame count."""
    pcm = speech_like()
    assert len(envelope_db(pcm)) == len(pcm) // CANONICAL_FRAME_SAMPLES


def test_envelope_is_deterministic() -> None:
    """The same PCM always produces the same envelope."""
    pcm = speech_like()
    assert np.array_equal(envelope_db(pcm), envelope_db(pcm))


def test_silence_maps_to_the_floor() -> None:
    """Digital silence lands on the floor rather than negative infinity."""
    envelope = envelope_db(np.zeros(SAMPLE_RATE * 2, dtype=np.int16))
    assert np.all(envelope == CANONICAL_FLOOR_DB)


def test_audio_shorter_than_one_frame_is_still_measured() -> None:
    """Short input produces a single frame instead of an empty envelope."""
    assert len(envelope_db(speech_like(seconds=0.05))) == 1


def test_empty_audio_is_rejected() -> None:
    """There is nothing to fingerprint in an empty array."""
    with pytest.raises(ValueError):
        envelope_db(np.array([], dtype=np.int16))


def test_multichannel_audio_is_rejected() -> None:
    """The fingerprint is defined on mono PCM."""
    with pytest.raises(ValueError):
        envelope_db(np.zeros((1000, 2), dtype=np.int16))


def test_fingerprint_is_plain_rounded_floats() -> None:
    """The committed form is JSON-friendly."""
    values = fingerprint(speech_like())
    assert all(isinstance(v, float) for v in values)
    assert len(values) == len(envelope_db(speech_like()))


# --- tolerance: small variation must PASS -------------------------------------


@pytest.mark.parametrize("lsb", [1, 2, 4, 8, 17])
def test_small_pcm_variation_passes(lsb: int) -> None:
    """Bounded LSB noise is accepted.

    17 LSB is the largest per-sample difference measured between ONNX graph
    optimisation modes on this project's real Piper samples.
    """
    reference = speech_like()
    matched, distance = envelopes_match(envelope_db(dither(reference, lsb)), envelope_db(reference))
    assert matched, f"+/-{lsb} LSB rejected at {distance:.6f} dB"
    assert distance < CANONICAL_TOLERANCE_DB


def test_identical_audio_has_zero_distance() -> None:
    """The same audio compares exactly equal."""
    pcm = speech_like()
    matched, distance = envelopes_match(envelope_db(pcm), envelope_db(pcm))
    assert matched
    assert distance == 0.0


# --- meaningful corruption must FAIL ------------------------------------------


@pytest.mark.parametrize("gain", [1.005, 1.01, 1.05, 0.9])
def test_amplitude_change_fails(gain: float) -> None:
    """A level change of half a percent or more is caught."""
    reference = speech_like()
    changed = np.clip(reference.astype(np.float64) * gain, -32767, 32767).astype(np.int16)
    matched, distance = envelopes_match(envelope_db(changed), envelope_db(reference))
    assert not matched, f"gain {gain} slipped through at {distance:.6f} dB"


def test_zeroed_frame_fails() -> None:
    """A dropout of one frame is caught."""
    reference = speech_like()
    broken = reference.copy()
    broken[CANONICAL_FRAME_SAMPLES * 5 : CANONICAL_FRAME_SAMPLES * 6] = 0
    matched, _ = envelopes_match(envelope_db(broken), envelope_db(reference))
    assert not matched


def test_different_audio_content_fails() -> None:
    """Different speech of the same length does not match."""
    a, b = speech_like(seed=0), speech_like(seed=99) * 2
    matched, _ = envelopes_match(envelope_db(b), envelope_db(a))
    assert not matched


def test_time_reversed_audio_fails() -> None:
    """Same samples in a different order is different audio."""
    reference = speech_like()
    matched, _ = envelopes_match(envelope_db(reference[::-1]), envelope_db(reference))
    assert not matched


def test_loud_noise_fails() -> None:
    """Audible added noise is caught even though LSB dither is not."""
    reference = speech_like()
    matched, _ = envelopes_match(envelope_db(dither(reference, 4000)), envelope_db(reference))
    assert not matched


def test_length_mismatch_is_infinite_distance() -> None:
    """A different frame count can never be mistaken for a close match."""
    a = envelope_db(speech_like(seconds=5.0))
    b = envelope_db(speech_like(seconds=4.0))
    assert envelope_distance(a, b) == float("inf")
    assert not envelopes_match(a, b)[0]


def test_tolerance_sits_between_the_noise_floor_and_real_corruption() -> None:
    """The chosen tolerance separates the two populations with margin.

    Guards the constant itself: if someone widens it far enough to accept a
    0.5% amplitude change, this fails.
    """
    reference = speech_like()
    noise = envelope_distance(envelope_db(dither(reference, 17)), envelope_db(reference))
    corruption = envelope_distance(
        envelope_db(np.clip(reference.astype(np.float64) * 1.005, -32767, 32767).astype(np.int16)),
        envelope_db(reference),
    )
    assert noise < CANONICAL_TOLERANCE_DB < corruption
