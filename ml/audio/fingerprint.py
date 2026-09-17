"""Canonical PCM fingerprinting for reproducibility checks.

Why this exists
---------------
Piper synthesis is float32 inference in ONNX Runtime. Its exact output depends
on the execution plan, which varies with ONNX Runtime version, build flags, and
CPU kernel dispatch. Piper then divides the whole waveform by its peak sample
(``normalize_audio``, upstream default), so any single-ULP difference is coupled
into every sample before int16 quantisation. A byte-identical WAV is therefore
NOT a sound cross-machine expectation, even though the audio is perceptually
identical.

Measured on this project's 40 Piper samples across all four ONNX
graph-optimisation modes: 143,901 int16 samples differ, by up to 17 LSB
(about -66 dBFS, inaudible).

Why not per-sample quantisation
-------------------------------
Quantising samples and hashing does not fix this at any tolerance. A sample
sitting near a bucket boundary flips buckets however coarse the buckets are,
and one flipped sample changes the hash. Measured: dropping the low k bits left
40/40 samples unstable for k <= 9, and still 37/40 unstable at k = 10, which is
a +/-512 LSB bucket (1.6% of full scale). Coarser quantisation is not
monotonically more stable either, which is why hash-equality on quantised values
is the wrong tool here.

What this does instead
----------------------
Reduce the waveform to a coarse energy envelope -- frame RMS in dBFS -- and
compare it numerically with a tolerance. Averaging over a frame suppresses
per-sample noise, and a tolerance comparison has no bucket boundary to straddle.

Choosing the tolerance (measured, not guessed)
----------------------------------------------
    execution-plan noise floor, worst over 40 samples x 6 mode pairs: 0.000120 dB
    0.5% amplitude change:                                            0.0430 dB
    1% amplitude change:                                              0.0861 dB
    one 186 ms frame zeroed:                                         77.2417 dB
    different speaker, same text:                                    10.1153 dB

``CANONICAL_TOLERANCE_DB`` sits at 0.01 dB: about 83x above the observed noise
floor and about 3.6x below the smallest corruption tested. It accepts the float
variation real hardware produces and rejects audio that is actually wrong.

The envelope is a ~186 ms-resolution energy summary. It is a verification
fingerprint, not audio: speech cannot be reconstructed from it, which is why it
is safe to commit when the audio itself is not.
"""

from __future__ import annotations

import numpy as np

#: Frame length in samples. At 22.05 kHz this is ~186 ms.
CANONICAL_FRAME_SAMPLES = 4096

#: Frames quieter than this all map to the floor, so near-silence does not
#: produce large dB swings from tiny absolute differences.
CANONICAL_FLOOR_DB = -90.0

#: Maximum per-frame dB difference still considered the same audio.
CANONICAL_TOLERANCE_DB = 0.01

#: Full-scale reference for 16-bit PCM.
_FULL_SCALE = 32767.0


def envelope_db(
    pcm: np.ndarray,
    frame_samples: int = CANONICAL_FRAME_SAMPLES,
    floor_db: float = CANONICAL_FLOOR_DB,
) -> np.ndarray:
    """Reduces int16 PCM to a per-frame RMS envelope in dBFS.

    Trailing samples that do not fill a whole frame are dropped, so the result
    is only comparable between signals of the same length. Callers must check
    the sample count separately; this function does not.
    """
    if frame_samples <= 0:
        raise ValueError(f"frame_samples must be positive, got {frame_samples}")

    samples = np.asarray(pcm)
    if samples.ndim != 1:
        raise ValueError(f"expected mono PCM, got {samples.ndim} dimensions")
    if samples.size == 0:
        raise ValueError("cannot fingerprint empty audio")

    # A signal shorter than one frame is measured whole rather than dropped.
    count = max(1, samples.size // frame_samples)
    usable = min(samples.size, count * frame_samples)
    width = usable // count
    block = samples[:usable].reshape(count, width).astype(np.float64)

    rms = np.sqrt(np.mean(block * block, axis=1))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(rms, 1e-12) / _FULL_SCALE)
    return np.maximum(db, floor_db)


def envelope_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Returns the largest per-frame dB difference between two envelopes.

    Returns infinity when the envelopes have different lengths, so a length
    mismatch can never be mistaken for a close match.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return float("inf")
    return float(np.max(np.abs(a - b)))


def envelopes_match(
    left: np.ndarray,
    right: np.ndarray,
    tolerance_db: float = CANONICAL_TOLERANCE_DB,
) -> tuple[bool, float]:
    """Reports whether two envelopes agree within ``tolerance_db``."""
    distance = envelope_distance(left, right)
    return distance <= tolerance_db, distance


def fingerprint(pcm: np.ndarray, decimals: int = 6) -> list[float]:
    """Returns the committed form of an envelope: rounded plain floats."""
    return [round(float(value), decimals) for value in envelope_db(pcm)]
