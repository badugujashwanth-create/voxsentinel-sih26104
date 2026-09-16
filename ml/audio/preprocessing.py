"""Audio loading, validation, and normalisation for spoof detection.

Anti-spoof models are trained on a specific sample rate and channel layout, and
they will happily return a confident number for audio that is unusable. This
module is the trust boundary: it converts what we were given into what the model
expects, and refuses input that cannot produce a meaningful answer.

No audio is written to disk or logged here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16_000

#: Shorter than this and the model is scoring mostly repetition padding, so the
#: answer says more about the tiling than about the speaker.
MIN_DURATION_SECONDS = 1.0

#: Peak amplitude at or below this counts as digital silence.
SILENCE_PEAK_THRESHOLD = 1e-4

#: RMS below this is audible-but-empty: scored, but flagged.
LOW_LEVEL_RMS_THRESHOLD = 1e-3


class AudioValidationError(ValueError):
    """Raised when audio cannot produce a meaningful spoof score."""


class MalformedAudioError(AudioValidationError):
    """Raised when a file cannot be decoded or holds non-finite samples."""


class AudioTooShortError(AudioValidationError):
    """Raised when audio is too short to score honestly."""


class SilentAudioError(AudioValidationError):
    """Raised when audio carries no signal."""


@dataclass(frozen=True)
class PreparedAudio:
    """Mono float32 audio at the target rate, plus what we had to do to it."""

    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    warnings: tuple[str, ...]


def to_mono(samples: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Averages a multi-channel array down to one channel."""
    if samples.ndim == 1:
        return samples, []
    if samples.ndim != 2:
        raise MalformedAudioError(f"expected 1-D or 2-D audio, got {samples.ndim} dimensions")
    channels = samples.shape[1]
    if channels == 1:
        return samples[:, 0], []
    return samples.mean(axis=1), [f"downmixed {channels} channels to mono"]


def resample(samples: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, list[str]]:
    """Resamples mono audio with a polyphase filter."""
    if source_rate <= 0:
        raise MalformedAudioError(f"sample rate must be positive, got {source_rate}")
    if source_rate == target_rate:
        return samples, []
    common = np.gcd(source_rate, target_rate)
    resampled = resample_poly(samples, target_rate // common, source_rate // common)
    return resampled.astype(np.float32, copy=False), [f"resampled {source_rate}Hz to {target_rate}Hz"]


def prepare(samples: np.ndarray, sample_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> PreparedAudio:
    """Validates and converts raw samples into model-ready mono audio.

    Raises ``AudioValidationError`` rather than returning a confident score for
    input the model cannot meaningfully judge.
    """
    array = np.asarray(samples)
    if array.size == 0:
        raise MalformedAudioError("audio contains no samples")
    if not np.issubdtype(array.dtype, np.floating):
        raise MalformedAudioError(f"expected floating-point samples, got dtype {array.dtype}")

    warnings: list[str] = []
    mono, mono_warnings = to_mono(array)
    warnings.extend(mono_warnings)

    mono = mono.astype(np.float32, copy=False)
    if not np.all(np.isfinite(mono)):
        raise MalformedAudioError("audio contains NaN or infinite samples")

    # Judge clipping on the source signal. Polyphase resampling overshoots the
    # original peak slightly (Gibbs ringing), so checking after resampling
    # reports clipping on perfectly clean audio.
    source_peak = float(np.max(np.abs(mono)))

    resampled, resample_warnings = resample(mono, sample_rate, target_rate)
    warnings.extend(resample_warnings)

    duration = len(resampled) / target_rate
    peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
    if peak <= SILENCE_PEAK_THRESHOLD:
        raise SilentAudioError(f"audio is silent (peak amplitude {peak:.2e})")
    if duration < MIN_DURATION_SECONDS:
        raise AudioTooShortError(f"audio is {duration:.2f}s, need at least {MIN_DURATION_SECONDS:.1f}s")

    rms = float(np.sqrt(np.mean(np.square(resampled))))
    if rms < LOW_LEVEL_RMS_THRESHOLD:
        warnings.append(f"very low signal level (RMS {rms:.2e}); score may be unreliable")
    if source_peak > 1.0:
        warnings.append(f"source samples exceed unit scale (peak {source_peak:.2f}); audio may be clipped")

    return PreparedAudio(samples=resampled, sample_rate=target_rate, duration_seconds=duration, warnings=tuple(warnings))


def read_raw(path: str | Path) -> tuple[np.ndarray, int]:
    """Decodes an audio file without validating or converting it.

    Preprocessing happens once, inside the detector's ``score``, so that every
    conversion it performs is reported in a single set of warnings.
    """
    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise MalformedAudioError(f"could not decode {path}: {exc}") from exc
    return samples, sample_rate


def load(path: str | Path, target_rate: int = TARGET_SAMPLE_RATE) -> PreparedAudio:
    """Reads an audio file and prepares it for the model in one step."""
    samples, sample_rate = read_raw(path)
    return prepare(samples, sample_rate, target_rate)
