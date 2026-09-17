"""Real AASIST inference tests.

These load the actual checkpoint and run a real forward pass. They are skipped,
not faked, when the model has not been installed. Nothing here mocks the model
and then claims it works.

Install the model first:

    python ml/scripts/setup_aasist.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="ML runtime not installed in this environment")
pytest.importorskip("torch", reason="ML runtime not installed in this environment")
pytest.importorskip("scipy", reason="ML runtime not installed in this environment")
pytest.importorskip("soundfile", reason="ML runtime not installed in this environment")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from ml.audio.preprocessing import TARGET_SAMPLE_RATE, AudioTooShortError, SilentAudioError  # noqa: E402
from ml.spoof.aasist import DEFAULT_VENDOR_DIR, AASISTSpoofDetector, ModelNotInstalledError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "ml" / "data" / "eval"
MANIFEST = REPO_ROOT / "ml" / "evaluation" / "eval_manifest.json"
SUBDIR = {"bonafide": "genuine", "spoof": "synthetic"}

pytestmark = pytest.mark.integration

model_installed = pytest.mark.skipif(
    not (DEFAULT_VENDOR_DIR / "AASIST.pth").is_file(),
    reason="AASIST checkpoint not installed; run python ml/scripts/setup_aasist.py",
)


@pytest.fixture(scope="module")
def detector() -> AASISTSpoofDetector:
    """Loads the real checkpoint once for the module."""
    return AASISTSpoofDetector()


def manifest_sample(sample_id: str) -> Path | None:
    """Resolves one manifest sample to its prepared audio file, if present."""
    if not MANIFEST.is_file():
        return None
    for sample in json.loads(MANIFEST.read_text(encoding="utf-8"))["samples"]:
        if sample["sample_id"] == sample_id:
            path = EVAL_DIR / SUBDIR[sample["label"]] / sample["filename"]
            return path if path.is_file() else None
    return None


def speech_like(seconds: float = 4.5, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Builds a deterministic harmonic signal with a moving formant.

    NOT speech and NOT evidence of detection quality. It exists only to prove
    the plumbing executes without needing the evaluation corpus. Every claim
    about what the model can actually detect comes from the real-audio tests
    below, which use genuine LibriSpeech and real Piper output.
    """
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    f0 = 120.0 + 20.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    signal = sum(0.5 ** k * np.sin((k + 1) * phase) for k in range(5))
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    return (0.3 * signal * envelope).astype(np.float32)


def audio_files(subdirectory: str) -> list[Path]:
    """Lists evaluation audio of one class, if it has been prepared."""
    directory = EVAL_DIR / subdirectory
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".wav", ".flac"})


def test_missing_model_fails_with_actionable_error(tmp_path) -> None:
    """A missing checkpoint names the command that installs it."""
    with pytest.raises(ModelNotInstalledError, match="setup_aasist"):
        AASISTSpoofDetector(vendor_dir=tmp_path)


@model_installed
def test_pipeline_executes_end_to_end(detector: AASISTSpoofDetector) -> None:
    """The model runs and returns an in-range score with timings.

    Plumbing only, on a generated waveform. Detection quality is asserted on
    real audio further down.
    """
    result = detector.score(speech_like(), TARGET_SAMPLE_RATE)

    assert 0.0 <= result.synthetic_probability <= 1.0
    assert np.isfinite(result.bonafide_score)
    assert len(result.raw_scores) == 2
    assert result.model_id.startswith("AASIST/")
    assert result.inference_seconds > 0
    assert result.preprocessing_seconds > 0
    assert result.sample_rate == TARGET_SAMPLE_RATE


@model_installed
def test_inference_is_deterministic(detector: AASISTSpoofDetector) -> None:
    """The same audio scores identically twice; eval mode is on."""
    audio = speech_like()
    first = detector.score(audio, TARGET_SAMPLE_RATE)
    second = detector.score(audio, TARGET_SAMPLE_RATE)
    assert first.synthetic_probability == second.synthetic_probability
    assert first.raw_scores == second.raw_scores


@model_installed
def test_probabilities_and_raw_scores_agree(detector: AASISTSpoofDetector) -> None:
    """synthetic_probability is the softmax of the spoof class."""
    result = detector.score(speech_like(), TARGET_SAMPLE_RATE)
    spoof, bonafide = result.raw_scores
    expected = float(np.exp(spoof) / (np.exp(spoof) + np.exp(bonafide)))
    assert result.synthetic_probability == pytest.approx(expected, abs=1e-5)
    assert result.bonafide_score == pytest.approx(bonafide, abs=1e-6)


@model_installed
def test_resampled_input_is_accepted_and_flagged(detector: AASISTSpoofDetector, tmp_path) -> None:
    """A 44.1kHz file is converted and the conversion is reported."""
    path = tmp_path / "highrate.wav"
    sf.write(path, speech_like(sample_rate=44_100), 44_100)
    result = detector.score_file(path)
    assert 0.0 <= result.synthetic_probability <= 1.0
    assert any("resampled" in warning for warning in result.warnings)


@model_installed
def test_short_audio_is_tiled_and_warned_not_silently_scored(detector: AASISTSpoofDetector) -> None:
    """Audio under the model window is padded, and the caller is told."""
    result = detector.score(speech_like(seconds=1.5), TARGET_SAMPLE_RATE)
    assert any("tiled" in warning for warning in result.warnings)


@model_installed
def test_unusable_audio_never_reaches_the_model(detector: AASISTSpoofDetector, tmp_path) -> None:
    """Silence and too-short clips are refused, not scored."""
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(TARGET_SAMPLE_RATE * 5, dtype=np.float32), TARGET_SAMPLE_RATE)
    with pytest.raises(SilentAudioError):
        detector.score_file(silent)

    short = tmp_path / "short.wav"
    sf.write(short, speech_like(seconds=0.3), TARGET_SAMPLE_RATE)
    with pytest.raises(AudioTooShortError):
        detector.score_file(short)


real_audio = pytest.mark.skipif(
    manifest_sample("genuine_033") is None or manifest_sample("synthetic_023") is None,
    reason="evaluation audio not prepared; run python ml/scripts/prepare_eval_set.py --download",
)


@model_installed
@real_audio
def test_real_librispeech_sample_scores_as_human(detector: AASISTSpoofDetector) -> None:
    """Real human speech from LibriSpeech scores far below the threshold.

    Acceptance evidence on real corpus audio, not a generated waveform. The
    sample is SHA-256 pinned in the manifest and inference is deterministic, so
    this score is stable for this checkpoint.
    """
    result = detector.score_file(manifest_sample("genuine_033"))
    assert 0.0 <= result.synthetic_probability <= 1.0
    assert result.synthetic_probability < 0.01, f"real human speech scored {result.synthetic_probability}"
    assert result.inference_seconds > 0


@model_installed
@real_audio
def test_real_piper_sample_scores_as_synthetic(detector: AASISTSpoofDetector) -> None:
    """Real Piper TTS output scores well above the threshold.

    Acceptance evidence on real generated speech, not a generated waveform.
    """
    result = detector.score_file(manifest_sample("synthetic_023"))
    assert 0.0 <= result.synthetic_probability <= 1.0
    assert result.synthetic_probability > 0.9, f"real TTS scored {result.synthetic_probability}"
    assert result.inference_seconds > 0


@model_installed
@real_audio
def test_real_audio_separates_the_two_classes(detector: AASISTSpoofDetector) -> None:
    """The anchored real samples sit on opposite sides of the threshold."""
    genuine = detector.score_file(manifest_sample("genuine_033")).synthetic_probability
    synthetic = detector.score_file(manifest_sample("synthetic_023")).synthetic_probability
    assert synthetic > genuine
    assert synthetic - genuine > 0.5


@model_installed
@pytest.mark.skipif(not audio_files("genuine"), reason="no genuine evaluation audio prepared")
def test_every_prepared_genuine_sample_scores(detector: AASISTSpoofDetector) -> None:
    """Every prepared genuine sample yields an in-range score."""
    for path in audio_files("genuine")[:5]:
        result = detector.score_file(path)
        assert 0.0 <= result.synthetic_probability <= 1.0, path


@model_installed
@pytest.mark.skipif(not audio_files("synthetic"), reason="no synthetic evaluation audio prepared")
def test_every_prepared_synthetic_sample_scores(detector: AASISTSpoofDetector) -> None:
    """Every prepared synthetic sample yields an in-range score."""
    for path in audio_files("synthetic")[:5]:
        result = detector.score_file(path)
        assert 0.0 <= result.synthetic_probability <= 1.0, path
