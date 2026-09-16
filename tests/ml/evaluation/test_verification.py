"""Evaluation-set verification tests.

Covers the rules that gate a reconstruction: strict bytes for corpus copies,
tolerant canonical PCM for regenerated speech, and hard failure when the
generator inputs are not the ones the manifest names.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="ML runtime not installed in this environment")
pytest.importorskip("scipy", reason="ML runtime not installed in this environment")
pytest.importorskip("soundfile", reason="ML runtime not installed in this environment")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import sys  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ml.audio.fingerprint import CANONICAL_TOLERANCE_DB, fingerprint  # noqa: E402
from ml.scripts.prepare_eval_set import verify_genuine, verify_synthetic, verify_voice  # noqa: E402

MANIFEST = REPO_ROOT / "ml" / "evaluation" / "eval_manifest.json"
SAMPLE_RATE = 22_050


def speech_like(seconds: float = 5.0, seed: int = 0, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Builds a deterministic int16 waveform with varying loudness."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    carrier = np.sin(2 * np.pi * 180.0 * t) + 0.4 * np.sin(2 * np.pi * 430.0 * t)
    envelope = 0.25 + 0.7 * np.abs(np.sin(2 * np.pi * 0.7 * t))
    return np.clip((carrier * envelope + 0.02 * rng.standard_normal(n)) * 12000, -32767, 32767).astype(np.int16)


def write(path: Path, pcm: np.ndarray, rate: int = SAMPLE_RATE) -> Path:
    """Writes int16 PCM to a WAV file."""
    sf.write(path, pcm, rate, subtype="PCM_16")
    return path


def synthetic_entry(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> dict:
    """Builds a manifest entry describing this audio."""
    return {
        "sample_id": "synthetic_test",
        "label": "spoof",
        "filename": "sample.wav",
        "sample_rate": rate,
        "channels": 1,
        "sample_count": int(len(pcm)),
        "sha256": "0" * 64,  # informational only
        "pcm_envelope_db": fingerprint(pcm),
        "voice_sha256": "a" * 64,
        "voice_config_sha256": "b" * 64,
    }


# --- strict path: LibriSpeech copies -----------------------------------------


def test_genuine_matching_bytes_pass(tmp_path) -> None:
    """A byte-identical corpus copy verifies."""
    path = write(tmp_path / "g.wav", speech_like(seconds=1.0))
    entry = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert verify_genuine(entry, path) is None


def test_genuine_single_changed_byte_fails(tmp_path) -> None:
    """Genuine samples stay on a strict gate: one byte breaks it."""
    path = write(tmp_path / "g.wav", speech_like(seconds=1.0))
    entry = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.write_bytes(path.read_bytes() + b"\x00")
    assert "SHA-256 mismatch" in (verify_genuine(entry, path) or "")


# --- tolerant path: regenerated Piper audio ----------------------------------


def test_identical_synthetic_passes(tmp_path) -> None:
    """Regenerating the same audio verifies.

    The distance is not exactly zero because the committed fingerprint is
    rounded to 6 decimal places; that residual is ~5e-7 dB against a 0.01 dB
    budget.
    """
    pcm = speech_like()
    failure, distance = verify_synthetic(synthetic_entry(pcm), write(tmp_path / "s.wav", pcm), CANONICAL_TOLERANCE_DB)
    assert failure is None
    assert distance < 1e-6, "stored-fingerprint rounding should be negligible"


@pytest.mark.parametrize("lsb", [1, 2, 8, 17])
def test_small_float_variation_passes(tmp_path, lsb: int) -> None:
    """Bounded LSB differences, as produced by float kernels, are accepted."""
    pcm = speech_like()
    entry = synthetic_entry(pcm)
    rng = np.random.default_rng(3)
    varied = np.clip(pcm.astype(int) + rng.integers(-lsb, lsb + 1, len(pcm)), -32767, 32767).astype(np.int16)
    failure, distance = verify_synthetic(entry, write(tmp_path / "s.wav", varied), CANONICAL_TOLERANCE_DB)
    assert failure is None, f"+/-{lsb} LSB rejected: {failure}"
    assert distance < CANONICAL_TOLERANCE_DB


def test_wav_sha_is_not_the_gate(tmp_path) -> None:
    """A different full-file hash does not fail the synthetic check.

    This is the R2 defect: bit equality was the gate and it is not reproducible
    across machines.
    """
    pcm = speech_like()
    entry = synthetic_entry(pcm)
    rng = np.random.default_rng(5)
    varied = np.clip(pcm.astype(int) + rng.integers(-3, 4, len(pcm)), -32767, 32767).astype(np.int16)
    path = write(tmp_path / "s.wav", varied)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
    assert verify_synthetic(entry, path, CANONICAL_TOLERANCE_DB)[0] is None


def test_meaningful_corruption_fails(tmp_path) -> None:
    """A half-percent level change is rejected."""
    pcm = speech_like()
    entry = synthetic_entry(pcm)
    louder = np.clip(pcm.astype(np.float64) * 1.005, -32767, 32767).astype(np.int16)
    failure, _ = verify_synthetic(entry, write(tmp_path / "s.wav", louder), CANONICAL_TOLERANCE_DB)
    assert failure is not None
    assert "envelope differs" in failure


def test_dropout_fails(tmp_path) -> None:
    """A zeroed span is rejected."""
    pcm = speech_like()
    entry = synthetic_entry(pcm)
    broken = pcm.copy()
    broken[4096 * 5 : 4096 * 6] = 0
    assert verify_synthetic(entry, write(tmp_path / "s.wav", broken), CANONICAL_TOLERANCE_DB)[0] is not None


def test_wrong_speaker_or_text_fails(tmp_path) -> None:
    """Different speech of the same length is rejected."""
    pcm = speech_like(seed=0)
    entry = synthetic_entry(pcm)
    other = speech_like(seed=77) * 2
    assert verify_synthetic(entry, write(tmp_path / "s.wav", other[: len(pcm)]), CANONICAL_TOLERANCE_DB)[0] is not None


def test_wrong_sample_rate_fails(tmp_path) -> None:
    """A rate mismatch is rejected before the envelope is even compared."""
    pcm = speech_like()
    entry = synthetic_entry(pcm)
    failure, _ = verify_synthetic(entry, write(tmp_path / "s.wav", pcm, rate=16_000), CANONICAL_TOLERANCE_DB)
    assert failure is not None
    assert "sample rate" in failure


def test_wrong_sample_count_fails(tmp_path) -> None:
    """A length mismatch is rejected, which also covers wrong length_scale."""
    pcm = speech_like()
    entry = synthetic_entry(pcm)
    failure, _ = verify_synthetic(entry, write(tmp_path / "s.wav", pcm[: len(pcm) - 4096]), CANONICAL_TOLERANCE_DB)
    assert failure is not None
    assert "sample count" in failure


def test_wrong_channel_count_fails(tmp_path) -> None:
    """Stereo output is rejected."""
    pcm = speech_like()
    entry = synthetic_entry(pcm)
    path = tmp_path / "s.wav"
    sf.write(path, np.stack([pcm, pcm], axis=1), SAMPLE_RATE, subtype="PCM_16")
    failure, _ = verify_synthetic(entry, path, CANONICAL_TOLERANCE_DB)
    assert failure is not None
    assert "channel count" in failure


def test_manifest_without_envelope_fails_loudly(tmp_path) -> None:
    """An old manifest cannot silently pass the new gate."""
    pcm = speech_like()
    entry = synthetic_entry(pcm)
    del entry["pcm_envelope_db"]
    failure, _ = verify_synthetic(entry, write(tmp_path / "s.wav", pcm), CANONICAL_TOLERANCE_DB)
    assert failure is not None
    assert "pcm_envelope_db" in failure


# --- generator inputs ---------------------------------------------------------


def test_wrong_voice_model_fails(tmp_path) -> None:
    """A voice model that is not the one the manifest names is fatal."""
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"not the real voice")
    (tmp_path / "voice.onnx.json").write_bytes(b"{}")
    entry = {"voice_sha256": "f" * 64, "voice_config_sha256": hashlib.sha256(b"{}").hexdigest()}
    with pytest.raises(SystemExit, match="voice model checksum mismatch"):
        verify_voice(voice, entry)


def test_wrong_voice_config_fails(tmp_path) -> None:
    """A correct model with the wrong config is still fatal."""
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"real voice bytes")
    (tmp_path / "voice.onnx.json").write_bytes(b'{"wrong": true}')
    entry = {"voice_sha256": hashlib.sha256(b"real voice bytes").hexdigest(), "voice_config_sha256": "e" * 64}
    with pytest.raises(SystemExit, match="voice config checksum mismatch"):
        verify_voice(voice, entry)


def test_matching_voice_and_config_pass(tmp_path) -> None:
    """The happy path verifies both files."""
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"real voice bytes")
    config = tmp_path / "voice.onnx.json"
    config.write_bytes(b'{"ok": true}')
    verify_voice(voice, {
        "voice_sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
        "voice_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    })


def test_missing_config_hash_in_manifest_fails(tmp_path) -> None:
    """A manifest predating config verification cannot pass."""
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"v")
    (tmp_path / "voice.onnx.json").write_bytes(b"c")
    with pytest.raises(SystemExit, match="voice_config_sha256"):
        verify_voice(voice, {"voice_sha256": hashlib.sha256(b"v").hexdigest()})


# --- the committed manifest ---------------------------------------------------


@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest not present")
def test_committed_manifest_carries_the_new_verification_fields() -> None:
    """Every synthetic entry has what the tolerant gate needs."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] >= 2
    assert manifest["verification"]["canonical_tolerance_db"] == CANONICAL_TOLERANCE_DB
    assert manifest["authoring_environment"]["onnxruntime"]

    for sample in manifest["samples"]:
        if sample["label"] != "spoof":
            continue
        assert {"pcm_envelope_db", "voice_sha256", "voice_config_sha256", "sample_count", "channels", "sample_rate"} <= sample.keys()
        assert len(sample["pcm_envelope_db"]) > 0
        assert all(isinstance(v, float) for v in sample["pcm_envelope_db"])
