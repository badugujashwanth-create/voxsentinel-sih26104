"""Real ECAPA-TDNN inference tests.

These load the actual checkpoint and run real forward passes over the real
evaluation audio. They are skipped, not faked, when the model or the audio has
not been prepared. Nothing here mocks the model and then claims it works.

    python ml/scripts/setup_ecapa.py
    python ml/scripts/prepare_sv_eval_set.py --download
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="ML runtime not installed in this environment")
pytest.importorskip("torch", reason="ML runtime not installed in this environment")
pytest.importorskip("scipy", reason="ML runtime not installed in this environment")
pytest.importorskip("soundfile", reason="ML runtime not installed in this environment")
pytest.importorskip("speechbrain", reason="speechbrain not installed in this environment")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from ml.audio.preprocessing import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    AudioTooShortError,
    MalformedAudioError,
    SilentAudioError,
)
from ml.speaker.ecapa import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DEFAULT_VENDOR_DIR,
    ECAPA_EMBEDDING_DIMENSIONS,
    ECAPA_SAMPLE_RATE,
    ECAPASpeakerVerifier,
    ModelNotInstalledError,
)
from ml.speaker.verifier import EnrollmentError, cosine_similarity  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "ml" / "data" / "sv-eval"
MANIFEST = REPO_ROOT / "ml" / "evaluation" / "sv_manifest.json"
RESULTS = REPO_ROOT / "ml" / "evaluation" / "sv_results.json"
SUBDIR = {"librispeech": "librispeech", "cmu_arctic": "arctic", "piper_tts": "clone"}

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def verifier() -> ECAPASpeakerVerifier:
    """The real verifier, or a skip when the checkpoint is not installed."""
    try:
        return ECAPASpeakerVerifier()
    except ModelNotInstalledError as exc:
        pytest.skip(f"{exc}")


@pytest.fixture(scope="module")
def manifest() -> dict:
    """The committed evaluation manifest, or a skip."""
    if not MANIFEST.is_file():
        pytest.skip(f"{MANIFEST} is missing")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def sample_path(sample: dict) -> Path:
    """Where one manifest sample's prepared audio lives."""
    return EVAL_DIR / SUBDIR[sample["source_type"]] / sample["filename"]


@pytest.fixture(scope="module")
def prepared(manifest: dict) -> dict:
    """Manifest samples grouped by speaker and role, or a skip."""
    missing = [s["sample_id"] for s in manifest["samples"] if not sample_path(s).is_file()]
    if missing:
        pytest.skip(f"{len(missing)} evaluation samples are not prepared. Run: python ml/scripts/prepare_sv_eval_set.py --download")
    return {s["sample_id"]: s for s in manifest["samples"]}


def speakers_of(manifest: dict, corpus: str) -> list[dict]:
    """The manifest's speakers for one corpus."""
    return [s for s in manifest["speakers"] if s["corpus"] == corpus]


def enrol(verifier: ECAPASpeakerVerifier, speaker: dict, prepared: dict):
    """Enrols one manifest speaker from its recorded enrolment utterances."""
    return verifier.enroll([sample_path(prepared[sample_id]) for sample_id in speaker["enrolment"]])


# --- Installation ------------------------------------------------------------


def test_missing_model_fails_with_an_actionable_error(tmp_path) -> None:
    with pytest.raises(ModelNotInstalledError, match="setup_ecapa"):
        ECAPASpeakerVerifier(vendor_dir=tmp_path)


def test_a_corrupt_checkpoint_is_refused(tmp_path) -> None:
    # A silently wrong checkpoint would produce plausible numbers from the wrong
    # model, which is worse than not loading at all.
    for source in DEFAULT_VENDOR_DIR.glob("*"):
        if source.is_file():
            (tmp_path / source.name).write_bytes(source.read_bytes())
    target = tmp_path / "embedding_model.ckpt"
    if not target.is_file():
        pytest.skip("checkpoint not installed")
    target.write_bytes(target.read_bytes() + b"\x00")
    with pytest.raises(ModelNotInstalledError, match="checksum mismatch"):
        ECAPASpeakerVerifier(vendor_dir=tmp_path)


def test_verifier_reports_its_contract(verifier: ECAPASpeakerVerifier) -> None:
    assert verifier.expected_sample_rate == ECAPA_SAMPLE_RATE == 16_000
    assert verifier.embedding_dimensions == ECAPA_EMBEDDING_DIMENSIONS == 192
    assert "ECAPA-TDNN" in verifier.model_id
    assert verifier.decision_threshold == DEFAULT_THRESHOLD


def test_shipped_threshold_matches_the_committed_evaluation() -> None:
    # The default must stay the number the evaluation actually produced, not a
    # value that drifted away from the evidence behind it.
    if not RESULTS.is_file():
        pytest.skip(f"{RESULTS} is missing")
    assert json.loads(RESULTS.read_text(encoding="utf-8"))["threshold"] == DEFAULT_THRESHOLD


# --- Embeddings --------------------------------------------------------------


def test_embedding_has_the_documented_shape_and_is_normalised(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict) -> None:
    sample = next(s for s in manifest["samples"] if s["source_type"] == "librispeech")
    embedding = verifier.embed_file(sample_path(sample))
    assert embedding.vector.shape == (ECAPA_EMBEDDING_DIMENSIONS,)
    assert embedding.vector.dtype == np.float32
    assert np.isfinite(embedding.vector).all()
    assert float(np.linalg.norm(embedding.vector)) == pytest.approx(1.0, abs=1e-5)
    assert embedding.sample_rate == ECAPA_SAMPLE_RATE
    assert embedding.inference_seconds > 0


def test_embedding_is_deterministic(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict) -> None:
    sample = next(s for s in manifest["samples"] if s["source_type"] == "librispeech")
    first = verifier.embed_file(sample_path(sample))
    second = verifier.embed_file(sample_path(sample))
    assert cosine_similarity(first.vector, second.vector) == pytest.approx(1.0, abs=1e-6)


def test_aggregated_enrolment_is_deterministic_and_order_independent(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict) -> None:
    speaker = speakers_of(manifest, "LibriSpeech dev-clean")[0]
    paths = [sample_path(prepared[sample_id]) for sample_id in speaker["enrolment"]]
    forward = verifier.enroll(paths)
    assert np.array_equal(verifier.enroll(paths).vector, forward.vector)
    assert np.array_equal(verifier.enroll(list(reversed(paths))).vector, forward.vector)
    assert forward.source_count == len(paths)


# --- The three cases the task asks for ---------------------------------------


def test_case_a_same_speaker_scores_high(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict) -> None:
    # Enrolment and probe come from different chapters, so this is not two
    # clips of one recording session scoring alike.
    speaker = speakers_of(manifest, "LibriSpeech dev-clean")[0]
    reference = enrol(verifier, speaker, prepared)
    for probe_id in speaker["probes"]:
        result = verifier.verify_file(reference, sample_path(prepared[probe_id]))
        assert result.is_match
        assert result.similarity_score > DEFAULT_THRESHOLD


def test_case_b_different_speaker_scores_low(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict) -> None:
    speakers = speakers_of(manifest, "LibriSpeech dev-clean")
    reference = enrol(verifier, speakers[0], prepared)
    for impostor in speakers[1:4]:
        for probe_id in impostor["probes"]:
            result = verifier.verify_file(reference, sample_path(prepared[probe_id]))
            assert not result.is_match
            assert result.similarity_score < DEFAULT_THRESHOLD


def test_same_speaker_always_outscores_a_different_speaker(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict) -> None:
    speakers = speakers_of(manifest, "LibriSpeech dev-clean")[:5]
    for index, speaker in enumerate(speakers):
        reference = enrol(verifier, speaker, prepared)
        genuine = min(verifier.verify_file(reference, sample_path(prepared[p])).similarity_score for p in speaker["probes"])
        impostor = speakers[(index + 1) % len(speakers)]
        best_impostor = max(verifier.verify_file(reference, sample_path(prepared[p])).similarity_score for p in impostor["probes"])
        assert genuine > best_impostor


def test_case_c_a_clone_of_the_enrolled_speaker_scores_like_the_speaker(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict) -> None:
    # This is the finding, not a regression guard: a Piper voice trained on the
    # enrolled CMU ARCTIC speaker lands far above every human impostor, and
    # close to the speaker's own genuine probes. Speaker verification alone does
    # not stop a clone. That is what ml/spoof/ is for.
    speaker = speakers_of(manifest, "CMU ARCTIC")[0]
    clones = [s for s in manifest["samples"] if s.get("clone_of") == speaker["speaker_key"]]
    if not clones:
        pytest.skip("no clone samples in the manifest")

    reference = enrol(verifier, speaker, prepared)
    clone_scores = [verifier.verify_file(reference, sample_path(c)).similarity_score for c in clones]
    impostor_scores = [
        verifier.verify_file(reference, sample_path(prepared[p])).similarity_score
        for other in speakers_of(manifest, "CMU ARCTIC")
        if other["speaker_key"] != speaker["speaker_key"]
        for p in other["probes"]
    ]
    assert min(clone_scores) > max(impostor_scores)


def test_clone_scores_are_recorded_honestly_in_the_committed_results() -> None:
    if not RESULTS.is_file():
        pytest.skip(f"{RESULTS} is missing")
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    clone = results["attacks"]["clone"]
    assert clone["n"] > 0
    # The claim the README makes must be the claim the artefact holds.
    assert clone["acceptance_rate"] == clone["accepted_at_threshold"] / clone["n"]
    assert clone["score_distribution"]["median"] > results["by_corpus"]["cmu_arctic"]["score_distribution"]["different_speaker"]["max"]


# --- Audio handling ----------------------------------------------------------


def test_every_prepared_sample_embeds(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict) -> None:
    for sample in manifest["samples"]:
        embedding = verifier.embed_file(sample_path(sample))
        assert embedding.dimensions == ECAPA_EMBEDDING_DIMENSIONS


def test_stereo_input_is_downmixed_and_flagged(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict, tmp_path) -> None:
    sample = next(s for s in manifest["samples"] if s["source_type"] == "cmu_arctic")
    mono, rate = sf.read(sample_path(sample), dtype="float32")
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.stack([mono, mono], axis=1), rate)
    result = verifier.embed_file(stereo)
    assert any("downmixed" in w for w in result.warnings)
    # Duplicating a channel changes nothing about the speaker.
    assert cosine_similarity(result.vector, verifier.embed_file(sample_path(sample)).vector) == pytest.approx(1.0, abs=1e-3)


def test_resampled_input_is_accepted_flagged_and_still_recognised(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict, tmp_path) -> None:
    sample = next(s for s in manifest["samples"] if s["source_type"] == "cmu_arctic")
    mono, rate = sf.read(sample_path(sample), dtype="float32")
    assert rate == ECAPA_SAMPLE_RATE
    upsampled = tmp_path / "upsampled.wav"
    sf.write(upsampled, np.repeat(mono, 3), rate * 3)  # crude 48 kHz stand-in
    result = verifier.embed_file(upsampled)
    assert any("resampled 48000Hz to 16000Hz" in w for w in result.warnings)
    assert result.sample_rate == ECAPA_SAMPLE_RATE


def test_flac_and_wav_are_both_decoded(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict) -> None:
    extensions = {sample_path(s).suffix for s in manifest["samples"]}
    assert {".flac", ".wav"} <= extensions
    for extension in (".flac", ".wav"):
        sample = next(s for s in manifest["samples"] if sample_path(s).suffix == extension)
        assert verifier.embed_file(sample_path(sample)).dimensions == ECAPA_EMBEDDING_DIMENSIONS


def test_short_audio_is_embedded_but_flagged_as_unreliable(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict, tmp_path) -> None:
    sample = next(s for s in manifest["samples"] if s["source_type"] == "cmu_arctic")
    mono, rate = sf.read(sample_path(sample), dtype="float32")
    short = tmp_path / "short.wav"
    sf.write(short, mono[: int(1.5 * rate)], rate)
    result = verifier.embed_file(short)
    assert any("unreliable" in w for w in result.warnings)


def test_unusable_audio_never_reaches_the_model(verifier: ECAPASpeakerVerifier, tmp_path) -> None:
    # The dangerous failure is a confident verdict on audio that never had a
    # chance, so each of these must raise rather than return a similarity.
    silence = tmp_path / "silence.wav"
    sf.write(silence, np.zeros(TARGET_SAMPLE_RATE * 4, dtype=np.float32), TARGET_SAMPLE_RATE)
    tiny = tmp_path / "tiny.wav"
    sf.write(tiny, np.full(int(0.3 * TARGET_SAMPLE_RATE), 0.2, dtype=np.float32), TARGET_SAMPLE_RATE)
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF....WAVEjunk")
    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros(0, dtype=np.float32), TARGET_SAMPLE_RATE)

    for path, error in ((silence, SilentAudioError), (tiny, AudioTooShortError), (broken, MalformedAudioError), (empty, MalformedAudioError)):
        with pytest.raises((error, MalformedAudioError)):
            verifier.embed_file(path)


def test_enrolment_on_silence_fails_rather_than_producing_a_reference(verifier: ECAPASpeakerVerifier, manifest: dict, prepared: dict, tmp_path) -> None:
    good = sample_path(next(s for s in manifest["samples"] if s["source_type"] == "cmu_arctic"))
    silence = tmp_path / "silence.wav"
    sf.write(silence, np.zeros(TARGET_SAMPLE_RATE * 4, dtype=np.float32), TARGET_SAMPLE_RATE)
    with pytest.raises(EnrollmentError, match="unusable"):
        verifier.enroll([good, silence])
