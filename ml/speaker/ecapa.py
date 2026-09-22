"""ECAPA-TDNN speaker verifier.

Model
-----
ECAPA-TDNN - "ECAPA-TDNN: Emphasized Channel Attention, Propagation and
Aggregation in TDNN Based Speaker Verification", Desplanques et al.,
Interspeech 2020. Checkpoint released by the SpeechBrain project.

- Source:     https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- Licence:    Apache-2.0 (both the SpeechBrain toolkit and this checkpoint)
- Revision:   pinned to commit ``0f99f2d0`` of that repository
- Trained on: VoxCeleb 1 + VoxCeleb 2 development sets
- Input:      mono waveform, 16 kHz, any length
- Features:   80-dimensional log-Mel filterbank, sentence-level mean
              normalisation (both from the released ``hyperparams.yaml``)
- Output:     a 192-dimensional speaker embedding
- Similarity: cosine, which is the scoring function the checkpoint was trained
              for (additive-angular-margin softmax shapes the embedding space
              so that angle is the speaker distance)

Neither the checkpoint nor its config is committed here. Run
``ml/scripts/setup_ecapa.py`` to fetch and verify them.

Why this model
--------------
It is the standard reproducible speaker-verification baseline: an official
release from the toolkit that trained it, an unambiguous Apache-2.0 licence, a
pinned revision with per-file hashes, small enough to run on CPU, and a fixed
16 kHz mono contract that matches the preprocessing ``ml/audio`` already does
for the spoof detector.

Domain caveats
--------------
VoxCeleb is interview and celebrity-video speech: wideband, varied but not
telephony. Scores on PSTN/VoIP audio, on narrowband codecs, and on languages
other than those VoxCeleb covers are untested here. The evaluation in
``ml/speaker/README.md`` uses clean read speech, which is easier than a phone
call, so its numbers are a ceiling rather than an expectation.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from ml.audio import preprocessing
from ml.speaker.verifier import (
    MIN_RELIABLE_PROBE_SECONDS,
    SpeakerEmbedding,
    SpeakerVerifier,
)

ECAPA_SAMPLE_RATE = 16_000
ECAPA_EMBEDDING_DIMENSIONS = 192

#: Pinned revision of huggingface.co/speechbrain/spkrec-ecapa-voxceleb.
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"

DEFAULT_VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "ecapa"

#: Every file the checkpoint needs, with the SHA-256 recorded at that revision.
EXPECTED_SHA256 = {
    "hyperparams.yaml": "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41",
    "embedding_model.ckpt": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
    "mean_var_norm_emb.ckpt": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
    "classifier.ckpt": "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
    "label_encoder.txt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
}

#: Cosine cut point separating match from mismatch.
#:
#: NOT an arbitrary midpoint and NOT a calibrated probability. It is derived
#: from measured data by ml/scripts/evaluate_speaker.py and recorded in
#: ml/evaluation/sv_results.json, which a test asserts this constant against.
#:
#: On the LibriSpeech dev-clean trials the two classes separate completely, so
#: the equal error rate is zero and its threshold is not unique: every cut point
#: in (0.4866, 0.6208] makes the same decisions. This is the midpoint of that
#: gap, which is the furthest a cut point can sit from both classes at once.
#:
#: That set is small and clean, so this is an exploratory default. See
#: ml/speaker/README.md for what it was measured on and what it is not.
DEFAULT_THRESHOLD = 0.55


class ModelNotInstalledError(RuntimeError):
    """Raised when the ECAPA checkpoint is missing or does not match."""


def _sha256(path: Path) -> str:
    """Returns a file's SHA-256 digest for checkpoint integrity validation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ECAPASpeakerVerifier(SpeakerVerifier):
    """Embeds and compares speakers with the released ECAPA-TDNN checkpoint."""

    def __init__(
        self,
        vendor_dir: Path | str = DEFAULT_VENDOR_DIR,
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        vendor = Path(vendor_dir).resolve()
        missing = [name for name in EXPECTED_SHA256 if not (vendor / name).is_file()]
        if missing:
            raise ModelNotInstalledError(f"missing {', '.join(missing)} in {vendor}. Run: python ml/scripts/setup_ecapa.py")
        for name, expected in EXPECTED_SHA256.items():
            if _sha256(vendor / name) != expected:
                raise ModelNotInstalledError(f"checksum mismatch for {name} in {vendor}. Run: python ml/scripts/setup_ecapa.py --force")

        # Imported here, not at module import time: speechbrain pulls in a large
        # dependency tree, and the contracts in verifier.py must stay importable
        # without it so the fast tests can run in a bare environment.
        from speechbrain.inference.speaker import EncoderClassifier

        self._device = torch.device(device)
        # hyperparams.yaml resolves its checkpoint paths against `pretrained_path`,
        # which defaults to the HuggingFace repo id. Pointing it at the verified
        # local directory is what keeps this load offline and hash-checked.
        self._encoder = EncoderClassifier.from_hparams(
            source=str(vendor),
            savedir=str(vendor),
            overrides={"pretrained_path": str(vendor)},
            run_opts={"device": str(self._device)},
        )
        self._encoder.eval()
        self._threshold = float(threshold)

    @property
    def model_id(self) -> str:
        """Identifies the model, revision, and training corpus."""
        return f"ECAPA-TDNN/spkrec-ecapa-voxceleb@{ECAPA_REVISION[:8]}"

    @property
    def model_revision(self) -> str:
        """Returns the full pinned Hugging Face revision."""
        return ECAPA_REVISION

    @property
    def expected_sample_rate(self) -> int:
        """ECAPA operates on 16 kHz audio."""
        return ECAPA_SAMPLE_RATE

    @property
    def embedding_dimensions(self) -> int:
        """The released checkpoint produces 192-dimensional embeddings."""
        return ECAPA_EMBEDDING_DIMENSIONS

    @property
    def decision_threshold(self) -> float:
        """Cosine cut point this verifier decides with."""
        return self._threshold

    def embed(self, audio: np.ndarray, sample_rate: int) -> SpeakerEmbedding:
        """Embeds one utterance, refusing audio it cannot honestly characterise."""
        preprocess_start = time.perf_counter()
        prepared = preprocessing.prepare(audio, sample_rate, ECAPA_SAMPLE_RATE)
        tensor = torch.from_numpy(np.ascontiguousarray(prepared.samples, dtype=np.float32)).unsqueeze(0).to(self._device)
        preprocess_seconds = time.perf_counter() - preprocess_start

        inference_start = time.perf_counter()
        with torch.no_grad():
            encoded = self._encoder.encode_batch(tensor)
        inference_seconds = time.perf_counter() - inference_start

        vector = encoded.reshape(-1).float().cpu().numpy()
        if vector.shape[0] != ECAPA_EMBEDDING_DIMENSIONS:
            raise ModelNotInstalledError(f"model returned {vector.shape[0]} dimensions, expected {ECAPA_EMBEDDING_DIMENSIONS}")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise preprocessing.MalformedAudioError("model produced a zero-length embedding for this audio")

        warnings = list(prepared.warnings)
        if prepared.duration_seconds < MIN_RELIABLE_PROBE_SECONDS:
            warnings.append(
                f"only {prepared.duration_seconds:.2f}s of speech (below {MIN_RELIABLE_PROBE_SECONDS:.1f}s); "
                f"the similarity from this short a window is unreliable"
            )

        return SpeakerEmbedding(
            vector=(vector / norm).astype(np.float32),
            model_id=self.model_id,
            source_count=1,
            total_duration_seconds=prepared.duration_seconds,
            sample_rate=ECAPA_SAMPLE_RATE,
            inference_seconds=inference_seconds,
            preprocessing_seconds=preprocess_seconds,
            warnings=tuple(warnings),
        )
