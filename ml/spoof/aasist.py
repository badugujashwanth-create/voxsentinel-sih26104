"""AASIST anti-spoofing detector.

Model
-----
AASIST - "Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention
Networks", Jung et al., ICASSP 2022.

- Source:     https://github.com/clovaai/aasist (official implementation)
- Licence:    MIT, (c) 2021-present NAVER Corp.
- Checkpoint: ``models/weights/AASIST.pth`` from that repository, pinned to
              commit ``a04c9863``. Trained by the authors on the ASVspoof 2019
              Logical Access (LA) train partition.
- Input:      raw waveform, mono, 16 kHz, exactly 64600 samples (4.04 s). The
              official pipeline tiles shorter audio and truncates longer audio.
- Output:     two class scores. Index 1 is bona fide, index 0 is spoof; the
              official evaluation uses index 1 as the detection score.

Neither the definition nor the checkpoint is committed here. Run
``ml/scripts/setup_aasist.py`` to fetch and verify them.

Domain caveat
-------------
ASVspoof 2019 LA contains text-to-speech and voice-conversion attacks built with
2019-era systems. Speech from newer synthesis models, and audio that has been
through telephony codecs, is out of that training domain. Treat scores on such
audio as indicative, not as a measured accuracy claim.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from ml.audio import preprocessing
from ml.spoof.detector import SpoofDetector, SpoofResult

#: Fixed input length the released checkpoint expects, in samples (4.0375 s).
AASIST_INPUT_SAMPLES = 64_600
AASIST_SAMPLE_RATE = 16_000

#: ``model_config`` from ``config/AASIST.conf`` in the upstream repository. The
#: released checkpoint only loads against exactly these dimensions.
AASIST_MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": AASIST_INPUT_SAMPLES,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

#: Index of the bona-fide class in the model's output.
BONAFIDE_INDEX = 1
SPOOF_INDEX = 0

DEFAULT_VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "aasist"


class ModelNotInstalledError(RuntimeError):
    """Raised when the AASIST definition or checkpoint is missing."""


def _load_upstream_module(definition_path: Path) -> ModuleType:
    """Imports the vendored upstream ``AASIST.py`` as a module."""
    spec = importlib.util.spec_from_file_location("voxsentinel_vendor_aasist", definition_path)
    if spec is None or spec.loader is None:
        raise ModelNotInstalledError(f"cannot import AASIST definition at {definition_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pad_to_model_length(samples: np.ndarray, length: int = AASIST_INPUT_SAMPLES) -> tuple[np.ndarray, list[str]]:
    """Tiles or truncates audio to the model's fixed input length.

    Mirrors ``pad()`` in the upstream ``data_utils.py`` so scores stay
    comparable with the authors' published evaluation.
    """
    count = len(samples)
    if count == length:
        return samples, []
    if count > length:
        return samples[:length], [f"truncated {count / AASIST_SAMPLE_RATE:.2f}s of audio to the model's {length / AASIST_SAMPLE_RATE:.2f}s window"]
    repeats = length // count + 1
    tiled = np.tile(samples, repeats)[:length]
    return tiled, [f"audio was {count / AASIST_SAMPLE_RATE:.2f}s; tiled {repeats}x to fill the model's {length / AASIST_SAMPLE_RATE:.2f}s window, which weakens the result"]


class AASISTSpoofDetector(SpoofDetector):
    """Scores audio with the released AASIST checkpoint."""

    def __init__(self, vendor_dir: Path | str = DEFAULT_VENDOR_DIR, device: str = "cpu") -> None:
        vendor = Path(vendor_dir)
        definition = vendor / "AASIST.py"
        weights = vendor / "AASIST.pth"
        missing = [path.name for path in (definition, weights) if not path.is_file()]
        if missing:
            raise ModelNotInstalledError(f"missing {', '.join(missing)} in {vendor}. Run: python ml/scripts/setup_aasist.py")

        self._device = torch.device(device)
        upstream = _load_upstream_module(definition)
        self._model = upstream.Model(json.loads(json.dumps(AASIST_MODEL_CONFIG)))
        self._model.load_state_dict(torch.load(weights, map_location=self._device, weights_only=True))
        self._model.to(self._device).eval()
        self._checkpoint_name = weights.name

    @property
    def model_id(self) -> str:
        """Identifies the model, checkpoint, and training corpus."""
        return f"AASIST/{self._checkpoint_name}@ASVspoof2019-LA"

    @property
    def expected_sample_rate(self) -> int:
        """AASIST operates on 16 kHz audio."""
        return AASIST_SAMPLE_RATE

    def score(self, audio: np.ndarray, sample_rate: int) -> SpoofResult:
        """Returns the probability that ``audio`` is synthetic."""
        preprocess_start = time.perf_counter()
        prepared = preprocessing.prepare(audio, sample_rate, AASIST_SAMPLE_RATE)
        window, pad_warnings = pad_to_model_length(prepared.samples)
        tensor = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(0).to(self._device)
        preprocess_seconds = time.perf_counter() - preprocess_start

        inference_start = time.perf_counter()
        with torch.no_grad():
            _, logits = self._model(tensor)
        inference_seconds = time.perf_counter() - inference_start

        scores = logits.squeeze(0).float().cpu()
        probabilities = torch.softmax(scores, dim=0)

        return SpoofResult(
            synthetic_probability=float(probabilities[SPOOF_INDEX]),
            bonafide_score=float(scores[BONAFIDE_INDEX]),
            raw_scores=tuple(float(value) for value in scores),
            model_id=self.model_id,
            inference_seconds=inference_seconds,
            preprocessing_seconds=preprocess_seconds,
            audio_duration_seconds=prepared.duration_seconds,
            sample_rate=AASIST_SAMPLE_RATE,
            warnings=tuple([*prepared.warnings, *pad_warnings]),
        )

    def score_model_window(self, samples: np.ndarray) -> SpoofResult:
        """Scores one backend-canonical exact model window without reprocessing it."""
        if not isinstance(samples, np.ndarray) or samples.ndim != 1 or samples.shape[0] != AASIST_INPUT_SAMPLES:
            raise ValueError(f"samples must contain exactly {AASIST_INPUT_SAMPLES} values")
        if samples.dtype != np.dtype("float32") or not np.isfinite(samples).all():
            raise ValueError("samples must be finite float32 values")
        contiguous_samples = np.ascontiguousarray(samples, dtype=np.float32)
        tensor = torch.from_numpy(contiguous_samples).unsqueeze(0).to(self._device)
        inference_start = time.perf_counter()
        with torch.no_grad():
            _, logits = self._model(tensor)
        inference_seconds = time.perf_counter() - inference_start
        scores = logits.squeeze(0).float().cpu()
        probabilities = torch.softmax(scores, dim=0)
        return SpoofResult(
            synthetic_probability=float(probabilities[SPOOF_INDEX]),
            bonafide_score=float(scores[BONAFIDE_INDEX]),
            raw_scores=tuple(float(value) for value in scores),
            model_id=self.model_id,
            inference_seconds=inference_seconds,
            preprocessing_seconds=0.0,
            audio_duration_seconds=AASIST_INPUT_SAMPLES / AASIST_SAMPLE_RATE,
            sample_rate=AASIST_SAMPLE_RATE,
            warnings=(),
        )
