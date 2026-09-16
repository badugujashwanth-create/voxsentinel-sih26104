# ML

Ownership boundary: Rohan.

Voice spoof / deepfake detection for VoxSentinel.

> ## What this subsystem does and does not answer
>
> It answers exactly one question: **how likely is it that this audio is
> synthetic or spoofed?**
>
> It is **not** a fraud decision, **not** an identity decision, and **not** a
> VoxSentinel 0-100 risk score. Speaker verification, prosody, and transaction
> context are separate signals, and fusing them into a final risk number is a
> later task. Nothing here is wired into the live backend yet:
> `MockRiskProvider` is still what serves the demo.

## Contents

| Path | Purpose |
| --- | --- |
| `spoof/detector.py` | `SpoofDetector` interface and the `SpoofResult` contract |
| `spoof/aasist.py` | `AASISTSpoofDetector` - the real model adapter |
| `audio/preprocessing.py` | decode, downmix, resample, and validate audio |
| `audio/chunker.py` | streaming windowing (from ROHAN-001) |
| `scripts/setup_aasist.py` | fetch and verify the model + checkpoint |
| `scripts/detect.py` | score one or more files |
| `scripts/evaluate.py` | metrics over a labelled manifest |
| `scripts/benchmark_latency.py` | measured latency on this machine |
| `scripts/prepare_eval_set.py` | assemble a labelled evaluation set |
| `scripts/generate_tts_samples.py` | generate synthetic samples (optional tool) |

## Model

**AASIST** - "Audio Anti-Spoofing using Integrated Spectro-Temporal Graph
Attention Networks", Jung et al., ICASSP 2022.

| | |
| --- | --- |
| Source | <https://github.com/clovaai/aasist> (official implementation, NAVER / Clova AI) |
| Licence | MIT, (c) 2021-present NAVER Corp. |
| Pinned commit | `a04c9863f63d44471dde8a6abcb3b082b07cd1d1` |
| Checkpoint | `models/weights/AASIST.pth` from that repository, trained by the authors |
| Checkpoint SHA-256 | `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0` |
| Parameters | 297,866 |
| Training corpus | ASVspoof 2019 Logical Access (LA), train partition |
| Input | raw waveform, mono, 16 kHz, exactly 64600 samples (4.0375 s) |
| Output | two scores; index 1 = bona fide, index 0 = spoof |

Chosen because it is the model the task named, it is the official
implementation rather than a re-upload, the licence is unambiguous, the
checkpoint ships with the source, and it is small enough to run on CPU.

Neither the model code nor the checkpoint is committed here. `setup_aasist.py`
downloads both from the pinned commit and refuses to install anything whose
SHA-256 does not match.

### Preprocessing applied before inference

1. Decode with `libsndfile` (WAV, FLAC, OGG).
2. Downmix to mono by averaging channels.
3. Resample to 16 kHz with a polyphase filter (`scipy.signal.resample_poly`).
4. Reject unusable audio (see below).
5. Tile or truncate to exactly 64600 samples, matching upstream `pad()` so
   scores stay comparable with the authors' published evaluation.

No amplitude normalisation is applied, because the released checkpoint was
trained on unnormalised waveforms.

### Input requirements

| | |
| --- | --- |
| Sample rate | any; resampled to 16 kHz |
| Channels | any; downmixed to mono |
| Minimum duration | 1.0 s, measured after resampling |
| Ideal duration | >= 4.04 s, so no tiling is needed |
| Format | anything `libsndfile` decodes |

Audio that cannot produce an honest answer is **refused, not scored**:

| Condition | Result |
| --- | --- |
| Silence (peak <= 1e-4) | `SilentAudioError` |
| Shorter than 1.0 s | `AudioTooShortError` |
| Undecodable, empty, NaN/Inf, or integer samples | `MalformedAudioError` |

Anything done to make audio usable, and anything that weakens confidence, is
reported in `SpoofResult.warnings` - resampling, downmixing, tiling short
audio, very low signal level, and clipping.

### Known limitations

- **Domain.** ASVspoof 2019 LA contains 2019-era TTS and voice-conversion
  attacks. Speech from newer synthesis systems is out of that training domain.
- **No telephony conditions.** The training data is clean studio-derived audio.
  Narrowband codecs, packet loss, and network jitter are not represented, and
  VoxSentinel's eventual input is a phone call.
- **English-centric.** The training corpus is English. No multilingual claim is
  made or supported here.
- **Fixed 4.04 s window.** Longer audio is truncated; a call needs windowing
  (see `audio/chunker.py`) and a strategy for combining per-window scores,
  which this task does not define.
- **No replay attacks.** The LA track covers synthetic speech, not the physical
  replay attacks of the PA track. `replay_risk_score` remains unaddressed.
- **Not calibrated.** The output is a softmax probability, not a calibrated
  likelihood. The 0.5 threshold is an arbitrary midpoint, not a tuned operating
  point.

## Setup

The ML runtime is deliberately separate from `backend/requirements.txt`; the
demo backend stays free of torch until `MLRiskProvider` lands.

```bash
# from the repository root
uv venv ml/.venv --python 3.12

# torch MUST come from the CPU index, or pip/uv resolves the CUDA build
# from PyPI and pulls several GB of unused NVIDIA wheels
uv pip install --python ml/.venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
uv pip install --python ml/.venv/bin/python -r ml/requirements.txt

# fetch and verify the model definition and checkpoint (~1.3 MB)
ml/.venv/bin/python ml/scripts/setup_aasist.py
```

`setup_aasist.py --verify-only` re-checks the installed files without
downloading. Everything it fetches lands in `ml/spoof/vendor/`, which is
gitignored.

## Inference

```bash
ml/.venv/bin/python ml/scripts/detect.py --audio path/to/sample.wav
```

Real output, one genuine and one synthetic sample:

```
File:                  ml/data/eval/genuine/librispeech_1272-128104-0000.flac
Synthetic probability: 0.0007
Bona-fide score:       +2.8367  (raw model output, higher = more human)
Model:                 AASIST/AASIST.pth@ASVspoof2019-LA
Audio duration:        5.86s at 16000 Hz
Preprocessing:         0.5 ms
Inference latency:     476.4 ms
Total:                 477.0 ms
Warnings:
  - truncated 5.86s of audio to the model's 4.04s window

File:                  ml/data/eval/synthetic/tts_000_spk000.wav
Synthetic probability: 0.6707
Bona-fide score:       -0.6457  (raw model output, higher = more human)
Model:                 AASIST/AASIST.pth@ASVspoof2019-LA
Audio duration:        4.95s at 16000 Hz
Preprocessing:         2.9 ms
Inference latency:     437.5 ms
Total:                 440.4 ms
Warnings:
  - resampled 22050Hz to 16000Hz
  - truncated 4.95s of audio to the model's 4.04s window

Spoof-detection score only - not an identity or fraud decision.
```

Multiple files and JSON output are supported:

```bash
ml/.venv/bin/python ml/scripts/detect.py --audio a.wav b.flac --json
```

## Tests

The ML tests live under `tests/ml/` and guard their imports, so the backend
suite skips them cleanly in its own torch-free environment.

```bash
# ML suite (fast unit tests + real-inference integration tests)
ml/.venv/bin/python -m pytest tests/ml

# fast tests only, no model needed
ml/.venv/bin/python -m pytest tests/ml -m "not integration"

# backend suite, unchanged and still torch-free
backend/.venv/bin/python -m pytest
```

Integration tests load the real checkpoint and run a real forward pass. They
skip with a clear reason when the model is not installed; they are never
mocked and then reported as passing.

## Evaluation

```bash
# assemble a labelled set, then score it
ml/.venv/bin/python ml/scripts/prepare_eval_set.py --librispeech ml/data/librispeech
ml/.venv/bin/python ml/scripts/evaluate.py --manifest ml/data/eval/manifest.csv --per-sample
```

The manifest is CSV with `path,label`, where label is `bonafide` or `spoof`,
and paths resolve relative to the manifest.

`--threshold` sets the decision point (default `0.5`, an arbitrary midpoint,
**not** a tuned operating point). The runner prints counts, the confusion
matrix, accuracy, precision, recall, and F1, and warns loudly when there are
fewer than 50 samples per class - at which point the numbers describe those
files and nothing more.

### Sample sources

| Class | Source | Licence |
| --- | --- | --- |
| Genuine | LibriSpeech `dev-clean` (OpenSLR 12), read speech from public-domain LibriVox audiobooks | CC BY 4.0 |
| Synthetic | Piper TTS, voice `en_US-libritts_r-medium` (904 speakers, trained on LibriTTS-R) | voice CC BY 4.0; Piper itself GPL-3.0 |

Both classes derive from the same audiobook domain, so the comparison isolates
real-versus-synthetic rather than also changing recording conditions.

Piper is a sample-generation tool only. It is **not** a dependency of the
detector and is not in `ml/requirements.txt`; install it in a throwaway
environment:

```bash
uv venv /tmp/ttsvenv --python 3.12
uv pip install --python /tmp/ttsvenv/bin/python piper-tts==1.8.0
/tmp/ttsvenv/bin/python ml/scripts/generate_tts_samples.py
```

No private or team voice recordings are used, and no audio is committed.

### Measured results

80 samples, 40 genuine and 40 synthetic, threshold 0.5:

```
Synthetic-probability distribution:
  bonafide  n=40   min=0.0000 median=0.0031 max=0.9580 stdev=0.2437
  spoof     n=40   min=0.0682 median=0.9285 max=0.9980 stdev=0.2714

Confusion matrix:
                    predicted spoof     predicted bonafide
  actual spoof                     33                    7
  actual bonafide                   4                   36

Metrics:
  accuracy   0.8625
  precision  0.8919
  recall     0.8250
  f1         0.8571
  EER        0.1000   at threshold 0.3060
```

**These numbers are not an accuracy claim.** 80 samples from one TTS system and
one read-speech corpus describe those 80 files. They are not a benchmark
result and must not be quoted as one.

What they do show:

- **The model separates the two classes.** Median genuine 0.0031 against median
  synthetic 0.9285 is real signal, not noise.
- **The domain gap is visible and large.** A 10% EER here against roughly 0.83%
  published by the authors on in-domain ASVspoof 2019 LA eval is the cost of
  scoring a 2023-era multi-speaker VITS voice with a model trained on 2019
  attacks. Seven of forty synthetic samples slipped under 0.5.
- **0.5 is the wrong threshold.** EER lands at 0.306. Any operating point must
  be tuned on a real dev set before it goes anywhere near a decision.

The honest next step is evaluation against ASVspoof 2019 LA eval, which is
in-domain and properly sized, before any performance claim is made.

## Latency

```bash
ml/.venv/bin/python ml/scripts/benchmark_latency.py --audio path/to/sample.flac
```

Measured on a real sample, 3 warm-up plus 20 measured passes:

```
Model:            AASIST/AASIST.pth@ASVspoof2019-LA
Source:           ml/data/eval/genuine/librispeech_1272-128104-0000.flac
Audio duration:   5.86s (model window 4.04s)

Hardware:
  device                       cpu
  cpu                          13th Gen Intel(R) Core(TM) i5-13420H
  cpu_threads_used_by_torch    6
  logical_cores                12
  platform                     Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
  torch                        2.9.1+cpu
  gpu_present_but_unused       NVIDIA GeForce RTX 3050 6GB Laptop GPU

Latency (milliseconds):
  stage               mean    median       min       max       p95
  preprocessing       0.33      0.30      0.24      0.54      0.52
  inference         498.41    450.27    409.34   1149.36    695.86
  total             498.74    450.52    409.75   1149.90    696.17

Real-time factor: 9.0x  (model window / median total)
```

Preprocessing is negligible; essentially all the cost is the forward pass.
Scoring a 4.04 s window in a median 450 ms is about 9x faster than real time
on CPU, which leaves headroom for the 0.5 s hop in `audio/chunker.py`.

Caveats: this is one laptop CPU under WSL2, the spread is wide (p95 696 ms,
max 1149 ms) because the machine is not otherwise idle, and this measures the
model call only. It says nothing about end-to-end latency in a live call
pipeline, which does not exist yet. The GPU was present but unused.

## Privacy

No raw audio is stored or logged by this subsystem. Downloaded corpora,
generated samples, and model checkpoints all live under gitignored paths and
are never committed.
