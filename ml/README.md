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
> later task. JASH-004 exposes a stateless local inference service for raw
> spoof evidence; backend call policy remains outside this subsystem.

## Contents

| Path | Purpose |
| --- | --- |
| `spoof/detector.py` | `SpoofDetector` interface and the `SpoofResult` contract |
| `spoof/aasist.py` | `AASISTSpoofDetector` - the real model adapter |
| `audio/preprocessing.py` | decode, downmix, resample, and validate audio |
| `audio/fingerprint.py` | canonical PCM envelope used to verify regenerated audio |
| `audio/chunker.py` | streaming windowing (from ROHAN-001) |
| `scripts/setup_aasist.py` | fetch and SHA-256 verify the model + checkpoint |
| `scripts/detect.py` | score one or more files |
| `scripts/evaluate.py` | score the set, write evidence, derive metrics |
| `scripts/prepare_eval_set.py` | rebuild the evaluation audio from the manifest |
| `scripts/build_eval_manifest.py` | author the manifest (run once; output committed) |
| `scripts/benchmark_latency.py` | measured latency on this machine |
| `runtime/` | stateless exact-window inference service for the backend |
| `evaluation/eval_manifest.json` | the 80 samples with provenance and hashes |
| `evaluation/predictions.csv` | per-sample evidence for the published metrics |
| `evaluation/evaluation_results.json` | summary derived from those predictions |

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

Carried forward deliberately. None of these is addressed in this task.

**Channel and domain**

- **No PSTN/VoIP codec degradation.** Training and evaluation audio is clean and
  studio-derived. VoxSentinel's eventual input is a phone call.
- **No mu-law / A-law evaluation.** Narrowband telephony coding is untested.
- **No noisy phone-channel evaluation.** No packet loss, jitter, or background noise.
- **Domain gap to modern synthesis.** ASVspoof 2019 LA contains 2019-era TTS and
  voice-conversion attacks; newer systems are out of that training domain, and
  the evaluation above shows the cost.
- **English only.** No multilingual claim is made or supported.
- **One TTS family evaluated** (Piper VITS, `en_US-libritts_r`).

**Model and method**

- **No replay / PA evaluation.** The LA track covers synthetic speech, not
  physical replay attacks. `replay_risk_score` remains unaddressed.
- **Fixed 4.04 s window.** Longer audio is truncated.
- **No cross-window aggregation.** Combining per-window scores across a call is
  undefined; `audio/chunker.py` produces the windows but nothing consumes them.
- **No calibration.** The output is a softmax score, not a calibrated likelihood.
- **Provisional threshold.** 0.5 is an arbitrary midpoint.

**Integration**

- **No final RiskProvider integration.** There is no `MLRiskProvider`; the
  backend still serves `MockRiskProvider`, and nothing here is wired into the
  live risk pipeline.

## Setup

The ML runtime is deliberately separate from `backend/requirements.txt`; the
demo backend stays free of torch until `MLRiskProvider` lands.

From a clean machine, at the repository root:

```bash
python -m venv ml/.venv
source ml/.venv/bin/activate          # Windows: ml\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r ml/requirements.txt
```

`ml/.venv/` is gitignored and must never be committed.

On Debian/Ubuntu (including WSL) `python -m venv` may fail with
`ensurepip is not available`. Install the stdlib venv package first:

```bash
sudo apt install python3.12-venv
```

**Optional but recommended.** The `torch` pin resolves to the CUDA build on
PyPI, which drags in several GB of NVIDIA wheels this project never uses.
Everything here runs on CPU, so install the small CPU build first and pip will
treat the requirement as already satisfied:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
python -m pip install -r ml/requirements.txt
```

Then fetch and verify the model (~1.3 MB):

```bash
python ml/scripts/setup_aasist.py
python ml/scripts/setup_aasist.py --verify-only   # re-check without downloading
```

`setup_aasist.py` pulls `models/AASIST.py`, `models/weights/AASIST.pth`, and the
upstream LICENCE from pinned commit `a04c9863`, and **refuses to install any
file whose SHA-256 does not match**. Everything lands in `ml/spoof/vendor/`,
which is gitignored. No weights are committed.

```bash
python -m pytest tests/ml
```

## Inference

### Local service

Start the loopback service from the isolated ML environment after installing and
verifying the checkpoint:

```bash
ml/.venv/bin/uvicorn ml.runtime.server:app --host 127.0.0.1 --port 8010
```

`GET /health` is a model-readiness check. It is ready only when the verified
AASIST definition and checkpoint are loaded. `POST /v1/spoof/infer` accepts one
backend-canonical window: mono, 16 kHz, little-endian float32, exactly 64600
samples. The service validates that contract and does not resample, downmix,
pad, truncate, or re-window it.

The service is stateless between requests. It reports an **uncalibrated** raw
spoof-model score and latency; it does not aggregate windows or decide call
risk.

```bash
ml/.venv/bin/python ml/scripts/detect.py --audio path/to/sample.wav
```

Real output on two real corpus samples — one genuine LibriSpeech utterance and
one real Piper-generated file. No synthetic test tones:

```
File:                  ml/data/eval/genuine/librispeech_652-129742-0000.flac
Spoof score:           0.0000   (uncalibrated, 0-1; higher = more synthetic-like)
Bona-fide score:       +4.4868  (raw model output, higher = more human)
Model:                 AASIST/AASIST.pth@ASVspoof2019-LA
Audio duration:        6.03s at 16000 Hz
Preprocessing:         0.5 ms
Inference latency:     364.7 ms
Total:                 365.2 ms
Warnings:
  - truncated 6.03s of audio to the model's 4.04s window

File:                  ml/data/eval/synthetic/tts_023_spk256.wav
Spoof score:           0.9988   (uncalibrated, 0-1; higher = more synthetic-like)
Bona-fide score:       -3.5299  (raw model output, higher = more human)
Model:                 AASIST/AASIST.pth@ASVspoof2019-LA
Audio duration:        5.06s at 16000 Hz
Preprocessing:         2.9 ms
Inference latency:     339.4 ms
Total:                 342.3 ms
Warnings:
  - resampled 22050Hz to 16000Hz
  - truncated 5.06s of audio to the model's 4.04s window

UNCALIBRATED spoof-model score. Not a probability that the audio is fake, not an
identity decision, not a fraud decision, and not a VoxSentinel risk score.
```

### Reading the score

The score is an **uncalibrated softmax output**. A value of `0.90` means "well
above this model's spoof decision boundary on this input", **not** "90% likely
to be fake". It carries no probability semantics and must never be presented to
a user, a judge, or a report as a percentage likelihood. Calibration is future
work.

## Tests

The ML tests live under `tests/ml/` and guard their imports, so the backend
suite skips them cleanly in its own torch-free environment.

```bash
# full ML suite, including real-inference integration tests
python -m pytest tests/ml

# fast tests only; no model and no audio needed
python -m pytest tests/ml -m "not integration"

# backend suite, unchanged and still torch-free
backend/.venv/bin/python -m pytest
```

Integration tests load the real checkpoint and run a real forward pass. The
acceptance assertions run against **real corpus audio** — a genuine LibriSpeech
utterance and a real Piper-generated file, both SHA-256 pinned in the manifest —
not against generated test tones. A single generated-waveform test remains, and
is labelled as plumbing-only; it is not evidence of detection quality.

Tests skip with a clear reason when the model or the evaluation audio is absent.
Nothing is mocked and then reported as working.

The metric tests in `tests/ml/evaluation/` check the confusion matrix, accuracy,
precision, recall, F1, and EER against hand-computed cases, and assert that the
committed summary is reproducible from the committed predictions.

## Evaluation

The evaluation audio is **not committed** — no corpora, no generated speech, no
private recordings. What is committed is everything needed to rebuild it
byte-for-byte and to audit the published numbers:

| File | What it is |
| --- | --- |
| `ml/evaluation/eval_manifest.json` | the 80 samples, with full provenance and a SHA-256 each |
| `ml/evaluation/predictions.csv` | one row per sample: id, ground truth, score, prediction, duration, latency, warnings |
| `ml/evaluation/evaluation_results.json` | the summary, derived from those rows |

### Reproducing it

```bash
# 1. rebuild all 80 samples and verify every SHA-256 (~420 MB of downloads)
python ml/scripts/prepare_eval_set.py --download

# 2. score them and refresh the committed artefacts
python ml/scripts/evaluate.py --per-sample

# verify an existing rebuild without regenerating
python ml/scripts/prepare_eval_set.py --verify-only
```

`prepare_eval_set.py` fetches LibriSpeech dev-clean and the Piper voice, checks
the voice model and config hashes, copies each genuine utterance named in the
manifest, re-synthesises each synthetic sample from its recorded
speaker/text/parameters, and verifies every result. It exits non-zero if
anything fails, so silent drift is not possible.

```
Verifying generator inputs:
  verified voice model: en_US-libritts_r-medium.onnx
  verified voice config: en_US-libritts_r-medium.onnx.json

All 80 samples verified against the manifest.
  LibriSpeech  40/40  strict SHA-256
  Piper        40/40  tolerant canonical PCM (worst 0.000000 dB of 0.01 dB allowed)
  Total        80/80
```

Genuine and synthetic samples are held to different standards, for reasons
measured and documented below.

You can also re-derive the whole summary from the committed predictions alone,
with no model, no audio, and no downloads:

```bash
python ml/scripts/evaluate.py --from-predictions ml/evaluation/predictions.csv
```

That reproduces the confusion matrix and every metric, which is what makes the
published numbers auditable: nothing is hardcoded.

`ml/scripts/build_eval_manifest.py` is the authoring step that produced the
manifest. It is committed so the selection rule is executable rather than
asserted, but it is not part of normal reproduction.

### Sample sources

| Class | Source | Licence |
| --- | --- | --- |
| Genuine (40) | LibriSpeech `dev-clean` (OpenSLR 12), read speech from public-domain LibriVox audiobooks | CC BY 4.0 |
| Synthetic (40) | Piper 1.8.0, voice `en_US-libritts_r-medium` (trained on LibriTTS-R) | voice CC BY 4.0; Piper itself GPL-3.0 |

**Genuine selection rule**, recorded in the manifest and applied by
`build_eval_manifest.py`: sort every `*.flac` under `dev-clean` by POSIX path;
walking that order, take the first utterance per speaker whose duration is at
least 4.5 s; stop at 40 speakers. One utterance per speaker, so all 40 are
distinct voices.

**Synthetic generation**: 12 prompts across 16 voice speakers, text and speaker
advancing at different strides so no pair repeats, with `length_scale` varied
over {0.95, 1.00, 1.05}. Each sample's exact text, speaker id, and synthesis
parameters are in the manifest.

Both classes derive from the same audiobook domain, so the comparison isolates
real-versus-synthetic rather than also changing recording conditions.

Piper is a data-generation tool. Nothing under `ml/spoof/` or `ml/audio/`
imports it, and the detector does not need it at runtime.

### Reproducibility: why Piper is not byte-identical

An independent reconstruction reproduced LibriSpeech 40/40 but every one of the
40 Piper WAVs differed. Root cause, established by measurement:

1. **Piper synthesis is float32 inference in ONNX Runtime, and its result
   depends on the execution plan, not only on the inputs.** Demonstrated on one
   machine with everything else held constant: changing only the ONNX
   graph-optimisation level changes the waveform.

   | optimisation level | synthetic_000 WAV SHA-256 | bytes |
   | --- | --- | --- |
   | `all` | `9d56ab6c…` | 212012 |
   | `extended` | `9d56ab6c…` | 212012 |
   | `basic` | `91da31ac…` | 212012 |
   | `disabled` | `4a1209a8…` | 212012 |

   Execution plans also vary with ONNX Runtime version, build flags, and CPU
   kernel dispatch — which is exactly what differs between machines.

2. **`normalize_audio=True` (Piper's upstream default) couples any local
   difference into every sample.** Piper divides the waveform by its peak, so a
   single-ULP difference in that peak rescales everything: measured, 26,671 of
   105,984 int16 samples shift.

3. **int16 quantisation then turns those sub-LSB differences into different
   bytes**, and SHA-256 is all-or-nothing.

Across the 40 samples and all four optimisation modes: 143,901 int16 samples
differ, by at most 17 LSB — about −66 dBFS, inaudible. The audio is the same
audio. Only the bytes differ.

Piper's `normalize_audio` is left at its upstream default. Changing it would
create a different evaluation dataset.

### The canonical PCM check

Per-sample quantisation does not fix this at any tolerance, because a sample
sitting near a bucket boundary flips buckets however coarse the buckets are.
Measured across all 40 samples and 4 modes:

| canonicalisation | stable? |
| --- | --- |
| drop low k bits, k ≤ 9 (up to ±256 LSB) | no — 40/40 samples unstable |
| drop low 10 bits (±512 LSB, 1.6% of full scale) | no — 37/40 unstable |
| frame-RMS dB, frame 4096, 2 decimals | yes — 0/40 unstable |

Stability is not even monotonic in the bucket size, which is why hash equality
on quantised values is the wrong instrument. The gate is therefore a **numeric
comparison of a coarse energy envelope**: frame RMS in dBFS over 4096-sample
frames (~186 ms), floored at −90 dBFS, compared with a tolerance.

**Tolerance chosen from measurement, not preference:**

| quantity | value |
| --- | --- |
| execution-plan noise floor (40 samples × 6 mode pairs, worst case) | **0.000120 dB** |
| 0.5% amplitude change | 0.0430 dB |
| 1% amplitude change | 0.0861 dB |
| one 186 ms frame zeroed | 77.2417 dB |
| different speaker, same text | 10.1153 dB |
| **`CANONICAL_TOLERANCE_DB`** | **0.01 dB** |

That sits ~83× above the observed noise and ~3.6× below the smallest corruption
tested. Verified against all four optimisation modes:

```
all        Piper 40/40   worst 0.000000 dB
extended   Piper 40/40   worst 0.000000 dB
basic      Piper 40/40   worst 0.000084 dB
disabled   Piper 40/40   worst 0.000097 dB
```

The worst case uses 1% of the budget. Under the previous byte-identical gate,
`basic` and `disabled` failed 40/40.

The envelope is a ~186 ms-resolution energy summary, around 25 numbers per
sample. Speech cannot be reconstructed from it, which is why it is safe to
commit when the audio is not.

### What each class is verified against

| Class | Gate |
| --- | --- |
| LibriSpeech (40) | **strict SHA-256** — these are byte copies of fixed corpus files, so bit equality is a sound expectation |
| Piper (40) | **tolerant canonical PCM** — exact sample rate, channel count and sample count; matching voice model and config hashes; envelope within 0.01 dB |

For Piper samples the full-file SHA-256 stays in the manifest as
**informational only**. It records that the authoring machine reproduces
bit-for-bit; it is not the cross-machine pass/fail gate.

The voice model **and** its config are hash-verified on every preparation run,
not only when `--download` is used, because a different voice or config silently
produces different speech.

`onnxruntime` is pinned explicitly. Piper declares `onnxruntime<2,>=1`, which
would otherwise leave the inference runtime floating.

The manifest records the authoring environment — Python, Piper, ONNX Runtime,
numpy, soundfile, OS/arch, execution providers, and CPU SIMD flags — so an
environment difference is visible rather than inferred.

### Measured results

80 samples, 40 genuine and 40 synthetic, threshold 0.5, generated from
`ml/evaluation/predictions.csv`:

```
Uncalibrated model-score distribution:
  bonafide  n=40   min=0.0000 median=0.0031 max=0.9580 stdev=0.2437
  spoof     n=40   min=0.0873 median=0.8823 max=0.9988 stdev=0.3090

Confusion matrix:
                    predicted spoof     predicted bonafide
  actual spoof                     31                    9
  actual bonafide                   4                   36

Metrics (computed from labels + scores, never hardcoded):
  accuracy   0.8375
  precision  0.8857
  recall     0.7750
  f1         0.8267
  EER        0.1250   at threshold 0.2401
```

### What these numbers are not

This is an **exploratory engineering evaluation**. Read every one of these
before quoting any figure above:

- **80 samples only**, 40 genuine and 40 synthetic.
- **One genuine corpus** (LibriSpeech dev-clean) and **one TTS family**
  (Piper VITS, `en_US-libritts_r`).
- **Not a benchmark.** Not comparable to published ASVspoof results.
- **Not production accuracy.** 83.75% is not "VoxSentinel is 83.75% accurate",
  and must never be presented that way.
- **Not multilingual validation.** English only.
- **Not telephony-channel validation.** Clean studio-derived audio throughout.
- **The score is uncalibrated.** See "Reading the score" above.
- **Threshold 0.5 is provisional**, an arbitrary midpoint, not tuned on a dev set.
- **The EER is specific to this set** and will move on different data.

What the numbers do support, narrowly:

- **The model separates these two classes.** Median genuine 0.0031 against
  median synthetic 0.8823 is real signal, not noise.
- **There is a visible domain gap.** 12.5% EER here against roughly 0.83%
  published by the authors on in-domain ASVspoof 2019 LA eval is the cost of
  scoring a modern multi-speaker VITS voice with a model trained on 2019
  attacks. Nine of forty synthetic samples fell below 0.5.
- **0.5 is the wrong cut point** for this data; EER lands at 0.2401.

The honest next step is evaluation against ASVspoof 2019 LA eval — in-domain and
properly sized — before any performance claim is made.

## Latency

```bash
ml/.venv/bin/python ml/scripts/benchmark_latency.py --audio path/to/sample.flac
```

Measured on a real corpus sample, 3 warm-up plus 20 measured passes:

```
Source:           ml/data/eval/genuine/librispeech_652-129742-0000.flac
Audio duration:   6.03s (model window 4.04s)

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
  preprocessing       0.31      0.29      0.24      0.43      0.43
  inference         326.89    325.99    312.79    348.86    344.76
  total             327.20    326.24    313.03    349.11    345.13

Real-time factor: 12.4x  (model window / median total)
```

Preprocessing is negligible; essentially all the cost is the forward pass.
Scoring a 4.04 s window in a median 326 ms is about 12x faster than real time on
CPU, which leaves headroom for the 0.5 s hop in `audio/chunker.py`.

Caveats: this is one laptop CPU under WSL2, timings move with machine load
(an earlier run on a busier machine measured a 450 ms median), and this measures
the model call only. It says nothing about end-to-end latency in a live call
pipeline, which does not exist yet. The GPU was present but unused.

## Privacy

No raw audio is stored or logged by this subsystem. Downloaded corpora,
generated samples, and model checkpoints all live under gitignored paths and
are never committed.
