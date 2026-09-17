# Speaker verification

Ownership boundary: Rohan. Task ROHAN-003.

> ## What this subsystem does and does not answer
>
> It answers exactly one question: **does this audio come from the speaker who
> was enrolled?**
>
> That is a different question from the one `ml/spoof/` answers. The spoof
> detector asks *is this audio synthetic?*; this asks *is this the same voice?*
> A perfect answer here says nothing about whether the voice is real, and the
> [clone result below](#case-c-a-clone-of-the-enrolled-speaker) is the
> demonstration: a text-to-speech model trained on the enrolled speaker was
> accepted **12 times out of 12**.
>
> The output is a **cosine similarity**, not a probability. Nothing here is
> calibrated, nothing here is a VoxSentinel 0-100 risk score, and nothing here
> is wired into the backend. `MLRiskProvider` does not consume it yet; that
> connection is later JASH work.

## Contents

| Path | Purpose |
| --- | --- |
| `verifier.py` | `SpeakerVerifier` interface, `SpeakerEmbedding` and `SpeakerVerificationResult` contracts, cosine similarity, deterministic aggregation |
| `ecapa.py` | `ECAPASpeakerVerifier` - the real model adapter |
| `../audio/preprocessing.py` | decode, downmix, resample, validate (shared with `ml/spoof/`, unchanged) |
| `../scripts/setup_ecapa.py` | fetch and SHA-256 verify the checkpoint |
| `../scripts/verify_speaker.py` | enrol a speaker and score probes |
| `../scripts/evaluate_speaker.py` | score every trial, write evidence, derive metrics |
| `../scripts/prepare_sv_eval_set.py` | rebuild the evaluation audio from the manifest |
| `../scripts/build_sv_manifest.py` | author the manifest (run once; output committed) |
| `../evaluation/sv_manifest.json` | 37 speakers, 172 samples, 442 trials, with provenance and hashes |
| `../evaluation/sv_predictions.csv` | one row per trial: the evidence behind every published number |
| `../evaluation/sv_results.json` | the summary, derived from those rows |

## Model

**ECAPA-TDNN** - "ECAPA-TDNN: Emphasized Channel Attention, Propagation and
Aggregation in TDNN Based Speaker Verification", Desplanques et al.,
Interspeech 2020. Checkpoint released by the SpeechBrain project.

| | |
| --- | --- |
| Source | <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb> (official SpeechBrain release) |
| Licence | Apache-2.0, for both the toolkit and the checkpoint |
| Pinned revision | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` |
| Checkpoint | `embedding_model.ckpt` at that revision |
| Checkpoint SHA-256 | `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2` |
| Training corpus | VoxCeleb 1 + VoxCeleb 2 development sets |
| Input | raw waveform, mono, **16 kHz**, any length |
| Embedding | **192 dimensions**, float32, L2-normalised on construction |
| Similarity | **cosine** |

Chosen because it is the standard reproducible speaker-verification baseline:
an official release from the toolkit that trained it, an unambiguous
Apache-2.0 licence, a pinned revision with a per-file SHA-256, small enough to
run on CPU, and a 16 kHz mono contract that matches the preprocessing
`ml/audio/` already does for AASIST. WavLM-based verification was the other
candidate and was not chosen: it needs a far larger runtime for a baseline
whose job is to be auditable.

Cosine is not an arbitrary choice of distance. The checkpoint was trained with
additive-angular-margin softmax, which shapes the embedding space so that
**angle** is the speaker distance; using anything else would be scoring the
model with a metric it was not trained for.

Neither the checkpoint nor its config is committed here. `setup_ecapa.py`
downloads all five files from the pinned revision and **refuses to install any
file whose SHA-256 does not match**. Everything lands in
`ml/speaker/vendor/ecapa/`, which is gitignored.

### Preprocessing applied before inference

Identical to the spoof path, and reusing the same module, so a file scored by
both subsystems is converted once the same way:

1. Decode with `libsndfile` (WAV, FLAC, OGG).
2. Downmix to mono by averaging channels.
3. Resample to 16 kHz with a polyphase filter (`scipy.signal.resample_poly`).
4. Reject unusable audio (see below).

The model then computes its own 80-dimensional log-Mel filterbank with
sentence-level mean normalisation, exactly as the released `hyperparams.yaml`
specifies. No amplitude normalisation is applied by us; there is no fixed input
length, because ECAPA pools over whatever duration it is given.

### Input requirements

| | |
| --- | --- |
| Sample rate | any; resampled to 16 kHz |
| Channels | any; downmixed to mono |
| Minimum duration | 1.0 s, measured after resampling |
| Reliable duration | >= 2.0 s for a probe; >= 3.0 s total for an enrolment |
| Format | anything `libsndfile` decodes |

Audio that cannot produce an honest answer is **refused, not scored**:

| Condition | Result |
| --- | --- |
| Silence (peak <= 1e-4) | `SilentAudioError` |
| Shorter than 1.0 s | `AudioTooShortError` |
| Undecodable, empty, NaN/Inf, or integer samples | `MalformedAudioError` |

The dangerous failure mode here is not an error, it is a confident **MISMATCH**
on audio that never had a chance - which reads as "impostor" to anything
downstream. So unusable probes raise instead of returning a verdict, and
`tests/ml/speaker/test_ecapa_integration.py` asserts that for silence,
sub-second audio, empty files, and corrupt files.

Anything done to make audio usable, and anything that weakens confidence, is
reported in `warnings`: resampling, downmixing, very low signal level,
clipping, a probe under 2 s, and an enrolment under 3 s.

## Enrolment

```text
enroll([reference utterances])  ->  SpeakerEmbedding
verify(reference, probe_audio)  ->  SpeakerVerificationResult
```

`enroll` accepts file paths or `(samples, sample_rate)` pairs, one or many.

**Aggregation of multiple references** is the mean of the per-utterance unit
vectors, renormalised. Two details make it deterministic rather than merely
repeatable:

- each utterance is **normalised before averaging**, so a long or loud
  utterance does not outweigh a short or quiet one;
- the mean is **accumulated in float64**. float32 addition is not associative,
  so summing in float32 would let the same two utterances produce two different
  references depending on which was read first. The order-independence test in
  `tests/ml/speaker/test_verifier.py` fails without this.

**Insufficient or invalid enrolment audio** fails loudly. If any enrolment
utterance is refused by the preprocessor, `enroll` raises `EnrollmentError`
rather than quietly building a reference from the rest - a reference silently
built from half the audio the caller supplied is a worse outcome than an error.
Enrolments totalling under 3 s are produced but carry a warning saying the
reference is weak.

There is no database, no user account, and no persistence. That is out of scope
for this task.

### Result contract

`SpeakerVerificationResult` carries:

| Field | |
| --- | --- |
| `similarity_score` | cosine in -1..1, **uncalibrated** |
| `is_match` | `similarity_score >= threshold`; a decision, not a confidence |
| `threshold` | the cut point this decision used |
| `model_id` | model, revision, and training corpus |
| `reference_duration_seconds` / `reference_utterances` | what was enrolled |
| `probe_duration_seconds` | what was scored, after resampling |
| `inference_seconds` / `preprocessing_seconds` | measured, not estimated |
| `warnings` | conversions applied and reasons to distrust the verdict |

The contract validates itself on construction: a score outside -1..1 is
refused, and so is an `is_match` that contradicts its own score and threshold.
A reference enrolled with a different model is refused at `verify`, because
embeddings from different models are not comparable and comparing them anyway
produces a plausible-looking number that means nothing.

## Setup

The ML runtime stays separate from `backend/requirements.txt`; the demo backend
is untouched by this task.

```bash
python -m venv ml/.venv
source ml/.venv/bin/activate          # Windows: ml\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r ml/requirements.txt
```

`ml/.venv/` is gitignored and must never be committed. The CPU-build note in
[`../README.md`](../README.md#setup) applies here too.

Then fetch and verify the checkpoint (~89 MB):

```bash
python ml/scripts/setup_ecapa.py
python ml/scripts/setup_ecapa.py --verify-only   # re-check without downloading
```

## Verification

```bash
ml/.venv/bin/python ml/scripts/verify_speaker.py \
    --reference ml/data/sv-eval/arctic/arctic_slt_a000{1,2,3,4}.wav \
    --probe     ml/data/sv-eval/arctic/arctic_slt_a0011.wav \
                ml/data/sv-eval/arctic/arctic_clb_a0011.wav \
                ml/data/sv-eval/clone/clone_slt_a0011.wav
```

Real output on real corpus audio - one genuine utterance from the enrolled
speaker, one from a different human, and one Piper clone of the enrolled
speaker. No synthetic test tones:

```
Enrolled:              4 utterance(s), 12.82s total
Embedding:             192-d, L2-normalised
Model:                 ECAPA-TDNN/spkrec-ecapa-voxceleb@0f99f2d0
Threshold:             0.55

Probe:                 ml/data/sv-eval/arctic/arctic_slt_a0011.wav
Similarity:            +0.8102   (uncalibrated cosine, -1..1)
Decision:              MATCH  (threshold 0.55)
Reference duration:    12.82s over 4 utterance(s)
Probe duration:        3.06s at 16000 Hz
Preprocessing:         0.1 ms
Inference latency:     216.9 ms
Total:                 217.1 ms

Probe:                 ml/data/sv-eval/arctic/arctic_clb_a0011.wav
Similarity:            +0.2574   (uncalibrated cosine, -1..1)
Decision:              MISMATCH  (threshold 0.55)
Reference duration:    12.82s over 4 utterance(s)
Probe duration:        3.60s at 16000 Hz
Preprocessing:         0.1 ms
Inference latency:     250.2 ms
Total:                 250.4 ms

Probe:                 ml/data/sv-eval/clone/clone_slt_a0011.wav
Similarity:            +0.6215   (uncalibrated cosine, -1..1)
Decision:              MATCH  (threshold 0.55)
Reference duration:    12.82s over 4 utterance(s)
Probe duration:        2.81s at 16000 Hz
Preprocessing:         2.1 ms
Inference latency:     205.6 ms
Total:                 207.8 ms
Warnings:
  - resampled 22050Hz to 16000Hz
```

That third block is the whole point of this document. Read on.

### Reading the score

The score is an **uncalibrated cosine similarity**. A value of `0.80` means
"well above this model's decision boundary on this input", **not** "80% likely
to be the same person". It carries no probability semantics and must never be
presented to a user, a judge, or a report as a percentage likelihood.
Calibration is future work.

It is also **not an identity claim**. It says the probe sounds like the enrolled
voice. Whether that voice is a live human is a different question, and
`ml/spoof/` is the subsystem that asks it.

## Evaluation

The evaluation audio is **not committed** - no corpora, no generated speech, no
private recordings. What is committed is everything needed to rebuild it and to
audit the published numbers:

| File | What it is |
| --- | --- |
| `ml/evaluation/sv_manifest.json` | 37 speakers, 172 samples, 442 trials, with provenance and a SHA-256 each |
| `ml/evaluation/sv_predictions.csv` | one row per trial: ids, relation, similarity, decision, durations, latency, warnings |
| `ml/evaluation/sv_results.json` | the summary, derived from those rows |

### Reproducing it

```bash
# 1. rebuild all 172 samples and verify each one (~420 MB of downloads)
python ml/scripts/prepare_sv_eval_set.py --download

# 2. score every trial and refresh the committed artefacts
python ml/scripts/evaluate_speaker.py

# verify an existing rebuild without regenerating
python ml/scripts/prepare_sv_eval_set.py --verify-only
```

```
Verifying generator inputs:
  verified ARCTIC txt.done.data
  verified ARCTIC COPYING
  verified voice model: en_US-arctic-medium.onnx
  verified voice config: en_US-arctic-medium.onnx.json

All 172 samples verified against the manifest.
  LibriSpeech  124/124  strict SHA-256
  CMU ARCTIC    36/36  strict SHA-256
  Piper clone   12/12  tolerant canonical PCM (worst 0.000000 dB of 0.01 dB allowed)
  Total        172/172
```

You can also re-derive the whole summary from the committed predictions alone,
with no model, no audio, and no downloads:

```bash
python ml/scripts/evaluate_speaker.py --from-predictions ml/evaluation/sv_predictions.csv
```

That reproduces every confusion matrix and every metric, which is what makes
the published numbers auditable: nothing is hardcoded. A test asserts it.

`ml/scripts/build_sv_manifest.py` is the authoring step that produced the
manifest. It is committed so the selection rules are executable rather than
asserted, but it is not part of normal reproduction.

The verification rules and the canonical-PCM tolerance are carried over
unchanged from ROHAN-002; see [`../audio/fingerprint.py`](../audio/fingerprint.py)
for why byte equality is the wrong gate for Piper output.

### Sample sources

| Part | Source | Licence |
| --- | --- | --- |
| LibriSpeech (124 files, 31 speakers) | LibriSpeech `dev-clean` (OpenSLR 12), read speech from public-domain LibriVox audiobooks | CC BY 4.0 |
| CMU ARCTIC (36 files, 6 speakers) | CMU ARCTIC, Carnegie Mellon University (c) 2003 | free for any use, commercial or otherwise |
| Piper clones (12 files) | Piper 1.8.0, voice `en_US-arctic-medium` (trained on CMU ARCTIC) | voice follows CMU ARCTIC; Piper itself GPL-3.0 |

CMU ARCTIC is served over plain HTTP - festvox.org has no HTTPS listener.
Integrity therefore rests on the per-file SHA-256 in the manifest, which
`prepare_sv_eval_set.py` checks on every run, including the prompt file that
decides what every clone says.

Piper is a data-generation tool. Nothing under `ml/speaker/` imports it, and the
verifier does not need it at runtime.

### Selection rules

**LibriSpeech.** For each `dev-clean` speaker, sort chapters by numeric id and
keep those holding at least 2 utterances of at least 4.0 s. A speaker qualifies
only if **two such chapters** exist. Enrol on the first 2 utterances of the
first chapter (sorted by POSIX path); probe with the first 2 of the second.

That cross-chapter requirement is the single most important choice in this
evaluation, and it is why 31 of the 40 dev-clean speakers are used rather than
all 40: the other 9 have only one chapter. Two clips recorded minutes apart in
one session share microphone, room, and channel, and a verifier can score them
as similar **without having learned anything about the voice**. Enrolling and
probing from different chapters removes that shortcut. It also makes the
numbers below worse than they would otherwise be, which is the point.

**CMU ARCTIC.** Speakers `awb, bdl, clb, ksp, rms, slt` - chosen for accent and
sex spread (two female US, two male US, one male Scottish, one male Indian) and
because the Piper `en_US-arctic` voice can synthesise all of them. Enrol on
`arctic_a0001`-`a0004`; probe with `arctic_a0011` and `a0012`.

**Impostors.** Each reference is probed against the speakers at offsets 1-4 in
the sorted speaker list, wrapping around. Every speaker gets the same number of
impostors and no random seed is involved.

**Clones.** For every ARCTIC speaker, both probe prompts are synthesised with
the Piper `en_US-arctic` voice using that speaker's own voice index. The prompt
text is read out of the corpus's own `txt.done.data`, not transcribed by hand,
so a clone provably says exactly what the genuine probe says. ECAPA is
text-independent, so matching the text removes content as a possible
explanation for any difference in score.

### Why CMU ARCTIC is in here at all

Case C needs synthetic speech that **imitates a speaker we can also enrol from
genuine audio**. ROHAN-002's synthetic set cannot supply that: its Piper voice
is `en_US-libritts_r`, whose 904 speakers come from LibriTTS-R train-clean-360,
and the intersection with the 40 LibriSpeech dev-clean speakers is **zero** -
checked, not assumed. Every one of those 40 synthetic samples is an unrelated
synthetic voice, not a clone of anybody enrollable.

Piper's `en_US-arctic` voice is trained on CMU ARCTIC, and CMU ARCTIC publishes
its speakers' genuine recordings as individually downloadable files. So
enrolling `slt` from real ARCTIC recordings and probing with Piper's `slt` is a
genuine clone attack on an enrolled speaker, reproducible from a few megabytes.

### Trial relations

| Relation | n | What it is |
| --- | --- | --- |
| `same_speaker` | 74 | target: another genuine utterance from the enrolled person |
| `different_speaker` | 296 | non-target: genuine speech from a different person |
| `clone` | 12 | attack: TTS trained on the enrolled person |
| `synthetic_non_target` | 60 | attack: TTS trained on a different person |

Only the first two are speaker-verification trials, and only those feed the
error rates. **Clone trials are deliberately excluded from FAR.** FAR is defined
over human impostors; a clone is a different attack, and folding it in would
quietly change what the number means. It is reported separately, in full, below.

## Threshold

**0.55.** Not a round number someone liked; derived by
`ml/scripts/evaluate_speaker.py` and recorded in `sv_results.json`, which a test
asserts `DEFAULT_THRESHOLD` against so the constant cannot drift from its
evidence.

It is fitted on the **LibriSpeech trials only** (310 of them). The CMU ARCTIC
trials and every attack trial are held out, so the clone numbers are measured
against a cut point that was not fitted to them.

The rule has two branches, because one does not cover both cases:

- **Classes overlap.** The equal error rate has a unique operating point; that
  is the threshold.
- **Classes separate completely**, which is what happened here. Then EER is
  zero and *every* cut point between the highest impostor score and the lowest
  genuine score makes identical decisions, so "the EER threshold" is not a
  number - a sweep just returns whichever end of the gap it reached first.
  Sitting on either edge is one awkward trial away from an error. The
  **midpoint of the gap** is used, which is the furthest any cut point can be
  from both classes at once.

Measured on the LibriSpeech trials:

| | |
| --- | --- |
| Highest impostor score | 0.4866 |
| Lowest genuine score | 0.6208 |
| Separating interval | (0.4866, 0.6208] |
| Margin | 0.1343 |
| **Threshold: midpoint, 2 dp** | **0.55** |

## Measured results

442 trials over 37 speakers at threshold 0.55, generated from
`ml/evaluation/sv_predictions.csv`.

### Speaker-verification trials

```
All trials                       (74 positive pairs, 296 negative pairs)
                        predicted match   predicted mismatch
  actual same speaker                74                    0
  actual different                    0                  296
  accuracy 1.0000   precision 1.0000   recall 1.0000   f1 1.0000
  FAR      0.0000   FRR       0.0000   EER    0.0000 at 0.6208
  same_speaker       n=74   min=+0.6208 median=+0.7872 max=+0.8829 stdev=0.0609
  different_speaker  n=296  min=-0.2038 median=+0.1197 max=+0.4866 stdev=0.1058

Corpus: librispeech              (62 positive pairs, 248 negative pairs)
  accuracy 1.0000   precision 1.0000   recall 1.0000   f1 1.0000
  FAR      0.0000   FRR       0.0000   EER    0.0000 at 0.6208
  same_speaker       n=62   min=+0.6208 median=+0.7835 max=+0.8829 stdev=0.0639
  different_speaker  n=248  min=-0.2038 median=+0.1216 max=+0.4866 stdev=0.1084

Corpus: cmu_arctic               (12 positive pairs, 48 negative pairs)
  accuracy 1.0000   precision 1.0000   recall 1.0000   f1 1.0000
  FAR      0.0000   FRR       0.0000   EER    0.0000 at 0.7223
  same_speaker       n=12   min=+0.7223 median=+0.8042 max=+0.8587 stdev=0.0353
  different_speaker  n=48   min=-0.0776 median=+0.0963 max=+0.3318 stdev=0.0901
```

### Case A: same human speaker

74 of 74 accepted. The lowest genuine score anywhere is **+0.6208**; the median
is **+0.7872**. Enrolment and probe never share a recording session.

### Case B: different human speaker

296 of 296 rejected. The highest impostor score anywhere is **+0.4866**; the
median is **+0.1197**. No human impostor came within 0.13 of any genuine
speaker.

### Case C: a clone of the enrolled speaker

```
clone                  n=12   accepted 12/12 (100.0%)  min=+0.6041 median=+0.6975 max=+0.8243
synthetic_non_target   n=60   accepted  0/60 (  0.0%)  min=-0.1071 median=+0.1118 max=+0.3178
```

**Every clone was accepted.** Not one of the 12 fell below the threshold, and
the lowest clone score (+0.6041) is **higher than every one of the 296 human
impostor scores** (max +0.4866). To this verifier a Piper clone of the enrolled
speaker is not a borderline case - it sits inside the genuine distribution.

Put the three side by side:

| Probe | n | median | range |
| --- | --- | --- | --- |
| Genuine, same speaker | 74 | +0.7872 | +0.6208 .. +0.8829 |
| **Clone of the enrolled speaker** | **12** | **+0.6975** | **+0.6041 .. +0.8243** |
| Synthetic, different speaker | 60 | +0.1118 | -0.1071 .. +0.3178 |
| Genuine, different speaker | 296 | +0.1197 | -0.2038 .. +0.4866 |

The contrast between rows 2 and 3 is the finding. Synthetic speech is **not**
inherently far from the enrolled speaker: TTS of a *different* speaker scores
like a human impostor, while TTS of the *enrolled* speaker scores like the
speaker. The verifier is measuring voice identity, and the clone genuinely has
the enrolled speaker's voice identity.

**No threshold fixes this.** Raising the cut point to reject the clones would
have to exceed +0.8243, which would reject most genuine speakers too. Speaker
verification alone cannot answer "is this a clone", and it is not a defect that
it does not: it is measuring voice identity, and the clone has the enrolled
speaker's voice identity.

There is no honest threshold change to make here. There is a system design
conclusion: **speaker match must never be treated as proof of a live human.**

#### And the spoof detector does not rescue it on this audio

The tempting conclusion is "speaker verification misses clones, but `ml/spoof/`
catches them, so the pair is fine". On *this* corpus that is not true, and it
would be dishonest to leave the inference standing. An ad-hoc cross-check -
not a committed metric, and outside this task's scope to fix:

```bash
ml/.venv/bin/python - <<'EOF'
import glob
from ml.spoof.aasist import AASISTSpoofDetector
d = AASISTSpoofDetector()
for label, pattern in (("clone", "ml/data/sv-eval/clone/*.wav"),
                       ("genuine arctic", "ml/data/sv-eval/arctic/*_a001*.wav")):
    s = sorted(d.score_file(f).synthetic_probability for f in sorted(glob.glob(pattern)))
    print(label, len(s), f"median={s[len(s)//2]:.4f}", f"above 0.5: {sum(v >= 0.5 for v in s)}/{len(s)}")
EOF
```

```
clone           12 median=0.6171  above 0.5: 7/12
genuine arctic  12 median=0.6127  above 0.5: 6/12
```

AASIST flags 7 of 12 clones - and 6 of 12 **genuine** CMU ARCTIC recordings.
The two distributions are on top of each other: on this audio the spoof
detector is close to uninformative, which is the ROHAN-002 domain gap showing
up at full strength. CMU ARCTIC was recorded in 2003 on different equipment and
is far outside ASVspoof 2019 LA's training domain.

So the honest statement is narrower than "use both signals": **on ARCTIC audio,
neither subsystem catches this clone.** Fusion is necessary but not sufficient,
and closing this gap needs an in-domain spoof evaluation, not a threshold
change. That work belongs to the spoof subsystem, not here.

### Latency

Measured during the evaluation run itself, 172 real utterance embeddings, on
this machine:

```
Hardware: 13th Gen Intel(R) Core(TM) i5-13420H, torch 2.9.1+cpu, WSL2, CPU only

preprocessing  median    0.37 ms
inference      median  442.62 ms   p95 1031.66 ms   min 165.67   max 1437.89
inference      median   63.75 ms per second of audio  (15.7x real time)
```

ECAPA has no fixed window, so its cost scales with utterance length, and these
utterances run from about 2.3 s to 23 s. The spread above is therefore mostly
duration, not jitter - **63.75 ms per second of audio** is the comparable
figure. Preprocessing is negligible; essentially all the cost is the forward
pass.

Caveats: one laptop CPU under WSL2, timings move with machine load, model load
is excluded, and this measures the model call only. It says nothing about
end-to-end latency in a live call pipeline, which does not exist yet.

## What these numbers are not

This is an **exploratory engineering evaluation**. Read every one of these
before quoting any figure above:

- **37 speakers and 442 trials.** 31 LibriSpeech and 6 CMU ARCTIC.
- **Only 6 speakers and 12 trials behind the clone result.** It is a
  demonstration, not a rate. "100% of clones accepted" means 12 of 12.
- **Perfect accuracy here is a statement about the data, not the system.** Two
  clean read-speech corpora with well-separated speakers is an easy task. Real
  call audio will overlap, the error rates will not be zero, and the threshold
  will need refitting on data that looks like production.
- **Not a benchmark.** Not comparable to published VoxCeleb EER figures, which
  are measured on VoxCeleb-O with thousands of trials and harder audio.
- **Not production accuracy.** 100% is not "VoxSentinel is 100% accurate", and
  must never be presented that way.
- **Not telephony-channel validation.** Clean studio-derived audio throughout;
  no PSTN/VoIP codec, no mu-law/A-law, no packet loss, no background noise.
- **English only.** No multilingual claim is made or supported.
- **One clone family** (Piper VITS, `en_US-arctic`). Newer zero-shot cloning
  systems are untested, and a system that clones from seconds of reference audio
  is a different and probably stronger attack.
- **The score is uncalibrated.** See "Reading the score" above.
- **The threshold is fitted, not validated.** It comes from the LibriSpeech
  trials and has no held-out validation set.
- **No cross-window aggregation.** Combining per-window similarities across a
  call is undefined; `ml/audio/chunker.py` produces the windows but nothing here
  consumes them.

What the numbers do support, narrowly:

- **The model separates these speakers cleanly.** A 0.13 margin between the
  lowest genuine and highest impostor score, over 310 LibriSpeech trials with
  no session overlap, is real signal.
- **Cross-session enrolment works.** The result is not an artefact of scoring
  two clips of one recording.
- **A cloned voice defeats speaker verification.** 12 of 12, every one above
  every human impostor. This is the load-bearing result for VoxSentinel's
  design, and it is why the spoof score and the speaker score have to be
  separate inputs to a risk decision rather than one number.

The honest next steps, in order: evaluation on telephony-channel audio,
evaluation against a zero-shot cloning system, a held-out validation set large
enough to fit a threshold that survives contact with real calls, and - on the
spoof side, not here - an in-domain spoof evaluation, since the cross-check
above shows AASIST is close to uninformative on CMU ARCTIC.

## Tests

```bash
# full speaker suite, including real-inference integration tests
python -m pytest tests/ml/speaker tests/ml/evaluation/test_speaker_metrics.py

# fast tests only; no model and no audio needed
python -m pytest tests/ml -m "not integration"

# backend suite, unchanged and still torch-free
backend/.venv/bin/python -m pytest
```

| File | What it guards |
| --- | --- |
| `tests/ml/speaker/test_verifier.py` | contracts, similarity range and maths, deterministic and order-independent aggregation, enrolment failures, threshold logic, silence/short/malformed probes - all model-free |
| `tests/ml/speaker/test_ecapa_integration.py` | real checkpoint, real forward passes, real corpus audio: embedding shape and norm, determinism, cases A/B/C, stereo, resampling, FLAC and WAV, short audio, refusal paths, checkpoint tamper detection |
| `tests/ml/evaluation/test_speaker_metrics.py` | confusion matrix, FAR, FRR, EER and the threshold rule against hand-computed cases; that attacks never leak into FAR; that the committed summary re-derives from the committed predictions; that enrolment and probe never share a chapter |

Integration tests load the real checkpoint and run real forward passes against
real corpus audio, all SHA-256 pinned in the manifest. They skip with a clear
reason when the model or the audio is absent. Nothing is mocked and then
reported as working.

## Known limitations

Carried forward deliberately. None of these is addressed in this task.

**Channel and domain**

- **No PSTN/VoIP codec degradation.** VoxSentinel's eventual input is a phone
  call; every corpus here is clean and wideband.
- **No noisy-channel evaluation.** No packet loss, jitter, or background noise.
- **VoxCeleb domain gap.** ECAPA was trained on interview and celebrity-video
  speech. Read speech is easier; telephony is harder and untested.
- **English only.**

**Model and method**

- **No calibration.** The output is a cosine, not a likelihood ratio, and no
  score normalisation (s-norm, as-norm) is applied.
- **Fitted threshold, no validation set.**
- **No cross-window aggregation** over a call.
- **No speaker diarization.** A probe is assumed to contain one speaker.
- **Cloned voices are accepted.** Measured above, by design of the model class,
  and not fixable inside this subsystem.

**Integration**

- **Nothing is wired into the backend.** There is no `MLRiskProvider` consumer,
  no API change, no WebSocket change, and no frontend change. `speaker_match_score`
  and `speaker_mismatch_score` are later JASH work.

## Privacy

No raw audio is stored or logged by this subsystem. Downloaded corpora,
generated clones, and model checkpoints all live under gitignored paths and are
never committed.
