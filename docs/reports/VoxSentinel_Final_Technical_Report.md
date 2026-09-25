# VoxSentinel Final Technical Report

## Executive summary

VoxSentinel is a frozen SIH26104 prototype for detecting voice-cloning impersonation during a sensitive call. The implementation keeps voice authenticity and speaker identity as separate evidence streams, combines available evidence with deterministic temporal policy, and exposes a judge-facing action with explicit degraded states.

This report describes the repository as stabilized on the deployment branch. Public Vercel/Render execution is **NOT VALIDATED YET** because this workspace has no authenticated Vercel or Render deployment client and no hosted service URLs. The deterministic DEMO path is a **SIMULATED POLICY DEMONSTRATION**, not model evidence.

## Problem and design goals

SIH26104 concerns impersonation through AI-generated or cloned speech. Speaker similarity alone cannot establish authenticity: a clone of an enrolled speaker can match the speaker model. The design goals are real-time browser capture, bounded processing, explainable dual evidence, privacy-preserving ephemeral audio, deterministic policy, and honest uncertainty.

## Architecture

```text
Browser getUserMedia
  -> AudioContext / AudioWorklet
  -> VXAF float32 transport
  -> backend audio WebSocket
  -> StreamingAudioCanonicalizer
  -> mono 16 kHz float32
  -> exact 64,600-sample AASIST windows
  -> ECAPA speaker probes when a reference exists
  -> deterministic temporal fusion
  -> risk/action
  -> backend risk WebSocket
  -> React console
```

The backend is Torch-free. The ML service owns AASIST and SpeechBrain ECAPA. DEMO mode uses deterministic scenario fixtures and never silently substitutes for LIVE mode.

## Browser and audio pipeline

LIVE requests microphone permission only after an explicit Start action. The AudioWorklet feeds VXAF frames to one bounded audio producer. The backend validates sequence, sample format, finite values, and ownership; downmixes and resamples statefully to mono 16 kHz; then schedules AASIST windows of 64,600 samples (4.0375 seconds) with bounded pending work. Stop and Reset release tracks, AudioContext, worklet, WebSockets, and call-scoped queues.

## Models and evidence

- AASIST uses the pinned official checkpoint with required SHA-256 `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0`. Its output is uncalibrated spoof evidence, not a probability of fraud.
- ECAPA uses SpeechBrain `spkrec-ecapa-voxceleb` revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, 192-dimensional normalized embeddings, cosine similarity, and the 0.55 engineering threshold.
- A missing model, reference, or sufficient audio is represented as unavailable/indeterminate. It is never mapped to safe or attack evidence.
- Replay, prosody, and context signals remain N/A where unsupported.

## Privacy and security

Raw live PCM is not persisted, logged, or committed. Reference PCM is converted to an embedding and discarded by the profile flow. Model assets are fetched and checksum-verified at deployment build time and are gitignored. Production CORS is explicit through `VOXSENTINEL_CORS_ORIGINS`; frontend production uses HTTPS and WSS origins from environment variables.

## Evaluation

The committed evaluation artifacts record controlled exploratory tests, not production benchmarks:

- AASIST: N=80, 40 bona fide and 40 Piper synthetic, accuracy 83.75%, F1 0.8267, EER 0.125.
- ECAPA ordinary human verification: 74 genuine and 296 human impostor trials, threshold 0.55, FAR 0%, FRR 0%, EER 0% on that controlled set only.
- Separate adversarial extension: 72 synthetic/attack trials, including 12 target-voice clones; 12/12 crossed the ECAPA threshold. Clone trials are not counted in FAR.

## Deployment

`render.yaml` defines a Render ML service and backend. ML assets are retrieved by checksum-verifying setup scripts; the ML health endpoint can require both AASIST and ECAPA with `VOXSENTINEL_REQUIRE_SPEAKER_MODEL=true`. The backend binds `0.0.0.0:$PORT`, uses `VOXSENTINEL_ML_SERVICE_URL`, and allows only configured origins. `vercel.json` builds `frontend` and rewrites SPA routes to `index.html`.

## Local measurement host

The recorded local inference host was an Intel 12th Gen Core i5-12450H with 15.7 GB RAM, Intel UHD Graphics, and Windows 11 Home Single Language 64-bit. These host facts describe the controlled local run only; hosted compute and hosted latency remain unvalidated.

## Limitations

There is no broad Indian-accent validation, multilingual validation, telecom codec benchmark, replay model, prosody model, context model, population calibration, or blockchain implementation. The AASIST set is small and exploratory; the ECAPA set is controlled clean speech. A physical microphone, hosted runtime, and hosted latency are **NOT VALIDATED YET** in this workspace.

## Judge procedure

Primary path, only after hosted acceptance is verified: open the Vercel URL, select LIVE, Start, grant microphone permission, wait for STREAMING/audio_ready, show real AASIST and ECAPA evidence, explain fusion/action, Stop, Reset, and repeat. Local fallback is the same flow against verified local services. Final fallback is DEMO, explicitly introduced as “SIMULATED POLICY DEMONSTRATION; these values are deterministic policy fixtures, not live model output.”
