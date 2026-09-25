# VoxSentinel Final Technical Report

## 1. Executive Summary

VoxSentinel is a frozen Smart India Hackathon prototype for real-time detection and prevention of voice-cloning impersonation attacks. It combines uncalibrated AASIST spoof evidence with uncalibrated ECAPA expected-speaker similarity and converts the two independent evidence dimensions into deterministic operational actions.

## 2. Problem Statement

SIH26104 addresses voice-cloning impersonation. Speaker familiarity alone is insufficient: a clone of an enrolled speaker can preserve speaker characteristics while remaining synthetic.

## 3. Design Goals

The implemented goals are real-time browser ingestion, privacy-preserving ephemeral audio processing, explainable evidence, bounded resources, honest uncertainty, and operational intervention.

## 4. System Architecture

The verified path is:

`getUserMedia → AudioContext → AudioWorklet → VXAF binary transport → backend audio WebSocket → StreamingAudioCanonicalizer → mono 16 kHz float32 → bounded call-scoped scheduling → ML service → AASIST + ECAPA → deterministic temporal fusion → risk/action → risk WebSocket → React forensic dashboard`.

The backend remains Torch-free. AASIST and ECAPA execute in the separate loopback ML service.

## 5. Browser Audio Capture

Microphone permission is requested only after explicit LIVE analysis start. The AudioWorklet accumulates render-quantum samples; transport owns VXAF sequence/header generation and backpressure. Stop and Reset close the worklet, AudioContext, sockets, and media tracks.

## 6. VXAF Protocol

VXAF uses a 32-byte little-endian header: magic, version, header length, flags, sequence, first source sample frame, sample count, and payload length. Payload samples are float32 little-endian. The backend validates channel/rate/count/payload consistency, finite values, overlap/replay, and sequence gaps.

## 7. Canonical Audio Pipeline

The backend downmixes before resampling and maintains one stateful `soxr` stream per producer. Output is mono 16 kHz float32. The source-segment ledger preserves conservative source/canonical correlation metadata.

## 8. AASIST

The runtime uses `AASIST/AASIST.pth@ASVspoof2019-LA`, checkpoint SHA-256 `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0`. It consumes exactly 64,600 samples at 16 kHz, approximately 4.0375 seconds. The result is **uncalibrated spoof evidence**, not a probability.

The streaming scheduler uses overlapping windows with bounded pending work. Missing or unavailable model evidence is not silently replaced with demo data.

## 9. ECAPA

The runtime uses SpeechBrain `spkrec-ecapa-voxceleb`, revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, 16 kHz audio, and 192-dimensional normalized embeddings. Verification uses cosine similarity and the 0.55 engineering threshold. Profiles retain derived embeddings and provenance, not live reference PCM. ECAPA output is **uncalibrated speaker similarity**, not a probability.

## 10. Why Dual Evidence

The controlled ECAPA evaluation contained 74 genuine, 296 human-impostor, and 12 target-voice clone trials. The 12 clone samples crossed the speaker-similarity threshold, demonstrating why speaker identity alone cannot establish authenticity.

## 11. Temporal Fusion

| AASIST condition | ECAPA condition | Fused state/action |
|---|---|---|
| Low/normal | Consistent | `NORMAL` / `MONITOR` |
| Low/normal | Inconsistent | `IDENTITY_REVIEW` / `VERIFY_IDENTITY` |
| Elevated | Consistent | `AUTHENTICITY_REVIEW` / `REQUIRE_CALLBACK` |
| Elevated | Inconsistent | `HIGH_RISK_REVIEW` / `HOLD_SENSITIVE_ACTION` |
| Available spoof, missing speaker | Unavailable | Authenticity-driven review; speaker remains not evaluated |
| Missing spoof, available speaker | Unavailable | Identity review/degraded policy; spoof remains not evaluated |
| Both unavailable/indeterminate | Unavailable | `INDETERMINATE` / conservative handling |

The implementation keeps evidence dimensions separate, uses source/window correlation and bounded staleness, persists promotions/demotions deterministically, clears histories on Reset/new call, and rejects out-of-order speaker evidence from replacing newer aligned evidence.

## 12. Real-Time Performance

Fresh controlled LIVE timing produced the first fused event 4,572.2 ms after MIC STREAMING. The nominal AASIST audio window is 4.0375 seconds. Fifty observations produced a 513.5 ms mean update interval, 502.8 ms p50, and 571.9 ms p95. Separate canonical-availability, AASIST-only, and ECAPA-only timestamps are not exposed by the current acceptance snapshot.

## 13. Frontend

LIVE and DEMO remain distinct. LIVE presents model-backed AASIST/ECAPA evidence, fusion state, operational action, privacy status, and honest N/A values for Replay, Prosody, and Context. DEMO retains deterministic scenario playback.

## 14. Privacy and Security

Live raw audio is not persisted, logged, downloaded, or stored in a database. Reference PCM is converted to an embedding and discarded by the profile flow. Acceptance telemetry is bounded metadata and token-gated. Model files, secrets, and audio are not committed.

## 15. Failure and Degraded Modes

The system explicitly handles ML unavailability, model mismatch, missing reference, insufficient audio, disconnects, duplicate producers, reset/stop races, and stale evidence. Missing evidence is neither treated as safe evidence nor as confirmed attack, and LIVE does not silently fall back to DEMO values.

## 16. Evaluation

AASIST: N=80 exploratory controlled samples, 40 LibriSpeech bona fide and 40 Piper synthetic; accuracy 83.75%, F1 0.8267, EER 0.1250. Training-overlap status is not established.

ECAPA: 74 genuine and 296 impostor trials yielded FAR/FRR/EER of 0% on the small controlled human set at the engineering threshold; 12/12 target clones passed speaker similarity. These are controlled prototype observations, not population benchmarks.

## 17. Deployment Architecture

The supported deployment is local loopback: ML on port 8010, backend on port 8000, and Vite frontend on port 5173. No public deployment configuration is present.

## 18. Deployment Runbook

See `docs/deployment/VoxSentinel_Deployment_Runbook.md`.

## 19. Testing

Fresh verification: 147 backend tests passed; 220 ML tests passed with 14 prepared-evaluation skips; 203 ML fast tests passed; 65 frontend tests passed; lint and build passed; deterministic DEMO E2E passed; controlled browser LIVE runtime and 60-second stress path passed. One stale E2E assertion needs rounding-aware maintenance.

## 20. Known Limitations

- No broad Indian-accent validation.
- No multilingual validation.
- No telephony codec benchmark.
- No replay model.
- No prosody model.
- No context model.
- Small exploratory AASIST dataset.
- Controlled ECAPA evaluation set.
- Engineering thresholds are not population-calibrated.
- Blockchain is not implemented.

## 21. Blockchain / Audit Position

Blockchain is not required for inference. Current capability is an evidence pipeline with audit-ready metadata. Optional future tamper-evident anchoring could independently prove that incident evidence and model-version records were not altered after the decision.

## 22. SIH Demo Procedure

Start ML, backend in ML mode, and frontend. Open the dashboard, select LIVE, start analysis, grant microphone permission, run the controlled or physical microphone flow, show AASIST and ECAPA evidence, explain the fused action, show unsupported signals as N/A, and Stop. DEMO mode remains available for deterministic policy demonstration and must be labeled simulated.

## 23. Future Work

Future work is limited to broader multilingual/telephony evaluation, replay and prosody models, context intelligence, calibration, optional external anchoring, and production-scale deployment. These are not required core engineering tasks for this frozen SIH build.

## 24. Conclusion

VoxSentinel provides a reproducible, privacy-preserving browser-to-model-to-action prototype that treats speaker identity and voice authenticity as separate evidence questions and exposes their operational consequence honestly.
