# JASH-006: Final Intelligence Integration Design

## Existing-state audit

JASH-005 provides browser getUserMedia capture, AudioWorklet batching, VXAF transport, a call-scoped stateful soxr canonicalizer, exact 64,600-sample AASIST windows with an 8,000-sample hop, one bounded AsyncCallSession, MLRiskProvider, risk WebSocket events, bounded metadata telemetry, and transactional cleanup. The backend remains Torch-free.

ROHAN-003 provides SpeechBrain ECAPA-TDNN at `ml/speaker/ecapa.py`, revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, mono 16 kHz input, 192-dimensional L2-normalized embeddings, cosine similarity, and exploratory threshold 0.55.

## Goals and non-goals

Add explicit speaker enrollment/reference handling, real ECAPA verification on live canonical audio, deterministic temporal fusion with AASIST, degraded evidence states, and truthful LIVE UI presentation. Do not add Twilio/PSTN, replay/prosody detection, database persistence, authentication, calibration, retraining, UI redesign, or JASH-007.

## Threat model

Distinguish expected human genuine speech, a different human, an enrolled-speaker clone, a clone of another speaker, noisy audio, missing reference, and unavailable detectors. High ECAPA similarity is voice-consistency evidence only; it never proves authenticity. AASIST spoof evidence and ECAPA similarity are separate uncalibrated dimensions.

## Enrollment/reference architecture

The backend owns an in-memory SpeakerProfileRegistry for this SIH build. A profile contains speaker_profile_id, expected_speaker_id, a verified reference embedding, model id/revision, dimension, reference duration, and provenance. Enrollment accepts a complete WAV/FLAC container through a narrow local API, sends canonical audio to the ML service, stores only the embedding, and discards source PCM. Profiles are process-local and isolated by id.

## ECAPA runtime architecture

The existing loopback ML service remains the only Torch boundary. It loads verified AASIST and ECAPA runtimes, each with a bounded concurrency gate of one model forward by default. It exposes health readiness, `/v1/speaker/embed`, and `/v1/speaker/verify`. Requests contain canonical mono 16 kHz float32 audio; the service validates format and performs tensor/model execution, not browser framing or call policy. Backend HTTP calls are asynchronous and timeouts become explicit unavailable evidence.

## Audio/window scheduling

AASIST keeps its exact 64,600/8,000 schedule. Speaker probes never use browser transport frames. A profile-enabled call accumulates canonical audio and emits a probe after `SPEAKER_MIN_PROBE_SAMPLES = 32000` (2 seconds), then every `SPEAKER_HOP_SAMPLES = 16000` (1 second), retaining at most 64,000 samples. Only one speaker request is in flight; newest complete data is retained. Before two seconds the state is INSUFFICIENT_AUDIO.

## Speaker evidence contract

Availability is EVALUATED, NOT_EVALUATED, INSUFFICIENT_AUDIO, NO_REFERENCE, or MODEL_UNAVAILABLE. Evaluated evidence includes expected speaker/profile ids, uncalibrated cosine, threshold, state CONSISTENT/INCONSISTENT/INDETERMINATE, model id/revision, window sequence, correlation, and latency. Cosine is never a probability or percentage.

## Fusion matrix

| Spoof state | Speaker state | Result | Action |
|---|---|---|---|
| NORMAL | CONSISTENT | NORMAL | MONITOR |
| NORMAL | INCONSISTENT | IDENTITY_REVIEW | VERIFY_IDENTITY |
| ELEVATED | CONSISTENT | AUTHENTICITY_REVIEW | REQUIRE_CALLBACK |
| ELEVATED | INCONSISTENT | HIGH_RISK_REVIEW | HOLD_SENSITIVE_ACTION |
| ELEVATED | unavailable | AUTHENTICITY_REVIEW | REQUIRE_CALLBACK |
| NORMAL | unavailable | NORMAL | MONITOR |
| unavailable | INCONSISTENT | IDENTITY_REVIEW | VERIFY_IDENTITY |
| unavailable | unavailable | INDETERMINATE | MONITOR |

Missing evidence is neither safe evidence nor confirmed attack. The existing mock table is untouched. Fusion is isolated behind a policy component.

## Final risk state machine

Real states are NORMAL, IDENTITY_REVIEW, AUTHENTICITY_REVIEW, HIGH_RISK_REVIEW, and INDETERMINATE. Promotion requires two consecutive supporting observations; demotion requires two lower-state observations. Operational labels are constants only: 20/LOW/MONITOR for NORMAL, 50/HIGH/VERIFY_IDENTITY for IDENTITY_REVIEW, 70/HIGH/REQUIRE_CALLBACK for AUTHENTICITY_REVIEW, 70/HIGH/HOLD_SENSITIVE_ACTION for HIGH_RISK_REVIEW, and 20/LOW/MONITOR for INDETERMINATE. Real ML never emits CRITICAL, REQUIRE_SUPERVISOR, or BLOCK_ACTION.

## Temporal logic

AASIST retains its last-five median and 0.5/0.45 persistence policy. Speaker uses threshold 0.55 and two-observation consistency persistence. Single observations cannot immediately change fused action. Stop, Reset, and new calls clear histories.

## Failure/degraded behavior

Missing profile is NO_REFERENCE; short audio is INSUFFICIENT_AUDIO; ECAPA timeout/load/error is MODEL_UNAVAILABLE. AASIST failure remains explicit ML unavailability. No failure falls back to MockRiskProvider or fabricated values. Degraded events may continue when one detector remains healthy; fully unavailable ML reports an error/terminated live session.

## Resource/concurrency model

One producer, canonicalizer, AASIST scheduler, speaker scheduler, runtime session, and risk stream exist per call. AASIST and ECAPA forwards are each gated at one concurrent forward. Buffers and histories are bounded. Duplicate producers remain isolated and rejected.

## Privacy model

Live PCM is ephemeral and never written to logs, browser storage, files, or a database. Enrollment source audio is discarded after embedding. Embeddings are sensitive process-local metadata and are never committed. Telemetry is bounded evidence metadata only.

## Backend/ML APIs

Backend-local development endpoints are `POST /api/v1/speaker-profiles` and `GET /api/v1/speaker-profiles/{id}`. ML endpoints are `POST /v1/speaker/embed` and `POST /v1/speaker/verify`; both validate mono 16 kHz float32 finite arrays and return model identity, evidence, availability, and measured latency.

## Frontend integration

Visual design and mode selection remain unchanged. LIVE shows real voice authenticity evidence, speaker consistency, and truthful N/A for replay/prosody/context. Copy says uncalibrated model evidence or uncalibrated speaker similarity, never probability.

## Telemetry

Bounded telemetry adds profile id, expected speaker, speaker window count, cosine, threshold, availability/state, fusion state/action, ECAPA latency, queue/drop counts, and cleanup state. It stores no audio and clears on Reset/disposal.

## Test strategy and acceptance

Fast tests cover contracts, profile isolation, scheduling, persistence, fusion, degraded states, and unchanged mock behavior. ML tests cover real ECAPA loading, normalized 192-d embeddings, same/different trials, and the documented clone limitation when fixtures exist. Browser tests use controlled microphone audio through getUserMedia and VXAF, never direct WAV POST. Acceptance covers expected speaker, different speaker, no reference, short audio, ECAPA/AASIST unavailable, Reset, duplicate producer, and mock. A clone fixture is used only if available; otherwise record CONTROLLED_CLONE_FIXTURE_NOT_AVAILABLE.

## Physical acceptance and SIH flow

Physical microphone acceptance is attempted only with honest hardware access; otherwise record PENDING_HUMAN. The judge flow is LIVE, select/create local profile, START, speak for multiple windows, show AASIST plus speaker evidence and fused action, then Stop. Future Twilio, replay, prosody, calibration, and production-scale work is outside JASH-006, the final core engineering task.
