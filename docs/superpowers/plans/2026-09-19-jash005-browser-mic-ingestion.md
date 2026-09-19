# JASH-005 Browser Microphone Ingestion Implementation Plan

> **Required subskill:** Use `superpowers:test-driven-development` for every behavior task and `superpowers:verification-before-completion` before the PR is opened.

**Goal:** Add truthful, bounded browser-microphone ingestion to the existing VoxSentinel live ML pipeline without changing JASH-004 policy, the mock demo, or the visual design.

**Architecture:** A browser `AudioWorklet` emits 1024-sample transport frames over a dedicated versioned binary audio WebSocket. The backend validates and correlates those frames, owns one stateful `StreamingAudioCanonicalizer` per producer, and feeds the existing shared `AsyncCallSession`/AASIST scheduler. The existing risk WebSocket remains separate.

**Tech stack:** Existing React/TypeScript/Vite frontend; existing FastAPI backend; Python `soxr==1.1.0` for stateful backend resampling; existing AASIST/ML service and risk provider; Vitest/RTL/Playwright and pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-jash005-browser-mic-ingestion-design.md` at approved commit `011e1196538fdb91b6440c517d3bf137f1d99376`.

**Global constraints:** No production edits before the implementation branch is created from the approved documentation state. No ECAPA, speaker enrollment/fusion, Twilio/PSTN, storage, calibration, threshold changes, reconnect, or UI redesign. Mock mode must remain microphone/ML independent. Raw audio is ephemeral and never logged or persisted.

## Repository map and shared interfaces

The audited repository paths used by this plan are `backend/app/api/calls.py`, `backend/app/services/`, `backend/app/state/`, `backend/tests/`, `frontend/src/`, `frontend/e2e/`, and `frontend/src/**/*.test.ts(x)`. New files use these paths; no alternate test tree is introduced.

Shared transport constants:

```text
VXAF_MAGIC = "VXAF"
AUDIO_PROTOCOL_VERSION = 1
AUDIO_HEADER_LENGTH = 32
AUDIO_FLAGS = 0
MIN_SOURCE_SAMPLE_RATE = 8000
MAX_SOURCE_SAMPLE_RATE = 96000
SUPPORTED_CHANNELS = {1, 2}
MAX_TRANSPORT_SAMPLES = 1024
AUDIO_WS_HIGH_WATERMARK_BYTES = 262144
AUDIO_WS_LOW_WATERMARK_BYTES = 65536
```

`audio_start` is JSON with protocol version, integer `sample_rate`, `channels`, and `sample_format: "float32le"`. Each binary frame uses the exact 32-byte little-endian header from the spec. `frame_sequence` starts at 1 and advances for every produced frame, including dropped frames. `first_sample_frame` identifies the first Web Audio sample index. For every accepted frame, `source_frame_end = first_sample_frame + sample_count_per_channel - 1`; the next accepted frame must start strictly after the previous end. Contiguous ranges are accepted, forward gaps are accepted and counted, overlap/replay is rejected, and uint64 overflow is rejected.

## Task 1 — Implementation branch, clean baseline, and dependency gate

**Commit:** `chore: establish JASH-005 implementation baseline`

**Files:** No production files. Create only the future branch `feat/jash005-browser-mic-ingestion`. Any baseline log belongs outside Git or in the review record.

**Interfaces:** Branch must start exactly at `011e1196538fdb91b6440c517d3bf137f1d99376` or an explicitly reviewed descendant. The implementation environment must install `backend/requirements.txt` plus the validated `soxr==1.1.0` candidate in a disposable environment before the requirements file is changed in Task 3.

**RED:** Run from a clean environment before adding the dependency:

```powershell
python --version
python -m venv backend/.venv-jash005
backend/.venv-jash005/Scripts/python.exe -m pip install -r backend/requirements.txt
backend/.venv-jash005/Scripts/python.exe -m pytest backend/tests -q
```

Record the clean baseline (expected current evidence: Python 3.13.7, 96 passed, 0 failed, 0 skipped, two deprecation warnings). Run frontend baseline from `frontend`: `npm ci`, `npm test -- --run`, `npm run lint`, `npm run build`, and `npm run test:e2e -- --workers=1`.

**GREEN:** Install `soxr==1.1.0` only in the disposable environment; verify import, `ResampleStream`, 48 kHz and 44.1 kHz streaming, arbitrary chunk boundaries, flush, finite float32 output, and exact/within-tolerance comparison with one-shot `soxr.resample`. Verify Torch is not imported by the backend environment.

**REFACTOR/commit:** Inspect status and diff; commit only the branch baseline if a commit is needed. Do not add generated environments.

## Task 2 — Backend VXAF protocol and source-frame continuity

**Commit:** `feat: add versioned microphone audio protocol`

**Files:** Create `backend/app/services/audio_protocol.py` and `backend/tests/test_audio_protocol.py`; modify only shared typed models if the repository requires it.

**Interfaces:** `AudioStartMetadata`; `ParsedAudioFrame`; `AudioProtocolError`; `parse_audio_start(payload) -> AudioStartMetadata`; `parse_audio_frame(bytes, expected_metadata) -> ParsedAudioFrame`; constants listed above. Parsed frames expose sequence, first source frame, count, payload byte length, channel count, and interleaved finite float32 samples.

**RED tests:** Exact header round-trip and offsets; wrong magic/version/header length/flags; unsupported format/rate/channels; sample count 0 or >1024; payload mismatch; NaN/Inf; truncated/extra bytes; binary before start/after stop. Add continuity tests for contiguous ranges, forward source gap accepted/counting, repeated range rejected, partial overlap rejected, increasing sequence with overlapping source rejected, advancing source with regressing sequence rejected, and uint64 end-position overflow rejected. Prove source gaps and transport-sequence gaps are separate telemetry and never authenticity evidence.

**GREEN:** Implement byte-level validation and typed parsing without resampling or audio persistence. Enforce `first_sample_frame > previous_source_frame_end`; compute end safely before accepting. Keep protocol errors explicit for route-level close codes/messages.

**REFACTOR/commit:** Keep parsing independent of `calls.py`; use named constants and one-line docstrings. Run the focused protocol tests and `git diff --check`.

## Task 3 — Stateful `StreamingAudioCanonicalizer`

**Commit:** `feat: add stateful streaming audio canonicalizer`

**Files:** Modify `backend/requirements.txt` to add `soxr==1.1.0`. Create or modify `backend/app/services/streaming_audio_canonicalizer.py` and `backend/tests/test_streaming_audio_canonicalizer.py`. Leave the existing complete WAV/FLAC conversion path unchanged.

**Interfaces:** `StreamingAudioCanonicalizer(source_rate, channels)`, `push(interleaved_float32, source_metadata) -> CanonicalAudioChunk`, `flush() -> CanonicalAudioChunk`, `close()`. It owns one persistent `soxr.ResampleStream`, downmixes before resampling, emits mono 16 kHz float32, carries source metadata segments, validates finite input, and disposes deterministically. It must not make AASIST decisions.

**RED tests:** Installation/import/readiness; mono and stereo downmix; 48 kHz→16 kHz and 44.1 kHz→16 kHz; arbitrary network-frame boundaries; finite float32 output; flush only emits samples derived from received audio and never zero-pads; closed input rejection. Compare one continuous source with the same source split at many arbitrary boundaries using an explicitly documented numerical tolerance (the validated `soxr==1.1.0` reference comparison is the baseline).

**GREEN:** Add the pinned dependency and implement persistent streaming conversion. Do not resample each frame independently, do not re-window, and do not add Torch.

**REFACTOR/commit:** Review memory bounds and disposal; run focused tests plus the unchanged WAV/FLAC conversion tests.

## Task 4 — Canonical source-segment ledger and AASIST window metadata

**Commit:** `feat: correlate microphone source ranges with model windows`

**Files:** Modify `backend/app/services/audio_windowing.py` and `backend/app/services/async_session.py`; create `backend/app/services/audio_source_ledger.py` and `backend/tests/test_audio_source_ledger.py`.

**Interfaces:** `AudioSourceSegment` and `AudioWindowMetadata` with `audio_source_frame_start/end`, transport sequence start/end, source gap count, and `audio_window_sequence`. Scheduler output must retain the same metadata while producing overlapping 64,600-sample windows with the existing JASH-004 hop.

**RED tests:** Continuous input; arbitrary network boundaries; source and transport gaps; overlap rejection; overlapping AASIST windows; resampler buffering/filter delay; flush; source-range ledger coverage. Assert metadata is derived from contributing source segments, not from naive `canonical_index * source_rate / 16000`, and does not influence score or policy.

**GREEN:** Maintain a bounded ordered segment ledger alongside canonical samples. When canonical output consumes a range, map the window to the union of contributing source segments, preserving gaps and sequence endpoints. Discard ledger entries only after no retained scheduler window can reference them.

**REFACTOR/commit:** Verify no unbounded audio/metadata retention and retain existing scheduler semantics.

## Task 5 — Single-producer backend audio WebSocket

**Commit:** `feat: add call-scoped microphone audio websocket`

**Files:** Create `backend/app/services/audio_stream.py` and `backend/tests/test_audio_stream.py`; modify `backend/app/api/calls.py` only to add the thin WS route and modify `backend/app/services/async_session.py` or `backend/app/state/session_store.py` only to claim/release the existing `AsyncCallSession`.

**Interfaces:** `AudioProducerSession`/ownership handle; `claim_audio_producer(call_id, session)`, `release_audio_producer(call_id)`, and `audio_stream_websocket(websocket, call_id)`. The route receives the exact shared `AsyncCallSession`; it never creates another scheduler, ML queue, or provider state.

**RED tests:** Unknown/non-LIVE call; invalid `audio_start`; binary before `audio_ready`; duplicate producer rejection without harming the owner; valid frames reaching the same scheduler/queue; late data; closed/cancelled session; normal `audio_stop`; unexpected disconnect; reset; idempotent release. Test the route does not emit risk events for malformed input.

**GREEN:** Implement `WS /api/v1/calls/{call_id}/audio-stream`: validate start, claim exactly one producer, create one canonicalizer, send `audio_ready`, process frames, flush on normal stop, release on every exit, and reject late/closed frames. Disconnect cancels the shared runtime session per JASH-004.

**REFACTOR/commit:** Keep route thin, no raw audio logs/files, and test cleanup in repeated execution.

## Task 6 — Optional correlation metadata on `LiveRiskEvent`

**Commit:** `feat: expose optional audio correlation on risk events`

**Files:** Modify `backend/app/models/risk.py`, the existing backend provider serialization, `frontend/src/domain/risk.ts`, `backend/tests/test_risk_stream.py`, and `frontend/src/domain/risk.test.ts`.

**Interfaces:** Optional event correlation fields match Task 4 metadata. Existing events without metadata remain valid. Unsupported speaker/prosody/replay/context numeric sentinels remain `0.0` with `evidence_availability: NOT_EVALUATED`; they are excluded from policy and render as N/A in ML mode.

**RED tests:** Old mock events validate unchanged; new metadata round-trips; missing optional metadata is accepted; unsupported evidence cannot create reasons or measured-safe UI values.

**GREEN:** Add only optional fields and serialization. Do not change AASIST score semantics, aggregation, thresholds, 20/70 policy, or mock values.

**REFACTOR/commit:** Verify frontend visual hierarchy is untouched and no unsupported signal is promoted.

## Task 7 — Frontend VXAF writer and bounded send transport

**Commit:** `feat: add bounded browser audio frame transport`

**Files:** Create `frontend/src/audio/protocol.ts`, `frontend/src/audio/audio-frame-accumulator.ts`, `frontend/src/audio/audio-transport.ts`, `frontend/src/audio/protocol.test.ts`, `frontend/src/audio/audio-frame-accumulator.test.ts`, and `frontend/src/audio/audio-transport.test.ts`.

**Interfaces:** `encodeAudioStart(metadata)`, `encodeAudioFrame(frame)`, `AudioFrameAccumulator`, `AudioTransport` with `start/send/stop/dispose`, and telemetry `{browser_frames_produced, browser_frames_sent, browser_frames_dropped, sequence_gap_count}`. Use `DataView` and BigInt-safe uint64 encoding.

**RED tests:** Exact 32-byte header bytes/endianness; payload bytes; mono/stereo; sequence and first frame; no frame >1024; residual final frame; no zero padding; BigInt overflow. Test congestion latch: `NORMAL → bufferedAmount > 262144 → CONGESTED`, drops produced frames while congested, resumes only below 65536, no retry queue, sequence gaps/telemetry visible.

**GREEN:** Implement exact writer and bounded transport. Sequence increments per produced frame even if dropped. Keep transport code independent of UI and risk events.

**REFACTOR/commit:** Add browser capability guards and deterministic fake socket tests.

## Task 8 — Minimal AudioWorklet accumulator

**Commit:** `feat: stream bounded microphone frames from AudioWorklet`

**Files:** Create `frontend/src/audio/microphone-worklet.ts`, `frontend/src/audio/audio-worklet-bridge.ts`, and `frontend/src/audio/audio-worklet-bridge.test.ts`.

**Interfaces:** Worklet message output contains Float32 channel arrays, `first_sample_frame`, and sample rate from the AudioContext timeline. Bridge exposes `connect`, `start`, `stop(flushResidual)`, and `dispose`.

**RED tests:** Render quanta of varied sizes accumulate exactly 1024 samples/channel; `currentFrame` produces correct first frame; max residual ≤1023; mono/interleaved and stereo layout; sequence per produced frame; final shorter aligned residual on normal stop; reset/error discards residual; no resampling/windowing/risk/storage.

**GREEN:** Copy quanta to a bounded accumulator and emit frames to Task 7. Use `currentFrame/currentTime/sampleRate`, not worklet `performance.now()`.

**REFACTOR/commit:** Keep worklet dependency-free and make channel behavior explicit (`channels` is 1 or 2, with mono capture preferred by the frontend).

## Task 9 — Transactional microphone session orchestration

**Commit:** `feat: add transactional browser microphone session`

**Files:** Create `frontend/src/audio/microphone-session.ts`, `frontend/src/audio/microphone-session.test.ts`, and `frontend/src/hooks/useMicrophoneSession.ts`; modify `frontend/src/hooks/useRiskStream.ts` only for orchestration injection if required.

**Interfaces:** `MicrophoneState = IDLE | REQUESTING_PERMISSION | CONNECTING | STREAMING | STOPPING | ERROR`; `MicrophoneSessionController` with `start`, `stop`, `reset`, `dispose`; injectable media devices, AudioContext, sockets, and generation clock for tests.

**RED tests:** No `getUserMedia` in mock mode; explicit gesture required; permission grant/deny/device failure; AudioContext/worklet/socket/backend failure at every startup boundary; cancellation while permission is pending; stale generation suppression; audio delivery disabled until backend `audio_ready`; pre-ready accumulator is reset/discarded; `STREAMING` is impossible before ready.

**GREEN:** Implement exact path: gesture→generation→getUserMedia→AudioContext→worklet→call create/start→risk WS→audio WS→audio_start→audio_ready→timing anchor→enable transport. Roll back all acquired resources on any partial failure, including late media tracks.

**REFACTOR/commit:** Keep network lifecycle outside presentation components and expose truthful status to existing controls.

## Task 10 — Coordinated stop/reset/disconnect lifecycle

**Commit:** `feat: coordinate microphone and risk session teardown`

**Files:** Modify `frontend/src/audio/microphone-session.ts`, `frontend/src/hooks/useMicrophoneSession.ts`, `frontend/src/services/call-session/CallSessionClient.ts`, `backend/app/services/audio_stream.py`, and add `frontend/src/audio/microphone-lifecycle.test.ts` plus `backend/tests/test_audio_stream.py` cases.

**Interfaces:** Idempotent `stop`, `reset`, `handleAudioDisconnect`, `handleRiskDisconnect`; backend normal-stop/failure hooks reuse JASH-004 stop/cancel APIs.

**RED tests:** Normal stop emits final permitted residual then `audio_stop`, closes audio/risk sockets, tracks, nodes, context, and backend call; unexpected audio disconnect releases ownership, closes canonicalizer/session, stops call and exposes error; unexpected risk disconnect does the same; reset is idempotent; no callbacks/resources remain.

**GREEN:** Implement teardown ordering and generation invalidation. No automatic reconnect. Never fabricate a risk event.

**REFACTOR/commit:** Test repeated stop/reset and inspect for orphan timers/sockets/tracks.

## Task 11 — Minimal existing UI integration

**Commit:** `feat: expose truthful microphone status in live console`

**Files:** Modify `frontend/src/components/console/SecurityConsole.tsx`, `frontend/src/app/App.tsx`, `frontend/src/hooks/useDemoController.ts` only where the existing live controller needs microphone state, and add `frontend/src/components/console/SecurityConsole.test.tsx`. No design-system rewrite.

**Interfaces:** Consume `MicrophoneState` and transport telemetry; display only necessary status: microphone off/requesting/connecting/live/error, `LIVE AUDIO ANALYSIS`, and `Raw audio is not retained` in the existing visual language.

**RED tests:** Mock mode never asks for permission or opens audio WS; state labels track actual lifecycle; error is visible; real ML unsupported signals remain N/A; existing mock sequence and blocked state remain exact.

**GREEN:** Wire existing start/stop/reset controls without moving the hierarchy or changing the risk policy. Keep DEMO mode entirely independent.

**REFACTOR/commit:** Compare screenshots/DOM against baseline and remove any unnecessary UI surface.

## Task 12 — Browser timing and latency telemetry

**Commit:** `feat: correlate browser capture timing with risk events`

**Files:** Create/modify `frontend/src/audio/timing.ts`, event correlation handling, and tests; modify backend event serialization only if Task 6 did not carry all metadata.

**Interfaces:** `createAudioTimingAnchor(contextCurrentTime, performanceNow, sampleRate)`, `captureEndPerformanceMs(sourceFrameEnd, anchor)`, and `steadyStateLatencyMs(eventArrival, captureEnd)`. Warm-up measurement is a separate session metric.

**RED tests:** Injected clocks produce deterministic values; final source sample of the analyzed AASIST window is used; source-rate/sample-index mapping is explicit; warm-up and steady-state values are not conflated and are labeled browser-observed, not ADC latency.

**GREEN:** Establish anchor only after `audio_ready`; use optional event source-end metadata to calculate latency on arrival.

**REFACTOR/commit:** Ensure telemetry is bounded and no audio payload is logged.

## Task 13 — Automated browser/integration harness and physical acceptance

**Commit:** `test: add browser microphone and live ML acceptance harness`

**Files:** Modify `frontend/playwright.config.ts`, create `frontend/e2e/microphone-live.spec.ts`, create `tests/integration/test_browser_mic_live_pipeline.py`, and create `docs/integration/JASH-005-BROWSER-MICROPHONE.md`.

**Interfaces:** Controlled fake media devices may test browser API/transport paths. The live acceptance command must start verified ML service, backend ML mode, and frontend, then exercise a real audio source through the existing risk WS. The final acceptance must use a real browser and physical microphone.

**RED tests:** Browser-generated PCM reaches backend; audio_ready gates delivery; canonicalizer creates real 64,600-sample windows; real AASIST executes; event reaches UI; unsupported fields are N/A; no CRITICAL/BLOCK_ACTION from real ML; mock path remains independent.

**GREEN:** Add opt-in acceptance harness with structured telemetry: browser/version, AudioContext rate/channels, duration, produced/sent/dropped frames, sequence/source gaps, canonical samples, windows, backend drops, raw scores, aggregates, policy states, warm-up, steady-state latency, inference latency. Use naturally long enough speech; never tune thresholds or fabricate expected output.

**REFACTOR/commit:** Keep physical acceptance separate from deterministic unit tests and document that human speech may score unexpectedly because AASIST is uncalibrated.

## Task 14 — Documentation, full verification, review, and PR handoff

**Commit:** `docs: document JASH-005 microphone ingestion and acceptance`

**Files:** Modify `frontend/README.md`, backend/runtime setup docs, `docs/integration/`, and relevant architecture docs. Do not modify `backend/` or `ml/` behavior beyond Task 3 dependency/runtime needs.

**Interfaces:** Document environment setup, `soxr==1.1.0`, audio/risk URLs, protocol fields, lifecycle, watermarks, source-frame correlation, privacy, mock/ML modes, warm-up versus steady-state latency, failure behavior, and known limitations.

**RED/GREEN:** Run fresh full checks and address only real failures with regression tests: `frontend` tests/lint/build/E2E; backend full pytest in clean env; relevant ML regression; mocked browser integration; physical mic acceptance. Run `git diff --check`, scope/secret/artifact inspection, and visual checks at 1920×1080 and 1366×768.

**REFACTOR/commit:** Invoke `superpowers:requesting-code-review`; fix every valid Critical/Important finding with tests; invoke `superpowers:verification-before-completion`; push branch and open PR only, never merge.

## Execution order and commit gates

Execute strictly: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14. Each task must have RED evidence, focused GREEN evidence, diff review, and its own logical commit. Do not parallelize tasks that share protocol/session files. Task 2 precedes Task 3; Task 3 and Task 4 precede Task 5; Tasks 7–8 precede Task 9; Task 6 precedes final UI/E2E validation.

## Verification and rollback/fallback checks

At every task boundary verify `git status`, `git diff --check`, and the task’s scope. Roll back a task by reverting its logical commit, never by resetting shared history. If soxr installation, clean backend baseline, browser capability, or physical microphone acceptance is unavailable, stop and report the exact evidence rather than substituting stateless resampling or fake audio.

Mock fallback must be tested after ML service shutdown: `VOXSENTINEL_RISK_PROVIDER=mock` produces exactly `18, 27, 43, 61, 79, 92 → CRITICAL → BLOCK_ACTION` without microphone, AudioContext, AudioWorklet, audio WS, ML service, or checkpoint. Real AASIST remains exactly `20/LOW/MONITOR` or `70/HIGH/REQUIRE_CALLBACK`, with ECAPA NOT_EVALUATED.

## Plan self-review

- Every approved spec requirement maps to Tasks 1–14.
- Audio and risk WebSockets are separate; one `AsyncCallSession`, producer, canonicalizer, and scheduler are reused per call.
- Source ranges cannot overlap; sequence and source gaps are distinct telemetry.
- Resampling is stateful and never per frame; canonicalization owns persistent phase/filter state.
- Browser backpressure is bounded and pre-`audio_ready` samples cannot enter analysis.
- Unsupported evidence stays unavailable; mock mode is fully microphone independent.
- Physical microphone acceptance is mandatory; no automatic reconnect or UI redesign is planned.
- No unresolved implementation decisions remain in the approved spec; all plan task boundaries name concrete files, interfaces, tests, commands, and commits.

**Open decisions: NONE.**
