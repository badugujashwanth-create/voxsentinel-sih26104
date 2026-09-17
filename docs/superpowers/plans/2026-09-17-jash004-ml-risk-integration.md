# JASH-004 Real AASIST Risk Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect real AASIST spoof evidence to the live VoxSentinel WebSocket through a stateless local ML service and backend-owned asynchronous session policy, while preserving the deterministic mock demo.

**Architecture:** The backend remains Torch-free. It decodes complete feeder WAV/FLAC containers, canonicalizes them to mono 16 kHz float32 PCM, schedules bounded overlapping 64,600-sample windows, and sends those windows to a long-lived loopback ML service. The ML service validates the exact window contract, runs the existing AASIST adapter, and returns raw uncalibrated evidence only. The backend asynchronously aggregates evidence and emits the existing LiveRiskEvent contract.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic, asyncio, httpx, NumPy, SciPy, SoundFile, PyTorch/AASIST in the isolated ML environment, pytest, React/TypeScript/Vitest for the minimal frontend availability metadata change.

**Spec:** docs/superpowers/specs/2026-09-17-jash004-ml-risk-integration-design.md

## Global Constraints

- Work only on feat/jash004-ml-risk-integration, created from the approved documentation state; do not modify main directly after branching.
- RiskProvider.stream() is an asynchronous iterator; no synchronous HTTP, model inference, blocking queue wait, or unbounded operation may run on the FastAPI event loop.
- MockRiskProvider keeps the exact existing scenario values and 18, 27, 43, 61, 79, 92 -> CRITICAL / BLOCK_ACTION behavior.
- MLRiskProvider is selected only through VOXSENTINEL_RISK_PROVIDER=mock|ml; ML failure never silently falls back to mock.
- The ML service is stateless between requests except for loaded model state and performs no call policy, aggregation, persistence, or raw-audio storage.
- Backend conversion is the only source-audio preprocessing authority: complete WAV/FLAC container -> mono 16 kHz float32 PCM -> 64,600-sample windows.
- The ML service accepts exactly mono, 16 kHz, float32 little-endian, 64,600-sample windows and performs no resampling, downmixing, padding, truncation, or re-windowing.
- Raw AASIST output remains an uncalibrated spoof-model score; it is never presented as a calibrated probability.
- Real ML policy has exactly NORMAL (20, LOW, MONITOR) and ELEVATED_AUTHENTICITY_REVIEW (70, HIGH, REQUIRE_CALLBACK). Real ML never emits CRITICAL, REQUIRE_SUPERVISOR, or BLOCK_ACTION.
- Unsupported speaker, prosody, replay, and unavailable context fields are explicitly marked NOT_EVALUATED, excluded from ML policy, and rendered as N/A by the frontend.
- Every new behavior follows RED -> GREEN -> REFACTOR, with the failing test run recorded before production implementation.
- Do not commit model weights, audio, datasets, .venv, node_modules, dist, .env, credentials, or API keys.

---

### Task 1: Create the implementation branch and verification baseline

**Files:**
- Create: no production files.
- Test: no new tests.

**Interfaces:**
- Consumes: approved spec commit 77fdb563a19eb33f30c1c306f09f57d16b585d09.
- Produces: branch feat/jash004-ml-risk-integration based on the current approved documentation state and a recorded clean baseline.

- [ ] **Step 1: Create the fresh branch from the approved state**

~~~
git fetch origin --prune
git checkout main
git status
git checkout -b feat/jash004-ml-risk-integration
git rev-parse HEAD
git status --short --branch
~~~

Expected: the branch contains the approved spec and plan documentation, the working tree is clean, and no existing JASH-004 implementation branch is reused.

- [ ] **Step 2: Run the baseline suites from the repository root**

~~~
cd frontend
npm test -- --run
npm run lint
npm run build
cd ..
python -m pytest backend/tests
$mlPython = Join-Path $env:TEMP 'rohan002-ml-venv\Scripts\python.exe'
& $mlPython -m pytest tests/ml
& $mlPython -m pytest tests/ml -m 'not integration'
~~~

Expected: frontend 30 tests pass, frontend lint/build pass, backend 62 tests pass with the known dependency warning, ML full 127 tests pass, and ML fast 115 tests pass with 12 integration tests deselected. If the isolated ML environment is unavailable, recreate it from ml/requirements.txt before continuing.

- [ ] **Step 3: Do not create a marker commit**

No marker commit is required. Do not create an empty commit.

### Task 2: Define asynchronous backend session contracts and provider selection

**Files:**
- Modify: backend/app/config.py
- Modify: backend/app/services/risk_provider.py
- Modify: backend/app/api/calls.py
- Create: backend/app/services/async_session.py
- Modify: backend/tests/test_risk_provider.py
- Modify: backend/tests/test_risk_stream.py
- Create: backend/tests/test_async_session.py
- Create: backend/tests/test_provider_selection.py

**Interfaces:**
- Consumes: existing CallSession, LiveRiskEvent, MockRiskProvider, and call WebSocket routes.
- Produces:
  - RiskProvider.stream(session: CallSession, cancellation: asyncio.Event) -> AsyncIterator[LiveRiskEvent].
  - AsyncCallSession with call_id, bounded asyncio.Queue[AudioWindow], asyncio.Event cancellation, generation token, sequence counter, and close().
  - AudioSessionRegistry with register(call_id), get(call_id), close(call_id, generation), and remove(call_id) operations.
  - Configuration validation accepting only VOXSENTINEL_RISK_PROVIDER=mock or ml; invalid values fail startup/configuration rather than silently choosing mock. Provider factory wiring is deferred to Task 7, after MLRiskProvider exists.

- [ ] **Step 1: Write failing async contract tests**

Add tests proving:

~~~
async def test_mock_provider_stream_is_async_and_preserves_fixture_values():
    events = [event async for event in MockRiskProvider().stream(session, asyncio.Event())]
    assert [event.overall_risk_score for event in events] == [18, 27, 43, 61, 79, 92]
    assert events[-1].recommended_action is RecommendedAction.BLOCK_ACTION

async def test_session_close_unblocks_waiting_consumer():
    session = AsyncCallSession(call_id="call-1", max_pending_windows=2)
    consumer = asyncio.create_task(session.next_window())
    await session.close()
    with pytest.raises(SessionClosedError):
        await consumer
~~~

Also test configuration validation for mock, ml, and invalid values without importing Torch or instantiating MLRiskProvider. Do not add a placeholder or stub MLRiskProvider in this task. Mock mode must never call ML health.

Add registry tests proving duplicate registration is rejected, closed sessions
reject input, close/remove are idempotent, generation tokens invalidate late
results, and a waiting consumer is released when its session closes.

- [ ] **Step 2: Run the focused tests to confirm RED**

~~~
cd backend
python -m pytest tests/test_async_session.py tests/test_provider_selection.py tests/test_risk_provider.py -q
~~~

Expected: failures because the async interfaces, session type, and configuration selection do not yet exist.

- [ ] **Step 3: Write the minimal async boundary**

Convert the provider abstraction and mock implementation to async iteration. Put mock pacing inside the mock provider so the WebSocket route only awaits the provider. Add the bounded AsyncCallSession and shared AudioSessionRegistry interfaces, including generation-token validation and idempotent close/remove behavior. Add configuration enum/string validation only; do not wire the ml value to a provider until Task 7. Keep fixture constants and event values unchanged.

The WebSocket route must use:

~~~
async for event in provider.stream(session, cancellation):
    await websocket.send_json(event.model_dump(mode="json"))
~~~

It must create cancellation in a try/finally, close the session in finally, and never call a synchronous iterator or time.sleep.

- [ ] **Step 4: Run focused tests to confirm GREEN**

~~~
python -m pytest backend/tests/test_async_session.py backend/tests/test_provider_selection.py backend/tests/test_risk_provider.py backend/tests/test_risk_stream.py -q
~~~

Expected: all focused tests pass and the mock stream remains contract compatible.

- [ ] **Step 5: Commit**

~~~
git add backend/app/config.py backend/app/services/risk_provider.py backend/app/api/calls.py backend/app/services/async_session.py backend/tests/test_risk_provider.py backend/tests/test_risk_stream.py backend/tests/test_async_session.py backend/tests/test_provider_selection.py
git commit -m "feat: add async risk provider session boundary"
~~~

### Task 3: Add backend-owned source conversion and AASIST window scheduling

**Files:**
- Modify: backend/requirements.txt
- Create: backend/app/services/audio_conversion.py
- Create: backend/app/services/audio_windowing.py
- Create: backend/app/services/audio_ingestion.py
- Modify: backend/app/services/async_session.py
- Modify: backend/app/api/calls.py
- Create: backend/tests/test_audio_conversion.py
- Create: backend/tests/test_audio_windowing.py
- Create: backend/tests/test_audio_ingestion.py

**Interfaces:**
- Consumes: complete independently decodable WAV/FLAC request bodies and AsyncCallSession.
- Produces:
  - decode_and_canonicalize(container: bytes, content_type: str) -> CanonicalAudio.
  - AASISTWindowScheduler.push(samples: np.ndarray) -> list[np.ndarray].
  - AudioSessionRegistry.ingest(call_id: str, container: bytes, content_type: str, chunk_sequence: int) -> AudioIngestAcknowledgement using the exact registry created in Task 2.
  - POST /api/v1/calls/{call_id}/audio accepting complete WAV/FLAC containers only.

- [ ] **Step 1: Write failing conversion/window tests**

Test that:

- mono 16 kHz input remains unchanged;
- stereo input is downmixed by the backend;
- non-16 kHz input is resampled by the backend;
- malformed, empty, non-finite, silent, and unsupported containers are rejected;
- arbitrary byte slices without a valid WAV/FLAC header are rejected;
- a 64,599-sample accumulation produces no window;
- 64,600 samples produce one window;
- adding 8,000 samples after a complete window produces the next overlapping window;
- a bounded queue never exceeds its configured capacity and reports drops.

- [ ] **Step 2: Run focused tests to confirm RED**

~~~
cd backend
python -m pytest tests/test_audio_conversion.py tests/test_audio_windowing.py tests/test_audio_ingestion.py -q
~~~

Expected: failures because the conversion, scheduler, registry, and endpoint do not exist.

- [ ] **Step 3: Implement backend-only preprocessing**

Add pinned non-Torch runtime dependencies numpy==2.3.4, scipy==1.16.3, and soundfile==0.13.1 to backend/requirements.txt. Decode only complete containers, convert to finite float32 mono at 16 kHz, reject unusable input, and schedule exactly 64,600-sample windows with an 8,000-sample default hop. Do not call the ML service from the ingestion request.

The endpoint must validate LIVE call state and X-Audio-Chunk-Sequence, reject duplicate/out-of-order chunks, and return accepted canonical sample count and dropped-window count. It must not retain raw bytes after conversion. If the session is closed or cancelled, reject the input; never enqueue into a dead queue.

When the bounded inference queue is full, remove the oldest pending window,
enqueue the newest complete window, increment dropped_window_count, and keep
the queue size at or below capacity. Add an identifiable-window test proving
the oldest pending sequence is evicted and the newest sequence is retained.

- [ ] **Step 4: Run focused tests to confirm GREEN**

~~~
python -m pytest backend/tests/test_audio_conversion.py backend/tests/test_audio_windowing.py backend/tests/test_audio_ingestion.py -q
~~~

Expected: all focused tests pass, including rejection of arbitrary container byte slices.

- [ ] **Step 5: Commit**

~~~
git add backend/requirements.txt backend/app/services/audio_conversion.py backend/app/services/audio_windowing.py backend/app/services/audio_ingestion.py backend/app/services/async_session.py backend/app/api/calls.py backend/tests/test_audio_conversion.py backend/tests/test_audio_windowing.py backend/tests/test_audio_ingestion.py
git commit -m "feat: add canonical audio ingestion and AASIST windows"
~~~

### Task 4: Build the stateless ML inference service contract

**Files:**
- Create: ml/runtime/__init__.py
- Create: ml/runtime/contracts.py
- Create: ml/runtime/inference.py
- Create: ml/runtime/server.py
- Modify: ml/spoof/aasist.py
- Create: tests/ml/runtime/test_contracts.py
- Create: tests/ml/runtime/test_service.py
- Modify: tests/ml/spoof/test_aasist_integration.py
- Modify: ml/README.md

**Interfaces:**
- Consumes: exact 64,600-sample float32le model-window requests and existing AASISTSpoofDetector.
- Produces:
  - SpoofInferenceRequest and SpoofInferenceResponse Pydantic models.
  - AASISTInferenceRuntime.infer(request: SpoofInferenceRequest) -> SpoofInferenceResponse.
  - FastAPI GET /health and POST /v1/spoof/infer.
  - AASISTSpoofDetector.score_model_window(samples: np.ndarray) -> SpoofResult for an already canonical exact-size window.

- [ ] **Step 1: Write failing service contract tests**

Test valid request parsing, exact byte-count validation, rejection of wrong sample rate/channels/format/sample count, response score_semantics == "uncalibrated", and service health without loading the model in unit tests. Add an integration-marked test that injects a ready detector double and verifies the runtime calls exact-window inference. Add regression tests proving ROHAN-002 file scoring still performs its existing preprocessing/padding behavior and that exact-window scoring rejects the wrong shape.

- [ ] **Step 2: Run focused tests to confirm RED**

~~~
$mlPython = Join-Path $env:TEMP 'rohan002-ml-venv\Scripts\python.exe'
& $mlPython -m pytest tests/ml/runtime/test_contracts.py tests/ml/runtime/test_service.py -q
~~~

Expected: failures because the runtime package and service contract do not exist.

- [ ] **Step 3: Implement validation and stateless inference**

Decode base64 only after validating metadata and exact byte length. Convert the bytes directly to a contiguous NumPy array/tensor. Add a focused exact-window method to AASISTSpoofDetector that does not resample, downmix, pad, truncate, or re-window, while leaving score_file/score behavior unchanged. Guard the loaded AASIST model with an asyncio.Semaphore(1) by default so at most one model forward runs at a time. Run model work through a bounded worker path rather than unbounded asyncio.to_thread submissions. GET /health reports ready only when the definition and checksum-verified checkpoint are loaded; missing, broken, or uninitialized model state reports unavailable/non-ready.

Return raw spoof score, model ID, provisional threshold metadata, prediction, warnings, and measured preprocessing/inference latency. Do not include call context or policy fields.

- [ ] **Step 4: Run focused tests to confirm GREEN**

~~~
& $mlPython -m pytest tests/ml/runtime/test_contracts.py tests/ml/runtime/test_service.py -q
~~~

Expected: all service tests pass; integration tests skip clearly when the verified checkpoint is unavailable.

- [ ] **Step 5: Commit**

~~~
git add ml/runtime ml/spoof/aasist.py ml/README.md tests/ml/runtime tests/ml/spoof/test_aasist_integration.py
git commit -m "feat: add stateless AASIST inference service"
~~~

### Task 5: Implement backend ML client and spoof-evidence adaptation

**Files:**
- Create: backend/app/services/ml_service_client.py
- Create: backend/app/models/ml_evidence.py
- Create: backend/tests/test_ml_service_client.py
- Create: backend/tests/test_ml_evidence.py
- Modify: backend/app/config.py

**Interfaces:**
- Consumes: SpoofInferenceRequest/Response and configured VOXSENTINEL_ML_SERVICE_URL.
- Produces:
  - async MLServiceClient.health() -> None.
  - async MLServiceClient.infer(window: np.ndarray) -> RawSpoofEvidence.
  - typed timeout, unavailable, malformed-response, and remote-error exceptions.

- [ ] **Step 1: Write failing client tests**

Test that the client sends the exact JSON model-window shape, awaits the async HTTP response, rejects invalid scores/semantics/latencies, applies request timeouts, and maps connection/HTTP failures to typed errors. Assert no synchronous requests or subprocess invocation exists.

- [ ] **Step 2: Run focused tests to confirm RED**

~~~
cd backend
python -m pytest tests/test_ml_service_client.py tests/test_ml_evidence.py -q
~~~

Expected: failures because the client and evidence model do not exist.

- [ ] **Step 3: Implement the async client and evidence adapter**

Use one reusable httpx.AsyncClient per backend application lifespan, with explicit connect/read/write/pool timeouts. Preserve raw score, model ID, threshold/version, score_semantics, warnings, and all latency measurements. Never convert the score to a percentage or call it calibrated.

- [ ] **Step 4: Run focused tests to confirm GREEN**

~~~
python -m pytest backend/tests/test_ml_service_client.py backend/tests/test_ml_evidence.py -q
~~~

- [ ] **Step 5: Commit**

~~~
git add backend/app/services/ml_service_client.py backend/app/models/ml_evidence.py backend/app/config.py backend/tests/test_ml_service_client.py backend/tests/test_ml_evidence.py
git commit -m "feat: add async ML service client and evidence contract"
~~~

### Task 6: Add backend temporal aggregation and exact two-state ML policy

**Files:**
- Create: backend/app/services/spoof_aggregation.py
- Create: backend/app/services/ml_risk_policy.py
- Create: backend/tests/test_spoof_aggregation.py
- Create: backend/tests/test_ml_risk_policy.py

**Interfaces:**
- Consumes: RawSpoofEvidence from the ML client.
- Produces:
  - TemporalSpoofAggregator.add(evidence) -> AggregatedSpoofEvidence.
  - MLRiskPolicy.evaluate(aggregate) -> MLPolicyDecision.

- [ ] **Step 1: Write failing aggregator/policy tests**

Test bounded five-observation median history, two-observation promotion at aggregate >= 0.5, two-observation demotion below 0.45, persistence counters, and reset behavior. Test exact decisions:

~~~
assert policy.initial().score == 20
assert policy.initial().level is RiskLevel.LOW
assert policy.initial().action is RecommendedAction.MONITOR

decision = after_two_observations_at_or_above_half()
assert decision.score == 70
assert decision.level is RiskLevel.HIGH
assert decision.action is RecommendedAction.REQUIRE_CALLBACK
~~~

Also assert real ML can never return CRITICAL, REQUIRE_SUPERVISOR, or BLOCK_ACTION, and call metadata does not change the decision.

- [ ] **Step 2: Run focused tests to confirm RED**

~~~
cd backend
python -m pytest tests/test_spoof_aggregation.py tests/test_ml_risk_policy.py -q
~~~

- [ ] **Step 3: Implement the deterministic backend policy**

Keep the latest five raw results, compute their median, and track threshold persistence/hysteresis. Use named constants SPOOF_REVIEW_THRESHOLD = 0.5, SPOOF_CLEAR_THRESHOLD = 0.45, SPOOF_PERSISTENCE_OBSERVATIONS = 2, NORMAL_OPERATIONAL_SCORE = 20, and ELEVATED_REVIEW_OPERATIONAL_SCORE = 70. The policy consumes only persisted spoof-evidence threshold state; call metadata is retained but ignored for score, level, and action.

- [ ] **Step 4: Run focused tests to confirm GREEN**

~~~
python -m pytest backend/tests/test_spoof_aggregation.py backend/tests/test_ml_risk_policy.py -q
~~~

- [ ] **Step 5: Commit**

~~~
git add backend/app/services/spoof_aggregation.py backend/app/services/ml_risk_policy.py backend/tests/test_spoof_aggregation.py backend/tests/test_ml_risk_policy.py
git commit -m "feat: add persistent spoof evidence policy"
~~~

### Task 7: Implement MLRiskProvider and wire async WebSocket events

**Files:**
- Modify: backend/app/services/risk_provider.py
- Modify: backend/app/api/calls.py
- Modify: backend/app/config.py
- Modify: backend/app/models/risk.py
- Modify: backend/app/services/async_session.py
- Create: backend/tests/test_ml_risk_provider.py
- Modify: backend/tests/test_provider_selection.py
- Modify: backend/tests/test_risk_stream.py

**Interfaces:**
- Consumes: audio-window session queue, MLServiceClient, TemporalSpoofAggregator, and MLRiskPolicy.
- Produces: async MLRiskProvider.stream() yielding validated LiveRiskEvent objects with optional evidence availability metadata.

The shared AudioSessionRegistry created in Task 2 is the only runtime-session
registry. Its exact lifecycle is:

1. POST /api/v1/calls/{call_id}/start awaits MLServiceClient.health() in ML mode.
2. Only a ready health result permits creation and registration of exactly one AsyncCallSession for call_id, followed by the call transition to LIVE.
3. POST /api/v1/calls/{call_id}/audio resolves that exact registered session and converts/windows/enqueues into it.
4. MLRiskProvider receives the same registry, resolves the same session by call_id, and consumes its bounded window queue.
5. The WebSocket asynchronously consumes MLRiskProvider with async for.
6. Stop, reset, or disconnect sets cancellation, closes the runtime session idempotently, removes the registry entry, and rejects late generation-token results.

- [ ] **Step 1: Write failing provider integration tests**

Cover:

- event sequence increments monotonically;
- raw synthetic evidence is copied without calibration;
- speaker_match_score, speaker_mismatch_score, prosody_anomaly_score, and replay_risk_score are marked unavailable and excluded from the policy;
- ML policy emits exactly 20/LOW/MONITOR or 70/HIGH/REQUIRE_CALLBACK;
- client timeout, malformed response, queue closure, and reset produce errors without fake events;
- cancellation prevents late events after reset;
- mock provider’s full scenario remains unchanged.

Add lifecycle tests proving ML start fails visibly when health is unavailable,
failed start creates no runtime session, audio before a valid ML session is
rejected, the audio route and MLRiskProvider resolve the same session object,
duplicate session creation is prevented, cleanup is idempotent, and no event is
emitted after reset. Assert mock mode never calls ML health.

- [ ] **Step 2: Run focused tests to confirm RED**

~~~
cd backend
python -m pytest tests/test_ml_risk_provider.py tests/test_risk_stream.py -q
~~~

- [ ] **Step 3: Implement provider orchestration**

Wire provider factory selection here, after MLRiskProvider exists: mock returns
MockRiskProvider and ml returns MLRiskProvider with the shared registry and
client dependencies. For each complete window, await the ML client, adapt
evidence, aggregate it, apply the exact two-state policy, and build one
LiveRiskEvent. Use numeric compatibility sentinel values of exactly 0.0 only
for the stable transport schema, accompanied by evidence_availability
metadata. Never put unsupported fields into reasons or policy calculations.

Update the WebSocket route to consume async for, await sends, set cancellation in finally, and close the session registry. Ensure the route never falls back from ML to mock.

- [ ] **Step 4: Run focused tests to confirm GREEN**

~~~
python -m pytest backend/tests/test_ml_risk_provider.py backend/tests/test_risk_stream.py backend/tests/test_risk_provider.py -q
~~~

- [ ] **Step 5: Commit**

~~~
git add backend/app/services/risk_provider.py backend/app/api/calls.py backend/app/models/risk.py backend/app/services/async_session.py backend/tests/test_ml_risk_provider.py backend/tests/test_risk_stream.py
git commit -m "feat: stream real spoof evidence through MLRiskProvider"
~~~

### Task 8: Preserve truthful frontend rendering for unavailable ML signals

**Files:**
- Modify: frontend/src/domain/risk.ts
- Modify: frontend/src/components/console/SecurityConsole.tsx
- Modify: frontend/src/domain/risk.test.ts
- Create: frontend/src/components/console/SecurityConsole.test.tsx

**Interfaces:**
- Consumes: optional evidence_availability metadata on LiveRiskEvent.
- Produces: existing visual layout with N/A and Not evaluated for unsupported ML evidence, while mock events render exactly as before.

- [ ] **Step 1: Write failing frontend tests**

Add a validated ML event with speaker, prosody, replay, and context marked NOT_EVALUATED and each corresponding transport field set to exactly 0.0. Assert the evidence rail shows N/A and Not evaluated, does not show 0%, 100% identity mismatch, or unsupported claims, and preserves normal numeric mock rendering.

- [ ] **Step 2: Run focused tests to confirm RED**

~~~
cd frontend
npm test -- --run src/domain/risk.test.ts src/components/console/SecurityConsole.test.tsx
~~~

- [ ] **Step 3: Implement the smallest contract-aware rendering change**

Extend validation to preserve the optional metadata without breaking existing events. Branch only the evidence display values and interpretations; do not alter layout, typography, signal visualization, or mock behavior.

- [ ] **Step 4: Run focused tests to confirm GREEN**

~~~
npm test -- --run src/domain/risk.test.ts src/components/console/SecurityConsole.test.tsx
~~~

- [ ] **Step 5: Commit**

~~~
git add frontend/src/domain/risk.ts frontend/src/components/console/SecurityConsole.tsx frontend/src/domain/risk.test.ts frontend/src/components/console/SecurityConsole.test.tsx
git commit -m "fix: label unavailable ML evidence honestly"
~~~

### Task 9: Add real-audio feeder, service setup, and integration acceptance harness

**Files:**
- Create: scripts/feed_audio.py
- Create: scripts/run_ml_acceptance.py
- Create: tests/integration/test_ml_live_pipeline.py
- Create: docs/integration/JASH-004-ML-RUNTIME.md
- Modify: backend/README.md
- Modify: ml/README.md
- Modify: .gitignore

**Interfaces:**
- Consumes: actual LibriSpeech and deterministic Piper-generated WAV/FLAC files recreated through documented ROHAN-002 tooling.
- Produces: a controlled feeder that posts complete independently decodable containers to the backend audio endpoint and records real raw/aggregate/policy/latency evidence.

The acceptance fixture must provide at least two complete AASIST observations:
64,600-sample first window plus an 8,000-sample hop requires at least 72,600
canonical 16 kHz samples, or 4.5375 seconds. Select a genuine Piper evaluation
sample that naturally meets this duration after backend canonicalization. Do
not pad, concatenate, repeat, or otherwise alter audio to force duration or
classification. Prefer an existing LibriSpeech sample that naturally provides
at least two windows as well. Record the selected sample and resulting window
count.

- [ ] **Step 1: Write failing acceptance harness tests**

Test feeder chunk sequencing, complete-container generation, backend URL configuration, refusal to send arbitrary byte slices, and assertions that ML acceptance events contain a real model ID and score_semantics == "uncalibrated". Keep tests integration-marked and skip only with an explicit missing-service/model/audio reason.

- [ ] **Step 2: Run focused tests to confirm RED**

~~~
python -m pytest tests/integration/test_ml_live_pipeline.py -q
~~~

- [ ] **Step 3: Implement the controlled feeder and documentation**

The feeder must:

1. read a complete real source file;
2. decode it in the ML environment;
3. split the decoded samples into time chunks without changing their content;
4. encode every chunk as a complete standalone WAV container;
5. POST each container in sequence to the backend;
6. never send labels, scores, arbitrary byte slices, or stored raw audio.

Document exact commands for:

- creating ml/.venv and installing ml/requirements.txt;
- running ml/scripts/setup_aasist.py --verify-only;
- starting the ML service;
- starting the backend with VOXSENTINEL_RISK_PROVIDER=ml;
- starting the frontend in live mode;
- recreating LibriSpeech/Piper evaluation data;
- feeding real bonafide and synthetic files;
- stopping/resetting sessions;
- running mock mode with the ML process stopped.

Document that real ML output is uncalibrated and that JASH-004 lacks speaker verification, replay detection, prosody detection, multilingual/telephony validation, and final risk calibration.

- [ ] **Step 4: Run focused harness tests to confirm GREEN**

~~~
python -m pytest tests/integration/test_ml_live_pipeline.py -q
~~~

- [ ] **Step 5: Commit**

~~~
git add scripts/feed_audio.py scripts/run_ml_acceptance.py tests/integration/test_ml_live_pipeline.py docs/integration/JASH-004-ML-RUNTIME.md backend/README.md ml/README.md .gitignore
git commit -m "test: add real audio ML pipeline acceptance harness"
~~~

### Task 10: Execute complete verification and inspect scope

**Files:**
- Modify: only files required to correct verified failures.
- Test: all existing and new suites.

**Interfaces:**
- Consumes: all prior tasks and real acceptance environments.
- Produces: fresh test, latency, mock-fallback, real-audio, browser, and repository-scope evidence for review.

- [ ] **Step 1: Run all automated suites freshly**

~~~
cd frontend
npm test -- --run
npm run lint
npm run build
npm run test:e2e -- --workers=1
cd ..
python -m pytest backend/tests
$mlPython = Join-Path $env:TEMP 'rohan002-ml-venv\Scripts\python.exe'
& $mlPython -m pytest tests/ml
& $mlPython -m pytest tests/ml -m 'not integration'
python -m pytest tests/integration/test_ml_live_pipeline.py -q
~~~

Expected: all existing suites remain green; the mock canonical E2E remains unchanged; ML unit and integration results are recorded separately.

- [ ] **Step 2: Run the complete real bonafide path**

Start the ML runtime, backend in ML mode, and frontend live mode. Feed one actual LibriSpeech file. Record the real model score, aggregate evidence, operational 20/LOW/MONITOR or 70/HIGH/REQUIRE_CALLBACK state, reasons, preprocessing/inference/provider/WebSocket latency, and browser validation. Confirm the output came from the returned model ID and not a mock scenario table.

- [ ] **Step 3: Run the complete real synthetic path**

Feed one actual Piper-generated file recreated from the committed evaluation tooling. Record the same evidence and verify the result uses real AASIST inference, never hardcoded 92, never file-label-based blocking, and never unsupported detector claims.

- [ ] **Step 4: Run the mock fallback with ML stopped**

Stop the ML service completely. Configure VOXSENTINEL_RISK_PROVIDER=mock and run the canonical high-value demo. Verify 18, 27, 43, 61, 79, 92, CRITICAL, and BLOCK_ACTION exactly as before.

- [ ] **Step 5: Verify reset, shutdown, and failure behavior**

Exercise ML unavailable, malformed service response, inference timeout, malformed audio, silence, too-short source, queue saturation, WebSocket disconnect, and reset during inference. Confirm errors are operator-visible, no fake event is emitted, no duplicate socket/task remains, and mock mode remains available.

- [ ] **Step 6: Perform browser and visual regression inspection**

Inspect the existing UI at 1920x1080 and 1366x768 in mock mode and ML mode. Confirm no visual redesign, no console errors, truthful N/A / Not evaluated labels, preserved critical mock screenshot, and no primary-content clipping. Capture fresh screenshots only as untracked verification artifacts.

- [ ] **Step 7: Review repository scope and secrets**

~~~
git status
git diff origin/main...HEAD --stat
git diff origin/main...HEAD --name-status
git ls-files | Select-String -Pattern '(^|/)(\.env|node_modules|dist|coverage|\.venv)(/|$)|\.(pth|pt|ckpt|wav|flac)$'
git diff --check
~~~

Expected: no backend/ML contamination outside the approved integration scope, no frontend visual redesign, no weights/audio/datasets/secrets/generated artifacts, and a clean working tree after removing only untracked verification outputs that are not ignored.

- [ ] **Step 8: Commit verification fixes only**

Use focused fixes with tests first. Do not squash away the logical commits. Finish with a clear commit such as:

~~~
git commit -m "test: verify JASH-004 live and mock paths"
~~~

### Task 11: Independent review and PR handoff

**Files:**
- Modify: none unless review finds a Critical or Important defect.

**Interfaces:**
- Consumes: origin/main...HEAD, approved spec, this plan, fresh verification evidence.
- Produces: reviewed branch and PR feat/jash004-ml-risk-integration -> main.

- [ ] **Step 1: Request independent code review**

Provide the reviewer:

- base: origin/main;
- current HEAD;
- JASH-004 specification;
- this implementation plan;
- exact mock/live acceptance evidence;
- statement that real ML never emits CRITICAL, REQUIRE_SUPERVISOR, or BLOCK_ACTION.

- [ ] **Step 2: Address findings**

Fix every valid Critical and Important finding with a failing regression test first. Report valid Minor findings separately. Re-run the affected focused suite and then the complete verification commands from Task 10.

- [ ] **Step 3: Push and open the PR without merging**

~~~
git push -u origin feat/jash004-ml-risk-integration
gh pr create --base main --head feat/jash004-ml-risk-integration --title "JASH-004: Integrate real AASIST inference into live risk pipeline" --body-file docs/integration/JASH-004-ML-RUNTIME.md
~~~

Do not merge the PR and do not start JASH-005 or speaker verification.

## Rollback and fallback procedure

1. If ML integration fails during development, keep VOXSENTINEL_RISK_PROVIDER=mock and run only the unchanged mock provider path.
2. To disable the real provider without code changes, set VOXSENTINEL_RISK_PROVIDER=mock and stop the ML runtime.
3. To revert the integration branch safely, revert the logical JASH-004 commits in reverse order; never reset or force-push main.
4. If an ML service session fails, close its bounded session, discard late results by generation token, report the error, and require operator restart; do not fabricate or substitute scores.
5. Preserve the approved frontend and backend contracts while reverting only the integration branch.

## Plan self-review

- Async event delivery is explicit and all backend/ML waits are awaitable or isolated from the FastAPI event loop.
- The ML service has one preprocessing authority boundary and cannot resample, pad, truncate, or re-window valid model requests.
- Aggregation and operational policy are backend-owned and do not derive overall risk by multiplying an uncalibrated AASIST score.
- Real ML has exactly two operational states and no blocking action.
- Mock behavior remains unchanged and is tested independently.
- Complete WAV/FLAC containers are required; arbitrary byte-slicing is rejected.
- Unsupported detector dimensions are unavailable, excluded from policy, and rendered honestly.
- Real LibriSpeech and Piper acceptance paths use the complete backend -> service -> provider -> WebSocket -> frontend flow.
- Every production behavior has a RED -> GREEN test sequence and a focused commit.
- The plan contains no unresolved TBD/TODO decisions or placeholder file paths.
