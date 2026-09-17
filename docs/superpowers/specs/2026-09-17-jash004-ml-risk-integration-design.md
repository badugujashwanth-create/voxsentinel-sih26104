# JASH-004 Real AASIST to Live Risk Pipeline Design

**Status:** Awaiting implementation approval  
**Date:** 2026-09-17  
**Owner:** Jashwanth  
**Branch for implementation:** `feat/jash004-ml-risk-integration`

## Goal

Connect the already-merged AASIST spoof detector to the backend live-call
pipeline through a long-lived local ML service, while preserving the existing
deterministic mock/demo path and keeping final risk fusion out of scope.

JASH-004 supplies real voice-synthetic evidence only. It does not provide
speaker verification, replay detection, prosody detection, fraud certainty, or
final intervention policy.

## Current repository boundary

The current system has three relevant seams:

- `backend/app/services/risk_provider.py` defines `RiskProvider` and the
  hand-authored `MockRiskProvider`.
- `backend/app/api/calls.py` owns call lifecycle routes and emits
  `LiveRiskEvent` objects over the WebSocket.
- `ml/spoof/detector.py` defines `SpoofDetector` and `SpoofResult`, while
  `ml/spoof/aasist.py` loads the verified official AASIST implementation and
  checkpoint from the ML environment.

The backend currently has no audio ingestion, no `MLRiskProvider`, and no ML
service. The existing `ml/audio/chunker.py` defaults to a 2-second window and
cannot be passed directly to AASIST, whose required model input is exactly
64,600 samples at 16 kHz.

Baseline evidence before implementation:

- Frontend: 30 tests passed; lint and production build passed.
- Backend: 62 tests passed with one existing deprecation warning.
- ML: 127 tests passed; fast suite 115 passed and 12 were deselected.
- Working tree: clean on `main` at
  `dc8b8e85051e5ec5d793405f02df37dc13f99c2b`.

## Architecture decision

### Selected approach: long-lived local ML inference service

The ML environment runs a persistent loopback service with AASIST loaded once.
The Torch-free FastAPI backend sends one validated, model-sized PCM window at a
time over a narrow typed HTTP boundary. The ML service returns raw model
evidence and latency metadata only. The backend owns call semantics,
aggregation, risk mapping, and `LiveRiskEvent` construction.

```text
Controlled real-audio feeder
        |
        v
Backend audio ingestion and bounded session buffer
        |
        v
Backend AASIST windowing: 64,600 samples, configurable overlap
        |
        v
Long-lived ML service: preprocessing -> AASIST -> raw evidence
        |
        v
Backend SpoofEvidenceAdapter
        |
        v
Backend TemporalSpoofAggregator
        |
        v
Backend conservative ML risk policy
        |
        v
MLRiskProvider -> existing LiveRiskEvent -> existing WebSocket -> frontend
```

The ML service is stateless between inference requests. It keeps only loaded
model state and request-local processing state. It does not own call sessions,
risk history, policy, or persistent audio.

### Asynchronous session and event architecture

`RiskProvider.stream()` becomes an asynchronous iterator:

```python
async def stream(
    self,
    session: CallSession,
    cancellation: asyncio.Event,
) -> AsyncIterator[LiveRiskEvent]: ...
```

Both providers implement this contract. `MockRiskProvider` awaits its existing
transport interval between fixture events. `MLRiskProvider` awaits an
async-session queue containing complete AASIST windows and yields events as
inference results become available.

The backend owns one `AsyncCallSession` per live ML call. It contains:

- a bounded incoming audio/window queue;
- an `asyncio.Event` cancellation signal;
- a monotonically increasing event sequence;
- a task-safe closed/disposed flag;
- the active provider stream task.

The WebSocket handler consumes the provider with `async for` and sends each
event with `await websocket.send_json(...)`. The audio-ingestion route only
performs bounded conversion/window scheduling and `put_nowait` operations; it
never waits on model inference. The ML client uses an asynchronous HTTP client
and awaits loopback responses. No synchronous HTTP call, model wait, or
unbounded queue operation is permitted on the FastAPI event loop.

On WebSocket close, stop, or reset, the handler sets the cancellation event,
closes the session queue, cancels the provider task when necessary, awaits its
termination, and releases the session. A late inference result is discarded if
the session generation no longer matches. This prevents duplicate sockets,
orphan tasks, and events emitted after reset.

### Rejected alternatives

**In-process Torch inside FastAPI** is rejected because it breaks the deliberate
backend/ML dependency boundary and makes mock-mode operation depend on Torch.

**A fresh subprocess per window** is explicitly rejected. Loading Python and
the AASIST checkpoint for every 4.04-second window would add unacceptable
startup overhead and create avoidable process-failure races.

**A long-lived IPC worker** remains technically viable, but is not selected for
JASH-004. It introduces more lifecycle and observability complexity than a
loopback HTTP service while providing no required product benefit at this
stage.

## Process and runtime contract

### Mock mode

Mock mode starts only the existing backend/frontend path:

```text
VOXSENTINEL_RISK_PROVIDER=mock
MockRiskProvider -> existing LiveRiskEvent WebSocket
```

It remains deterministic and continues to emit the existing scenario sequence,
including the high-value demo progression `18, 27, 43, 61, 79, 92` and its
`CRITICAL / BLOCK_ACTION` final event. Mock mode never contacts the ML service.

### ML mode

ML mode requires:

1. The ML environment and verified AASIST checkpoint.
2. The long-lived loopback ML service.
3. The backend configured with `VOXSENTINEL_RISK_PROVIDER=ml`.
4. The backend audio-ingestion path.
5. A controlled feeder for real LibriSpeech or Piper audio during acceptance.
6. The existing frontend configured for live backend mode.

The backend must health-check the ML service before or at session start. If the
service is unavailable, ML mode reports an operator-visible unavailable/error
state. It never silently switches to `MockRiskProvider`.

## ML service responsibility and API

The ML service owns only stateless model-local work:

1. Validate the request representation.
2. Validate that the request is already mono, 16 kHz, float32 little-endian,
   and exactly 64,600 samples.
3. Convert the validated byte payload directly to the model tensor.
4. Run the already-verified `AASISTSpoofDetector` model-window inference.
5. Return raw score, model identity, score semantics, warnings, and latency.

The service does not know `call_id`, scenario identity, transaction value,
speaker identity, or recommended action.

The implementation should expose a loopback-only contract equivalent to:

```text
GET  /health
POST /v1/spoof/infer
```

The typed response must include:

```text
model_id: string
raw_spoof_score: number       # 0..1, uncalibrated model output
score_semantics: "uncalibrated"
prediction: "bonafide" | "spoof"
threshold: number             # provisional model operating threshold
audio_duration_ms: number
preprocessing_ms: number
inference_ms: number
warnings: string[]
```

The request will be a JSON object with this exact first-version shape:

```json
{
  "sample_rate": 16000,
  "channels": 1,
  "sample_format": "float32le",
  "sample_count": 64600,
  "pcm_base64": "..."
}
```

`pcm_base64` must decode to exactly `sample_count * 4` bytes. The service
rejects any other sample rate, channel count, sample format, byte alignment, or
sample count with a typed 4xx response. It must not resample, downmix, pad,
truncate, or re-window a valid request. The ML runtime will add a focused
model-window inference method to the existing AASIST adapter so this exact
contract bypasses its file-oriented preparation path without duplicating
AASIST code.

The service must not write raw audio, model weights, or inference payloads to
the repository or normal logs.

## Audio ingestion and AASIST windowing

JASH-004 will not add Twilio, PSTN, browser microphone capture, or permanent
audio storage. A controlled development feeder will produce short, complete,
independently decodable WAV or FLAC containers and send them to the backend
session ingestion boundary. Arbitrary byte-slicing of a WAV or FLAC file is
forbidden. The backend is the single authority for conversion and window
scheduling.

The backend owns call-scoped buffering and AASIST window scheduling:

- source chunks may declare a supported source sample rate and channel count;
- backend conversion produces exactly one canonical mono channel at 16,000 Hz;
- model window: 64,600 samples, approximately 4.0375 seconds;
- default hop: 8,000 samples, 0.5 seconds;
- hop remains configurable through named settings;
- a first inference is not attempted until a complete model window exists;
- subsequent windows overlap and use meaningful preceding context;
- short audio does not get repeatedly padded as independent 0.5-second
  observations.

The feeder must use one actual LibriSpeech sample and one actual
Piper-generated sample. Sine waves, noise proxies, and scenario-known scores
are not valid acceptance inputs.

The backend conversion layer decodes the feeder's WAV/FLAC chunk, downmixes
channels, resamples to 16 kHz, rejects non-finite/silent/empty input, and feeds
only canonical PCM to the AASIST window scheduler. The ML service does not
repeat any of these operations. It validates the canonical model-window
contract and converts bytes to a tensor only. This prevents two preprocessing
authorities from producing different model inputs.

The backend development ingestion endpoint is:

```text
POST /api/v1/calls/{call_id}/audio
Content-Type: audio/wav | audio/flac
X-Audio-Chunk-Sequence: integer >= 1
```

The body is one complete, independently decodable WAV or FLAC container
produced by the controlled feeder. A request body that is merely an arbitrary
byte slice from a larger container is invalid and must be rejected. The
endpoint accepts audio only for an existing `LIVE` call, converts it through
the backend's single canonical conversion path, adds samples to that call's
bounded buffer, and returns an acknowledgement containing accepted canonical
sample count and dropped-window count. It does not write the body to disk. The
feeder does not send a scenario label or a precomputed score.

The backend-to-ML boundary is separate from this source-ingestion endpoint and
always uses the exact JSON model-window request defined above. No source-audio
format reaches the ML service.

## Bounded backpressure

Each live ML session has a bounded audio buffer and bounded inference work
queue. The backend must not allow an unlimited queue to accumulate when CPU
inference is slower than the selected hop.

When capacity is reached, the policy is:

1. Never enqueue more than the configured maximum number of pending windows.
2. Drop stale pending windows before dropping the newest complete context.
3. Preserve the most recent complete window for the next available inference.
4. Track dropped-window count in internal diagnostics.
5. Return an explicit unavailable/backpressure error if the stream cannot
   continue safely.

The implementation must measure preprocessing, ML inference, provider
round-trip, and end-to-end event timing and report whether the selected hop
keeps up on the verification machine.

## Backend provider and policy layers

The backend keeps this provider structure:

```text
RiskProvider
├── MockRiskProvider
└── MLRiskProvider
```

Provider selection is configuration-driven and has no hardcoded provider choice
in route handlers. `MockRiskProvider` remains behaviorally unchanged.

`MLRiskProvider` coordinates the session’s audio windows, calls the ML client,
passes results through the evidence adapter and temporal aggregator, and
constructs the existing `LiveRiskEvent` contract.

### Spoof evidence adapter

The adapter preserves, without fake calibration:

- raw AASIST spoof score;
- model/checkpoint identifier;
- score semantics (`uncalibrated`);
- provisional threshold and threshold version;
- warnings;
- preprocessing, inference, and round-trip latency.

The value carried in the existing `synthetic_probability` field remains an
ML-derived evidence score for compatibility. It must not be described as a
calibrated probability or as the percentage chance that a voice is fake.

### Temporal aggregation

Aggregation belongs to the backend because it is call/session semantics, not
model semantics. The initial deterministic strategy is:

- retain at most the five most recent model results;
- calculate their median as the aggregate spoof evidence;
- require two consecutive aggregate results across a threshold before
  promoting the reported level;
- require two consecutive results at least `0.05` below a threshold before
  demoting the reported level;
- expose latest raw score, aggregate score, observed-window count, history,
  threshold state, and latency metadata to the provider.

The model's documented provisional decision threshold is `0.5` raw spoof
score. It is an operating threshold inherited from the current AASIST
evaluation setup, not a calibrated probability threshold. The backend policy
uses it only as an evidence trigger: an aggregate at or above `0.5` must
persist for two consecutive windows before entering the elevated authenticity
review state; it must remain below `0.45` for two consecutive windows before
leaving that state. These values are named settings and are explicitly
provisional.

The aggregator does not convert aggregate evidence into `overall_risk_score`.
It returns the latest raw score, recent median, persistence counters,
threshold state, observed-window count, bounded history, and latency metadata.
The subsequent policy layer decides the operational event score and level from
that persisted spoof-evidence threshold state only. Call metadata may be
retained for the session, but it must not alter `overall_risk_score`,
`risk_level`, or `recommended_action` in JASH-004. This prevents an
uncalibrated AASIST score from being presented as a calibrated 0–100 risk
number.

### Conservative ML risk mapping

The ML mapper converts the aggregator's persisted threshold state into a
conservative authenticity-review state in the existing 0–100 event contract.
The mapping is isolated so later speaker evidence and final fusion can replace
it. It is an operational policy, not calibration and not a mathematical claim
that the model score equals the event score.

JASH-004 policy rules:

- AASIST evidence is not identity mismatch, fraud certainty, or authorization.
- Persistent evidence above the documented provisional model threshold may
  produce elevated authenticity risk and request secondary review.
- `BLOCK_ACTION` is not emitted by the real-ML path in JASH-004.
- No one-window spike may directly produce a catastrophic intervention.
- The deterministic mock path retains its existing `BLOCK_ACTION` behavior.
- Final CRITICAL/action fusion remains a later task.

JASH-004 has exactly two real-ML policy states:

| Policy state | Entry/exit rule | Operational score | Risk level | Action |
| --- | --- | ---: | --- | --- |
| `NORMAL` | Initial state; re-enter after aggregate evidence is below `0.45` for two consecutive observations | `20` | `LOW` | `MONITOR` |
| `ELEVATED_AUTHENTICITY_REVIEW` | Enter after aggregate evidence is at least `0.5` for two consecutive observations | `70` | `HIGH` | `REQUIRE_CALLBACK` |

The real-ML provider never emits `CRITICAL`, `REQUIRE_SUPERVISOR`, or
`BLOCK_ACTION`. Scores `20` and `70` are operational severity constants only;
they are not transformed AASIST probabilities and not calibrated risk
probabilities. The exact constants must be named in backend policy settings
and documented as operational labels.

The event’s `overall_risk_score` and `risk_level` must continue to satisfy the
existing frontend/backend invariant. Reasons in ML mode may describe only
actual spoof evidence, for example “Synthetic speech characteristics detected
by AASIST evidence.” They must not claim speaker verification, replay
detection, or prosody analysis.

## Unsupported LiveRiskEvent evidence

The stable event currently requires numeric fields for:

- `speaker_match_score`;
- `speaker_mismatch_score`;
- `prosody_anomaly_score`;
- `replay_risk_score`.

Those detectors do not exist. Numeric placeholders must therefore be treated as
unavailable internally, excluded from ML-mode risk calculation, and prevented
from generating reasons or UI claims. They must never be interpreted as safe,
verified, or measured zero-risk evidence.

The implementation will use a small additive optional event metadata field,
without changing existing mock payloads:

```typescript
evidence_availability?: {
  synthetic: "AVAILABLE" | "NOT_EVALUATED";
  speaker: "AVAILABLE" | "NOT_EVALUATED";
  prosody: "AVAILABLE" | "NOT_EVALUATED";
  replay: "AVAILABLE" | "NOT_EVALUATED";
  context: "AVAILABLE" | "NOT_EVALUATED";
};
```

ML events mark `synthetic: "AVAILABLE"` and unsupported detector dimensions
as `"NOT_EVALUATED"`. Context is marked available only when it is actually
derived from call metadata; otherwise it is not evaluated.

The current frontend renders numeric evidence values unconditionally. The
smallest truthful frontend adjustment is to preserve the existing layout and
render `N/A` plus `Not evaluated` for dimensions marked unavailable in ML mode.
Mock events without the optional metadata retain their current presentation.

This is a contract-clarity correction, not a visual redesign.

## Error and lifecycle behavior

The backend must handle these cases explicitly:

- ML health check unavailable;
- model definition/checkpoint missing;
- checkpoint checksum/setup failure;
- malformed or unsupported PCM;
- invalid sample metadata;
- silence;
- too-short audio;
- ML inference timeout;
- ML inference exception or invalid response;
- bounded queue saturation;
- session reset while inference is running;
- client WebSocket disconnect.

Errors are surfaced through backend logs/diagnostics and the existing
operator-visible connection/error path. No error creates a fabricated score.

Reset/disposal cancels or invalidates outstanding session work, closes the
ingestion stream, clears the bounded buffer, prevents late results from being
emitted, and leaves no orphan worker or socket. A normal stop closes the live
session cleanly. Automatic reconnect is not required for JASH-004; an
unexpected ML or WebSocket disconnect is reported as unavailable and requires
operator restart.

## Real acceptance paths

### Bonafide

An actual LibriSpeech file is fed through:

```text
real file -> controlled feeder -> backend ingestion
-> 64,600-sample window -> ML service -> AASIST
-> MLRiskProvider -> existing WebSocket -> frontend validation
```

The acceptance record includes sample identity, raw score, aggregate score,
overall risk, level, reasons, and preprocessing/inference/round-trip/event
latencies. The result must not be blocked solely because a single bonafide
sample was selected; the threshold is not tuned for this acceptance case.

### Synthetic

An actual deterministic Piper-generated WAV from the committed reproducible
evaluation setup follows the same complete path. The final result must come
from model inference and backend policy; it must not use scenario tables,
hardcoded `92`, or file-label-based blocking.

### Mock fallback

With the ML service stopped and the provider explicitly configured to mock,
the existing high-value scenario must still emit:

```text
18 -> 27 -> 43 -> 61 -> 79 -> 92
CRITICAL
BLOCK_ACTION
```

This is a separate acceptance path and does not prove ML integration.

## Testing strategy

Tests are written before production behavior using RED → GREEN → REFACTOR.

Backend unit/component coverage must include:

1. configuration-based provider selection;
2. unchanged mock provider behavior;
3. ML health/unavailable behavior;
4. typed ML response validation;
5. uncalibrated score semantics;
6. score range and invariant checks;
7. 64,600-sample windowing and overlapping hop behavior;
8. median/persistence/hysteresis aggregation;
9. prevention of one-window catastrophic escalation;
10. bounded queue/backpressure;
11. malformed PCM, silence, and insufficient audio;
12. timeout and inference failure;
13. reset/disposal during active inference;
14. absence of fabricated speaker/replay/prosody evidence;
15. event sequence ordering and `LiveRiskEvent` validation;
16. mock backend operation without ML dependencies.

ML service tests must include independent health, request validation, real
detector invocation through the existing abstraction, and invalid-response
handling. Heavy checkpoint tests are integration-marked.

Acceptance tests must execute the real checkpoint for both actual LibriSpeech
and actual Piper audio. They must assert that results came through the ML
service and provider path, not the mock scenario table.

The existing frontend tests and canonical deterministic E2E flow must remain
green. New frontend tests cover only the optional evidence-availability
rendering and genuine ML error state if those changes are needed.

## Documentation updates

Implementation must update backend and ML setup documentation with:

- starting the backend;
- starting the ML service from the isolated ML environment;
- selecting `mock` versus `ml` provider;
- obtaining and checksum-verifying the AASIST checkpoint;
- feeding real acceptance audio;
- the 64,600-sample window and hop;
- aggregation and provisional thresholds;
- score semantics and latency fields;
- error/fallback behavior;
- bounded-backpressure behavior;
- known limitations.

Documentation must explicitly state that JASH-004 does not provide speaker
verification, replay detection, prosody detection, multilingual validation,
telephony validation, or final risk calibration.

## Expected implementation scope

Expected production files are limited to the following concrete areas:

```text
backend/app/config.py
backend/app/api/calls.py
backend/app/services/risk_provider.py
backend/app/services/ml_service_client.py
backend/app/services/audio_ingestion.py
backend/app/services/spoof_aggregation.py
backend/app/services/ml_risk_policy.py
backend/app/models/risk.py
backend/tests/test_ml_service_client.py
backend/tests/test_audio_ingestion.py
backend/tests/test_spoof_aggregation.py
backend/tests/test_ml_risk_provider.py

ml/runtime/contracts.py
ml/runtime/server.py
ml/runtime/inference.py
ml/audio/aasist_windowing.py
tests/ml/runtime/test_service.py
tests/ml/audio/test_aasist_windowing.py
scripts/feed_audio.py

frontend/src/domain/risk.ts
frontend/src/components/console/SecurityConsole.tsx
frontend/src/domain/risk.test.ts
frontend/src/components/console/SecurityConsole.test.tsx
backend/README.md
ml/README.md
docs/integration/JASH-004-ML-RUNTIME.md
```

No frontend visual redesign is permitted. No backend or ML work from later
tasks is included. No weights, audio, datasets, environments, secrets, Twilio,
authentication, databases, microphone capture, or speaker-verification code
will be committed.

## Non-goals

JASH-004 does not implement:

- speaker enrollment or verification;
- replay or prosody detectors;
- final multi-signal risk fusion;
- automatic blocking from real ML evidence;
- Twilio/PSTN/telephony ingestion;
- browser microphone capture;
- OTP or callback workflows;
- databases or authentication;
- calibration or production accuracy claims;
- AASIST retraining or fine-tuning;
- UI redesign.

## Self-review

The design was reviewed against the requested amendments:

- ML service is stateless apart from loaded model state.
- Temporal aggregation and policy are backend-owned.
- Unsupported numeric fields are explicitly unavailable and excluded from ML
  calculations and operator claims.
- AASIST evidence cannot directly produce `BLOCK_ACTION` in JASH-004.
- Mock `92 / CRITICAL / BLOCK_ACTION` behavior is preserved.
- AASIST’s exact 64,600-sample input is not confused with the existing
  2-second chunker.
- No per-window process startup is used.
- Real LibriSpeech and Piper acceptance paths are specified without proxy data.
- The optional event metadata preserves compatibility for existing mock events
  while preventing misleading ML-mode rendering.
- No unresolved `TBD`, `TODO`, or placeholder-style implementation path
  remains.
