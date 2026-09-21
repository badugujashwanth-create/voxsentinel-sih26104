# Backend

JASH-004 keeps this service Torch-free. The default
`VOXSENTINEL_RISK_PROVIDER=mock` preserves the deterministic judge demo.
`VOXSENTINEL_RISK_PROVIDER=ml` requires the separate loopback ML service and
`VOXSENTINEL_ML_SERVICE_URL`; ML failures are surfaced and never replaced with
mock results.

Browser microphone mode additionally requires `soxr==1.1.0` from
`backend/requirements.txt`. The backend owns stateful source-rate conversion,
downmixing, source-frame correlation, and 64,600-sample AASIST windowing.

See `docs/integration/JASH-004-ML-RUNTIME.md` for the real-audio setup.

Ownership boundary: Rohan.

FastAPI service backing the VoxSentinel live-call console.

> ## Provider modes
>
> `mock` returns hand-written scenario data for the deterministic demo. `ml`
> consumes raw AASIST spoof evidence from the separate loopback ML service. ML
> output is uncalibrated evidence, not identity verification, fraud certainty,
> or final risk fusion.

## Setup

```bash
cd backend
python3 -m venv .venv            # or: uv venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or: uv pip install -r requirements.txt
```

## Run

```bash
cd backend
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>

## Test

Run from the **repository root** (the suite spans `backend/tests` and `tests/`):

```bash
backend/.venv/bin/python -m pytest
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOXSENTINEL_RISK_EMIT_INTERVAL_MS` | `700` | Delay between risk events on the socket. |
| `VOXSENTINEL_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated browser origins allowed to call the API. |

Emission pacing is a transport concern only. Each event's `timestamp_ms` comes
from the scenario table, so the event data stays identical however fast it is
emitted.

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe → `{"status": "ok", "service": "voxsentinel-backend"}` |
| `POST` | `/api/v1/calls` | Open a call session (`201`) |
| `GET` | `/api/v1/calls/{call_id}` | Full current session state |
| `POST` | `/api/v1/calls/{call_id}/start` | `CREATED` → `LIVE` |
| `POST` | `/api/v1/calls/{call_id}/stop` | `LIVE` → `COMPLETED` |
| `WS` | `/api/v1/calls/{call_id}/audio-stream` | Versioned browser PCM input |

Create body:

```json
{
  "claimed_identity": "CEO Demo",
  "scenario": "HIGH_VALUE_TRANSFER_ATTACK",
  "transaction_value": 2500000,
  "currency": "INR"
}
```

Sessions are held in memory only and are lost when the process exits. There is
no database and no authentication in this build.

### Lifecycle

States: `CREATED, LIVE, VERIFYING, BLOCKED, COMPLETED, FAILED`.

```
CREATED ──▶ LIVE ──▶ VERIFYING ──▶ BLOCKED ──▶ COMPLETED
   │         │           │            │
   └─────────┴───────────┴────────────┴──────▶ FAILED
```

`COMPLETED` and `FAILED` are terminal. An illegal transition returns `409`; an
unknown call id returns `404`.

## WebSocket

```
WS /api/v1/calls/{call_id}/risk-stream
```

The call must already be `LIVE`. The socket emits one `LiveRiskEvent` per
scenario step, then closes normally.

| Condition | Close code |
| --- | --- |
| Unknown `call_id` | `4404` |
| Call is not `LIVE` | `4409` |

Event shape — field names and ranges mirror `frontend/src/domain/risk.ts`:

```json
{
  "call_id": "abc123",
  "sequence": 6,
  "timestamp_ms": 4200,
  "synthetic_probability": 0.91,
  "speaker_match_score": 0.28,
  "speaker_mismatch_score": 0.72,
  "prosody_anomaly_score": 0.74,
  "replay_risk_score": 0.21,
  "context_risk_score": 0.95,
  "overall_risk_score": 92,
  "risk_level": "CRITICAL",
  "reasons": [
    "Synthetic speech characteristics detected",
    "Speaker identity mismatch",
    "High-value financial request"
  ],
  "recommended_action": "BLOCK_ACTION"
}
```

The six probability fields are `0.0–1.0`; `overall_risk_score` is `0–100`.

`risk_level` bands: `0–29 LOW · 30–59 MEDIUM · 60–79 HIGH · 80–100 CRITICAL`.
The console rejects any event whose `risk_level` disagrees with its
`overall_risk_score`, so the model enforces that invariant before sending.

`recommended_action` is one of `NONE, MONITOR, REQUIRE_OTP, REQUIRE_CALLBACK,
REQUIRE_VOICE_CHALLENGE, REQUIRE_SUPERVISOR, BLOCK_ACTION`.

## Mock scenarios

| Scenario | Progression | Final level | Final action |
| --- | --- | --- | --- |
| `GENUINE` | 12, 10, 14, 11, 13 | `LOW` | `MONITOR` |
| `HUMAN_IMPOSTOR` | 18, 26, 39, 54, 67, 76 | `HIGH` | `MONITOR` |
| `AI_CLONE` | 20, 31, 46, 63, 78, 88 | `CRITICAL` | `BLOCK_ACTION` |
| `HIGH_VALUE_TRANSFER_ATTACK` | 18, 27, 43, 61, 79, 92 | `CRITICAL` | `BLOCK_ACTION` |

These tables mirror `SCENARIOS` in `frontend/src/scenarios/scenarios.ts`
field for field, so the console renders identically whether it replays its
own offline fixtures or streams from this backend. If those fixtures change,
change `app/services/risk_provider.py` to match.

`HUMAN_IMPOSTOR` is the scenario that separates this product from a plain
deepfake detector: synthetic probability stays low the whole way through while
the speaker mismatch climbs, so a real human using someone else's identity is
still caught.

## Layout

```
backend/app/
├── main.py                     app factory, CORS, error mapping
├── config.py                   environment settings
├── api/health.py               health endpoint
├── api/calls.py                call routes + risk WebSocket
├── models/call.py              lifecycle models
├── models/risk.py              risk event contract
├── services/call_service.py    transition rules
├── services/risk_provider.py   RiskProvider + MockRiskProvider
└── state/session_store.py      in-memory sessions
```

`RiskProvider` is the seam shared by `MockRiskProvider` and `MLRiskProvider`.
`get_risk_provider()` selects them from `VOXSENTINEL_RISK_PROVIDER`.

Streaming audio windowing lives at repo root in `ml/audio/chunker.py`
(buffering only — no model, no classification).

The browser path uses one `StreamingAudioCanonicalizer` per active producer;
frames are never resampled independently. It reuses the same `AsyncCallSession`
and bounded AASIST queue as the existing live ML path.

## Privacy

No raw audio is stored or logged. No voice data is committed to this
repository.

The microphone path is ephemeral: no raw PCM is logged, persisted, downloaded,
or stored in a database. Real AASIST output remains an uncalibrated model score;
JASH-005 operational states remain only 20/LOW/MONITOR and
70/HIGH/REQUIRE_CALLBACK. Speaker verification, replay detection, telephony,
automatic reconnect, and physical microphone evidence are outside automated CI
and require the documented acceptance run.
