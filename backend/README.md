# Backend

Ownership boundary: Rohan.

FastAPI service backing the VoxSentinel live-call console.

> ## Risk output is MOCK DATA
>
> Every risk score, probability, and reason this service returns comes from a
> hand-written scenario table in `app/services/risk_provider.py`. There is **no
> model, no audio analysis, and no inference of any kind** in this build. Do not
> describe this output as real AI detection in a demo, a report, or a
> submission. The real detector arrives in a later task behind the same
> `RiskProvider` interface.

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
    "High-value transaction context"
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
| `GENUINE` | 12, 10, 14, 11, 13 | `LOW` | `NONE` |
| `HUMAN_IMPOSTOR` | 18, 26, 39, 54, 67, 76 | `HIGH` | `REQUIRE_CALLBACK` |
| `AI_CLONE` | 20, 31, 46, 63, 78, 88 | `CRITICAL` | `REQUIRE_VOICE_CHALLENGE` |
| `HIGH_VALUE_TRANSFER_ATTACK` | 18, 27, 43, 61, 79, 92 | `CRITICAL` | `BLOCK_ACTION` |

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

`RiskProvider` is the seam for real detection. A future `MLRiskProvider`
implements the same interface and is returned from `get_risk_provider()` in
`api/calls.py`; no routing or WebSocket code changes.

Streaming audio windowing lives at repo root in `ml/audio/chunker.py`
(buffering only — no model, no classification).

## Privacy

No raw audio is stored or logged. No voice data is committed to this
repository.
