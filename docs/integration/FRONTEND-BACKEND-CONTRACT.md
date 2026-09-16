# VoxSentinel Frontend–Backend Integration Contract

Status: contract extracted from the merged frontend on `main` (`e5d4c2f`).

This document records what `WebSocketRiskStreamSource` currently expects from
ROHAN-001. It does not define backend implementation details that are absent
from the frontend.

## 1. Integration boundary

The frontend selects one call ID and creates a `WebSocketRiskStreamSource`.
The source implements the same `RiskStreamSource` interface as the offline
mock source:

```typescript
interface RiskStreamSource {
  start(handlers: RiskStreamHandlers): Promise<void>;
  stop(): Promise<void>;
  pause(): void;
  resume(): void;
  reset(): void;
  dispose(): void;
}

interface RiskStreamHandlers {
  onEvent(event: LiveRiskEvent): void;
  onStatusChange(status: RiskStreamStatus): void;
  onError(error: Error): void;
}
```

The frontend currently has no REST client for call creation, call start, or
call stop. Those operations are therefore a ROHAN-001 backend/API design
dependency, not an already-consumed frontend contract.

## 2. Backend base URL

Configuration is supplied through the Vite variable:

```text
VITE_API_BASE_URL=
```

When `VITE_DEMO_MODE=false`, the WebSocket adapter uses this value as the
backend origin. If it is empty or absent, it uses the browser's current
`window.location.origin`.

Expected examples:

```text
https://api.example.test
http://127.0.0.1:8000
```

The adapter converts `https:` to `wss:` and every other URL protocol used by
the current implementation to `ws:`. It removes the base URL query string
and appends the stream path. A deployment path prefix is retained.

## 3. Call lifecycle

### Call creation

No frontend call-creation request currently exists. ROHAN-001 may expose a
call-creation endpoint, but the frontend will not call it until a reviewed
contract adds that behavior. The integration must provide a resulting
`call_id` to the stream adapter.

### Call start

No separate frontend HTTP call-start request currently exists. Calling
`RiskStreamSource.start(handlers)` is the current frontend start operation:

1. reset the adapter sequence cursor to `0`;
2. emit `CONNECTING` through `onStatusChange`;
3. open the WebSocket;
4. emit `LIVE` when the socket opens;
5. deliver validated risk events as messages arrive.

`start()` returns after the socket has been constructed, not after the
backend has accepted or acknowledged a call. The backend should begin sending
risk events after the WebSocket connection is established.

### Call stop

No separate frontend HTTP call-stop request currently exists. Calling
`stop()` closes the WebSocket, clears the local socket reference and sequence
cursor, and reports `DISCONNECTED`. The browser close callback may report the
same status again; consumers must treat status notifications as idempotent.

### Call identity requirement

The frontend constructs the stream with a `call_id` and URL-encodes it in the
path. Backend events must send the same `call_id`. The current validator checks
that the event contains a non-empty string but does not compare it to the
constructor call ID; ROHAN-001 must therefore enforce call/session consistency
server-side.

## 4. WebSocket URL

The exact target path is:

```text
/api/v1/calls/{call_id}/risk-stream
```

Examples after protocol conversion:

```text
wss://api.example.test/api/v1/calls/call-17/risk-stream
ws://127.0.0.1:8000/api/v1/calls/call-17/risk-stream
```

The backend should send each risk event as a JSON text WebSocket message.
The browser adapter also accepts an object payload in tests, but JSON text is
the interoperable wire format for a backend WebSocket implementation.

No server command envelope, acknowledgement envelope, heartbeat schema, or
binary audio protocol is currently defined by this frontend.

## 5. `LiveRiskEvent` wire schema

Every message must contain all fields below. JSON field names are case
sensitive and must remain unchanged.

```typescript
interface LiveRiskEvent {
  call_id: string;
  sequence: number;
  timestamp_ms: number;

  synthetic_probability: number;
  speaker_match_score: number;
  speaker_mismatch_score: number;
  prosody_anomaly_score: number;
  replay_risk_score: number;
  context_risk_score: number;

  overall_risk_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  reasons: string[];
  recommended_action:
    | "NONE"
    | "MONITOR"
    | "REQUIRE_OTP"
    | "REQUIRE_CALLBACK"
    | "REQUIRE_VOICE_CHALLENGE"
    | "REQUIRE_SUPERVISOR"
    | "BLOCK_ACTION";
}
```

Validation rules currently enforced by the frontend:

- `call_id` is a non-empty string.
- `sequence` is a positive integer and must strictly increase for each
  delivered event on the current adapter instance.
- `timestamp_ms` is a finite non-negative number.
- Every probability/score field ending in `_probability` or `_score` except
  `overall_risk_score` is a finite number in the inclusive range `0.0–1.0`.
- `overall_risk_score` is a finite number in the inclusive range `0–100`.
- `risk_level` must match the score bands:
  - `0–29`: `LOW`
  - `30–59`: `MEDIUM`
  - `60–79`: `HIGH`
  - `80–100`: `CRITICAL`
- `reasons` must be an array of non-empty strings.
- `recommended_action` must be one of the enum values above.

Malformed or out-of-order events are not delivered to `onEvent`.

## 6. Status and error behavior

`RiskStreamStatus` values are:

```text
IDLE
CONNECTING
LIVE
PAUSED
DISCONNECTED
ERROR
```

Behavior currently exposed to the UI:

- Socket construction emits `CONNECTING`.
- WebSocket `open` emits `LIVE`.
- WebSocket `error` emits `ERROR` and calls `onError` with
  `Risk stream connection failed`.
- Invalid JSON, invalid schema, or non-increasing sequence emits `ERROR` and
  calls `onError` with the validation error.
- WebSocket `close` emits `DISCONNECTED`.
- `pause()` stops frontend delivery but keeps the socket open; it emits
  `PAUSED`.
- `resume()` re-enables delivery and emits `LIVE`.
- `reset()` clears sequence state and emits `IDLE`; it does not reconnect.
- `dispose()` closes the socket and releases handlers; it does not reconnect.

The adapter does not currently implement automatic reconnect, backoff,
heartbeat handling, server-side error-message parsing, or event replay.

## 7. Ordering, reconnect, and retry checklist

ROHAN-001 should verify:

- one stream is scoped to one `call_id`;
- event sequences are strictly increasing;
- timestamps are non-negative milliseconds;
- the server closes or reports an error for an invalid session rather than
  silently changing call identity;
- a reconnect starts a fresh frontend adapter or follows an explicitly
  reviewed reset/replay contract;
- replaying an old sequence is not treated as a new event;
- normal close and error close are distinguishable in backend observability,
  even though the current UI receives `ERROR` and/or `DISCONNECTED` statuses;
- no assumption is made that `start()` means backend analysis is already
  available before the first event.

## 8. Demo-mode fallback

Demo mode is selected by the frontend when:

```text
VITE_DEMO_MODE is absent or exactly "true" → MockRiskStreamSource
VITE_DEMO_MODE is exactly "false"             → WebSocketRiskStreamSource
```

Demo mode is deterministic and does not require a backend. It is the default
offline path and is visibly identified in the product as simulation mode.

There is no automatic fallback from a failed WebSocket connection to the mock
source. If live mode is selected and the backend is unavailable, the adapter
reports an error; the operator must select demo mode/configuration through the
normal reviewed application flow.

## 9. ROHAN-001 integration checklist

- [ ] Provide a reviewed mechanism that supplies a valid `call_id` to the
  frontend.
- [ ] Confirm whether call creation, start, and stop REST endpoints are needed;
  none are currently consumed by the frontend.
- [ ] Implement `GET/upgrade` handling for
  `/api/v1/calls/{call_id}/risk-stream` as a WebSocket endpoint.
- [ ] Send JSON text messages matching the exact `LiveRiskEvent` schema.
- [ ] Preserve snake_case field names and enum spellings.
- [ ] Emit strictly increasing positive `sequence` values per call stream.
- [ ] Emit score values in the frontend’s documented ranges.
- [ ] Ensure `risk_level` matches `overall_risk_score` exactly.
- [ ] Provide non-empty explanatory `reasons` and a valid
  `recommended_action`.
- [ ] Keep event `call_id` equal to the path/session call ID.
- [ ] Define backend behavior for unauthorized, unknown, expired, and stopped
  call IDs.
- [ ] Define normal-close and error-close observability.
- [ ] Do not require binary audio frames, server envelopes, or heartbeat
  messages unless a future frontend contract is reviewed and updated.
- [ ] Test malformed payloads, duplicate sequences, reconnects, and clean
  disconnects against the adapter behavior.
- [ ] Keep `VITE_DEMO_MODE=true` usable when the backend is unavailable.
- [ ] Keep real credentials and environment-specific origins out of source
  control.

## 10. Current contract gaps for review

These are intentionally documented, not implemented in this task:

1. Call creation/start/stop API shapes are undefined because the frontend has
   no client calls for them.
2. Automatic reconnect and retry policy are undefined and absent in the
   adapter.
3. Authentication/authorization for the WebSocket is undefined.
4. Server-side error payloads, heartbeats, replay, and stream completion are
   undefined.
5. The frontend does not compare event `call_id` to the URL call ID; backend
   session validation is required until that contract is revised.

Any change to these behaviors should be added as a reviewed contract change
before frontend or backend implementation diverges.
