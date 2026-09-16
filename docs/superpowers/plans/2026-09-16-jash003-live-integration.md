# JASH-003 Live Frontend ↔ Backend Integration Plan

## Goal

Connect live mode to the merged backend call lifecycle and risk WebSocket
while preserving the frozen UI, deterministic mock mode, and existing
`RiskStreamSource` contract.

## Baseline

- Base branch: `main`
- Starting SHA: `48d0b33ef5e4490cd2891bf0e0820b44a081eca7`
- Working branch: `feat/jash-live-integration`
- Backend routes: `POST /api/v1/calls`, `POST /api/v1/calls/{call_id}/start`,
  `POST /api/v1/calls/{call_id}/stop`, and
  `WS /api/v1/calls/{call_id}/risk-stream`
- Backend scenario enum values match the frontend scenario IDs.

## Design decisions

1. Keep presentation components unaware of HTTP and WebSocket lifecycle.
2. Add a small `CallSessionClient` for create/start/stop requests using the
   configured backend origin.
3. Extend demo orchestration so live mode creates a session, starts it, then
   constructs the WebSocket source with the returned real `call_id`.
4. Keep `MockRiskStreamSource` as the complete default offline path; it must
   not make HTTP requests.
5. Surface lifecycle failures through the existing stream status/error path so
   live mode cannot appear healthy after a failed create/start/connect.
6. Stop and dispose an existing live source/session during reset, scenario
   replacement, and unmount where appropriate.
7. Avoid introducing UI redesign; only use existing status/error presentation
   hooks if available, adding minimal operator-visible connection copy only if
   the current UI otherwise hides a genuine failure.

## Files expected to change

- `frontend/src/services/call-session/CallSessionClient.ts`
  - Typed create/start/stop client.
  - Base URL resolution and HTTP error normalization.
- `frontend/src/services/call-session/CallSessionClient.test.ts`
  - Request methods, payloads, returned `call_id`, and HTTP failures.
- `frontend/src/hooks/useDemoController.ts`
  - Live session orchestration and real call ID handoff.
  - Preserve mock source creation and cleanup behavior.
- `frontend/src/hooks/useDemoController.test.tsx`
  - Demo no-network behavior, live create/start ordering, failure states, and
    reset cleanup.
- `frontend/src/services/risk-stream/WebSocketRiskStreamSource.ts`
  - Only if needed to expose or normalize the observed 4404/4409/disconnect
    state without changing the event contract.
- Relevant existing stream tests, only where behavior is newly covered.
- `frontend/.env.example` and `frontend/README.md`, only if live lifecycle
  configuration or run instructions need correction.
- `frontend/e2e/live-integration.spec.ts`
  - Backend-backed acceptance flow, if the existing test setup can run the
    backend reliably without changing repository architecture.

No files under `backend/` or `ml/` are in scope.

## TDD sequence

### RED

1. Add client tests proving exact POST paths, JSON payloads, response parsing,
   and failure behavior.
2. Add orchestration tests proving live mode calls create then start before
   WebSocket construction, and uses the returned ID.
3. Add reset/disposal and offline no-fetch tests.
4. Add live acceptance coverage for the backend lifecycle if test harness
   startup is available.

### GREEN

1. Implement the typed HTTP client.
2. Implement live session orchestration outside presentation components.
3. Normalize errors and ensure failed live setup is visible through the
   existing controller state.
4. Keep the mock path unchanged.

### REFACTOR

Review for duplicate URL logic, stale session references, duplicate sockets,
and accidental fixture call IDs in live mode. Keep the diff minimal.

## Verification plan

1. Frontend unit tests, lint, typecheck/build, and existing Playwright flow.
2. Backend regression suite from repository root.
3. Live backend acceptance with a real created call ID and exact risk sequence:
   `18, 27, 43, 61, 79, 92`.
4. Offline acceptance with backend stopped and demo mode enabled.
5. Browser console check and visual smoke check at existing presentation
   sizes; no design changes.
6. Diff/scope audit confirming no backend/ML modifications, secrets, or
   generated artifacts.

## Commit checkpoints

- `feat: add typed live call session client`
- `feat: orchestrate live backend risk sessions`
- `test: cover live frontend backend integration`

Push `feat/jash-live-integration` and open a PR to `main`; do not merge.
