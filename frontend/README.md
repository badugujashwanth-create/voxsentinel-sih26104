# VoxSentinel frontend

The frontend is a React + TypeScript + Vite implementation of the VoxSentinel progressive forensic demo. It is deliberately independent of the backend and starts in deterministic demo mode.

## Install and run

```bash
npm install
npm run dev
```

Choose a scenario, then use the discreet console controls to start, pause, resume, reset, or change it.

## Demo mode

Demo mode is the default and requires no backend:

```text
VITE_DEMO_MODE=true
```

The four deterministic scenarios are Genuine Caller, Human Impostor, AI Voice Clone, and High-Value Transfer Attack. The canonical judge path progresses to 92 / CRITICAL and blocks the protected action before simulated secondary verification.

## Live WebSocket mode

Set `VITE_DEMO_MODE=false` and provide the backend origin with `VITE_API_BASE_URL`. The client connects to `/api/v1/calls/{call_id}/risk-stream` using the same `RiskStreamSource` contract as the mock adapter.

In live mode, starting a selected scenario first creates a backend call with `POST /api/v1/calls`, starts it with `POST /api/v1/calls/{call_id}/start`, then opens the WebSocket with the returned `call_id`. Reset or normal stream completion stops the live call with `POST /api/v1/calls/{call_id}/stop` where the backend session is already live. Live setup failures are shown in the console; there is no automatic fallback to mock data.

Live mode requests microphone permission only after the explicit `Start
analysis` gesture and after the backend risk stream is ready. The browser opens
the separate `/api/v1/calls/{call_id}/audio-stream`, reports the actual
`AudioContext.sampleRate`, and sends bounded VXAF v1 float32 PCM frames from an
AudioWorklet. The backend performs canonical mono 16 kHz conversion and
reuses the same risk stream. `MIC STREAMING` is not shown until `audio_ready`.

Microphone mode is ephemeral. Raw audio is not retained, downloaded, placed in
browser storage, or written to backend logs/files. Browser backpressure drops
new frames above 262144 buffered bytes and resumes below 65536; no unbounded
retry queue is used.

## Checks

```bash
npm test
npm run lint
npm run build
npm run test:e2e
```

## Current limitations

Scenario values and verification outcomes remain simulated in DEMO mode. Real
microphone mode requires the backend and verified ML service. The AASIST score
is an uncalibrated model score, not a probability of fakery. Real microphone
policy remains NORMAL 20/LOW/MONITOR or ELEVATED_AUTHENTICITY_REVIEW
70/HIGH/REQUIRE_CALLBACK; speaker verification remains NOT_EVALUATED.
