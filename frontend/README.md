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

## Checks

```bash
npm test
npm run lint
npm run build
npm run test:e2e
```

## Current limitations

Scenario values and verification outcomes are simulated. There is no real model inference, SMS, telephony, authentication, persistence, or backend session management in this task. Voice authenticity risk remains separate from secondary identity verification and protected-action authorization.
