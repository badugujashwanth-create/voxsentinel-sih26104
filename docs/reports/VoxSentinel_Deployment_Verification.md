# VoxSentinel Deployment Verification

## Target

Production-like local deployment on frozen `main` at commit `edcb1ee835fc065f9023161bed1abab2e2e8c4cc`.

The repository does not contain a supported cloud deployment configuration; public deployment is therefore **not configured**.

## Services

| Service | URL | Result |
|---|---|---|
| ML runtime | `http://127.0.0.1:8010` | Health PASS; AASIST and ECAPA ready |
| Backend | `http://127.0.0.1:8000` | Health PASS; ML provider configured |
| Frontend | `http://127.0.0.1:5173` | HTTP 200 |

## Runtime verification

- AASIST: loaded and ready.
- ECAPA: loaded and ready.
- Provider: `ml`.
- Controlled browser microphone: reached AudioContext, AudioWorklet, VXAF, backend, AASIST, ECAPA, fusion, and UI.
- Cleanup: tracks, AudioContext, and audio socket inactive after Stop.
- 60-second stress run: 61.1 seconds, 2,832 frames produced/sent, zero browser drops, zero transport gaps, 25 observations, zero console errors.

## Constraints

The existing HTTP live-integration E2E contains a stale exact `LIVE CALL` selector and failed at that assertion before audio verification. The current browser microphone harness independently verified the live path and cleanup. This is a test-harness maintenance issue, not a production runtime failure.
