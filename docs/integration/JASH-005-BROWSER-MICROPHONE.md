# JASH-005 browser microphone acceptance

JASH-005 adds a browser-only microphone transport. It is separate from the
existing risk WebSocket:

```text
WS /api/v1/calls/{call_id}/audio-stream
WS /api/v1/calls/{call_id}/risk-stream
```

The browser requests microphone permission only after the operator presses
`Start analysis` in live mode. Demo mode never calls `getUserMedia`, creates an
`AudioContext`, loads the worklet, or opens the audio WebSocket.

## Local processes

Use the clean verified environments documented by JASH-004:

```powershell
ml/.venv/Scripts/python.exe -m uvicorn ml.runtime.server:app --host 127.0.0.1 --port 8010
$env:VOXSENTINEL_RISK_PROVIDER = "ml"
$env:VOXSENTINEL_ML_SERVICE_URL = "http://127.0.0.1:8010"
backend/.venv-jash005/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Run the frontend with `VITE_DEMO_MODE=false` and `VITE_API_BASE_URL=http://127.0.0.1:8000`.

## Transport facts

- AudioContext.sampleRate is the browser/Web Audio processing rate, not a
  guaranteed physical-device rate.
- The worklet emits one 1024-sample/channel transport frame accumulator.
- Frames use VXAF v1, a 32-byte little-endian header, and float32le PCM.
- The backend owns downmixing, stateful `soxr==1.1.0` conversion, and AASIST
  windows.
- Browser watermarks are 262144 bytes high and 65536 bytes low. Frames are
  dropped while congested; there is no retry queue.
- Raw microphone audio is not retained or logged.

## Opt-in browser check

The automated Playwright check validates the browser lifecycle and is never run
by default:

```powershell
$env:VITE_DEMO_MODE = "false"
$env:JASH005_LIVE_MIC_ACCEPTANCE = "1"
cd frontend
npm run test:e2e -- --workers=1 microphone-live.spec.ts
```

For merge evidence, run this with the real backend and verified ML service. A
fake browser device is suitable for transport API checks but does not replace
the required physical-microphone acceptance.

## Physical acceptance record

Speak naturally for at least 4.5375 seconds and preferably long enough for
multiple 64,600-sample AASIST windows. Record browser/version, AudioContext
sample rate, channels, duration, produced/sent/dropped frames, sequence and
source gaps, canonical sample count, generated and dropped AASIST windows, raw
uncalibrated scores, aggregates, policy states, warm-up latency, steady-state
browser-observed latency, and model latency.

Do not tune thresholds based on one microphone session. Real AASIST-only policy
remains 20/LOW/MONITOR or 70/HIGH/REQUIRE_CALLBACK; it never emits CRITICAL,
REQUIRE_SUPERVISOR, or BLOCK_ACTION. ECAPA remains NOT_EVALUATED.
