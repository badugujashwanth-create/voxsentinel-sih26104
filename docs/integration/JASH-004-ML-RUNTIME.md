# JASH-004 ML runtime integration

JASH-004 connects the existing backend WebSocket to a separate, loopback-only
ML process. The backend remains Torch-free and owns call policy.

## Processes

1. Install and verify AASIST in `ml/.venv` with `python ml/scripts/setup_aasist.py --verify-only`.
2. Start the ML service:

   ```bash
   ml/.venv/bin/uvicorn ml.runtime.server:app --host 127.0.0.1 --port 8010
   ```

3. Start the backend in ML mode:

   ```bash
   VOXSENTINEL_RISK_PROVIDER=ml VOXSENTINEL_ML_SERVICE_URL=http://127.0.0.1:8010 \
     backend/.venv/bin/uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
   ```

The frontend can use its existing live mode against the backend. No browser
microphone, Twilio, persistence, or raw-audio storage is part of this task.

## Real-audio acceptance

Use a naturally long evaluation file (at least 72,600 canonical samples / 4.5375
seconds) from `ml/data/eval/genuine` and `ml/data/eval/synthetic`. Do not pad,
repeat, concatenate, alter, or byte-slice source audio. Each request body must be
one complete independently decodable WAV/FLAC container.

```bash
python scripts/run_ml_acceptance.py --audio ml/data/eval/genuine/<sample>.flac
python scripts/run_ml_acceptance.py --audio ml/data/eval/synthetic/<sample>.wav
```

The feeder posts the complete source container to the backend. The backend
decodes, downmixes, resamples to mono 16 kHz float32, and creates 64,600-sample
windows with an 8,000-sample hop. The ML service validates that exact window and
returns raw uncalibrated spoof evidence only.

## Operational policy

Backend aggregation uses the recent five-window median. Evidence at or above
0.5 for two aggregate observations enters `ELEVATED_AUTHENTICITY_REVIEW` and
emits operational score 70 / HIGH / `REQUIRE_CALLBACK`. Evidence below 0.45 for
two observations demotes to NORMAL, score 20 / LOW / `MONITOR`. These are
severity constants, not calibrated probabilities. JASH-004 real ML never emits
CRITICAL, `REQUIRE_SUPERVISOR`, or `BLOCK_ACTION`.

Speaker, prosody, replay, and unavailable context values are transport-only
`0.0` sentinels with `NOT_EVALUATED` metadata; they are excluded from policy and
must not be read as safe or verified.

## Failure behavior and fallback

Model readiness failure, timeout, malformed service response, malformed audio,
silence, insufficient audio, queue saturation, reset, and WebSocket disconnect
are explicit errors. ML mode never substitutes mock data. For the deterministic
offline judge demo, stop the ML service and set `VOXSENTINEL_RISK_PROVIDER=mock`;
the original 18, 27, 43, 61, 79, 92 -> CRITICAL / `BLOCK_ACTION` sequence remains.

No raw request body, decoded source, or model window is logged or persisted.
