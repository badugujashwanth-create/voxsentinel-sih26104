# VoxSentinel Cloud Deployment Verification

## Status

Cloud deployment is **PENDING_HUMAN**. The repository now contains reproducible Vercel and Render configuration, but this workspace has no authenticated Vercel CLI, no Render CLI/API token, and no provider-created public URLs. No cloud deployment is claimed.

## Target topology

```text
Vercel React/Vite frontend
  HTTPS + WSS
Render backend (FastAPI, public)
  private service URL where supported
Render ML (AASIST + ECAPA)
```

## Configuration

- `vercel.json` installs and builds `frontend`, publishes `frontend/dist`, and rewrites SPA routes.
- `render.yaml` defines `voxsentinel-ml` and `voxsentinel-backend`.
- Both services now request Render's `free` web-service plan. This avoids a paid Blueprint requirement, but free services have 512 MB RAM, spin down after inactivity, cannot use private networking, and may OOM while loading both PyTorch models. Hosted ML readiness must prove the models actually loaded.
- ML build installs CPU Torch, installs `ml/requirements.render.txt`, runs both checksum-verifying model setup scripts, and starts on `$PORT`.
- Backend starts on `0.0.0.0:$PORT`, uses `VOXSENTINEL_RISK_PROVIDER=ml`, and receives `VOXSENTINEL_ML_SERVICE_URL` and `VOXSENTINEL_CORS_ORIGINS` as deployment environment values.
- Frontend values are `VITE_API_BASE_URL=https://<backend>` and `VITE_API_WS_URL=wss://<backend>`; no production request may use localhost or `ws://`.

## Required provider verification

After human provider setup, verify ML `/health` returns `ready=true`, `speaker_ready=true`, the expected AASIST checkpoint is loaded, and ECAPA reports the pinned revision. Verify backend `/health`, backend-to-ML inference, Vercel HTTPS, audio WSS, risk WSS, explicit CORS, clean browser console, DEMO, LIVE, Stop, Reset, and second session.

## Free-plan risk and cold start

Free Render services create `SIH_DEMO_COLD_START_RISK`. Because free services cannot receive private network traffic, the backend must use the ML service's public HTTPS URL. Warm both Render services 15 minutes before judging, run one health request and one controlled inference, open Vercel, verify WSS, then run a short LIVE acceptance before the presentation. If the ML service OOMs, hosted ML is blocked on the free plan and must not be represented as ready.

## Not validated yet

Public URLs, hosted first-fused latency, hosted p50/p95, hosted 60-second soak, TLS/CORS from a public browser, provider memory behavior, and hosted screenshots/video are all **NOT VALIDATED YET**.
