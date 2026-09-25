# VoxSentinel Cloud Deployment Verification

## Status

Cloud services were created and authenticated from `deploy/vercel-render` at commit `d6410e8`.

- Vercel: `https://voxsentinel.vercel.app` (production build READY)
- Render backend: `https://voxsentinel-backend.onrender.com` (`/health` 200)
- Render ML: `https://voxsentinel-ml.onrender.com` (`/health` has reported AASIST and ECAPA ready)

Hosted LIVE is **BLOCKED on the free Render ML runtime**. Public WSS routing and backend startup were observed, but real hosted inference caused repeated ML process restarts/no-port intervals and subsequent 502/unavailable responses. No hosted LIVE model evidence or hosted latency is claimed.

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

## Verification performed

ML `/health` returned `ready=true`, `speaker_ready=true`, AASIST `AASIST/AASIST.pth@ASVspoof2019-LA`, and ECAPA revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`; a direct hosted ECAPA embed returned HTTP 200. Backend `/health` returned HTTP 200, explicit CORS returned HTTP 200 preflight, and the public browser accepted both `wss://` risk and audio sockets. Hosted LIVE did not produce a stable real inference stream because the free ML service restarted during inference.

## Free-plan risk and cold start

Free Render services create `SIH_DEMO_COLD_START_RISK`. Because free services cannot receive private network traffic, the backend must use the ML service's public HTTPS URL. Warm both Render services 15 minutes before judging, run one health request and one controlled inference, open Vercel, verify WSS, then run a short LIVE acceptance before the presentation. If the ML service OOMs, hosted ML is blocked on the free plan and must not be represented as ready.

## Not validated yet

Hosted first-fused latency, hosted p50/p95, hosted 60-second inference soak, stable AASIST/ECAPA evidence, second-session hosted LIVE, and hosted action evidence are **NOT VALIDATED YET**. The hosted run artifacts contain failure diagnostics only and must not be presented as model evidence. Use the verified local controlled LIVE video as fallback.
