# VoxSentinel Deployment Runbook

## Local fallback

1. Install backend dependencies from `backend/requirements.txt` and ML dependencies from `ml/requirements.txt` using separate environments.
2. Fetch models only through `python ml/scripts/setup_aasist.py` and `python ml/scripts/setup_ecapa.py`; both scripts checksum-verify and refuse mismatches.
3. Start ML on 8010 with `uvicorn runtime.server:app --host 127.0.0.1 --port 8010`.
4. Start backend on 8000 with `VOXSENTINEL_RISK_PROVIDER=ml`, `VOXSENTINEL_ML_SERVICE_URL=http://127.0.0.1:8010`, and `uvicorn app.main:app --host 127.0.0.1 --port 8000`.
5. Start frontend with `VITE_DEMO_MODE=false`, `VITE_API_BASE_URL=http://127.0.0.1:8000`, and `VITE_API_WS_URL=ws://127.0.0.1:8000`.
6. Run `scripts/pre_demo_check.ps1`. It must print `VOXSENTIN DEMO READY` before a LIVE presentation.

## Render

Use the Blueprint in `render.yaml`. Create the ML service first, confirm `/health` with both models ready, then set the backend ML service URL and Vercel origin in Render environment settings. Do not commit secrets or URLs that belong to a private Render service.

## Vercel

Create a project from the deployment branch, keep the repository root as the project root, and set `VITE_DEMO_MODE=false`, `VITE_API_BASE_URL=https://<render-backend>`, and `VITE_API_WS_URL=wss://<render-backend>`. Deploy only after `npm run build` passes.

## Warmup and acceptance

Before judging: health-check backend and ML, run one real inference, open the Vercel URL, inspect browser console, run DEMO, then run a short controlled LIVE. Use the exact acceptance sequence in `docs/demo/SIH_Final_Demo_Runbook.md`.
