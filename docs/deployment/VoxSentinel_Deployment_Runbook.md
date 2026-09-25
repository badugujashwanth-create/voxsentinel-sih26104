# VoxSentinel Deployment Runbook

This runbook describes the repository-supported local deployment for the frozen SIH prototype. No public cloud deployment is configured in this repository.

## Prerequisites

- Windows PowerShell or an equivalent shell.
- Python environments already provisioned for the backend and ML runtime.
- Node.js and npm for the frontend.
- The verified AASIST checkpoint in the ML runtime location.
- ECAPA runtime assets available under `ml/speaker/vendor/ecapa`.

The backend is Torch-free. Model inference runs in the separate ML service.

## Services and ports

| Service | Bind | Health |
|---|---|---|
| ML runtime | `127.0.0.1:8010` | `GET /health` |
| Backend | `127.0.0.1:8000` | `GET /health` |
| Vite frontend | `127.0.0.1:5173` | HTTP root |

## Environment

Backend:

```powershell
$env:VOXSENTINEL_RISK_PROVIDER = "ml"
$env:VOXSENTINEL_ML_SERVICE_URL = "http://127.0.0.1:8010"
$env:VOXSENTINEL_ACCEPTANCE_TELEMETRY = "1" # local acceptance only
$env:VOXSENTINEL_ACCEPTANCE_TELEMETRY_TOKEN = "<local-token>" # never commit
```

Frontend:

```powershell
$env:VITE_DEMO_MODE = "false"
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
$env:VITE_ACCEPTANCE_TELEMETRY = "1" # local acceptance only
```

## Startup order

```powershell
cd C:\path\to\voxsentinel-sih26104
ml\.venv-verification\Scripts\python.exe -m uvicorn ml.runtime.server:app --host 127.0.0.1 --port 8010
backend\.venv-jash005-review\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
cd frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Run each service in its own terminal/process. Verify the ML health response reports both AASIST and speaker readiness before starting the backend.

## Health checks

```powershell
Invoke-RestMethod http://127.0.0.1:8010/health
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:5173 -UseBasicParsing
```

The backend must run with `VOXSENTINEL_RISK_PROVIDER=ml`; otherwise the frontend uses the deterministic demo provider.

## Shutdown and recovery

Stop only confirmed VoxSentinel processes. Inspect listeners before terminating a process:

```powershell
Get-NetTCPConnection -LocalPort 8010,8000,5173 -State Listen
Get-CimInstance Win32_Process -Filter "ProcessId=<PID>" | Select-Object ProcessId,Name,CommandLine
```

If a service fails, inspect its redirected log, verify the port is free, and restart in the documented order. Do not switch ports silently because the frontend API and WebSocket origins must remain aligned.

## Model placement

The AASIST definition/checkpoint and ECAPA assets are runtime dependencies and are not committed to Git. The AASIST setup script verifies the pinned checkpoint hash before installation. Do not commit weights or model caches.

## Clean redeployment

1. Stop only confirmed VoxSentinel ML/backend/frontend processes.
2. Verify ports 8010, 8000, and 5173 are free.
3. Verify model assets and their hashes.
4. Start ML, backend, and frontend in that order.
5. Run the health checks.
6. Run the controlled LIVE smoke test.

## Known constraints

- Sessions and speaker profiles are process-local and in-memory.
- There is no public deployment configuration in this repository.
- Acceptance telemetry is local, bounded metadata and requires an explicit token.
- Raw microphone audio is not persisted or logged.
