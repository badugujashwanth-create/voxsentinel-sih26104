# VoxSentinel Final Test Report

## Current verification record

The following commands were executed in this workspace on the deployment branch. Cloud-provider checks are intentionally separated from local evidence.

| Area | Command | Result |
| --- | --- | --- |
| Frontend unit | `npm test -- --run` | PASS: 66 tests in 19 files |
| Frontend lint | `npm run lint` | PASS |
| Frontend build | `npm run build` | PASS |
| DEMO E2E | `npm run test:e2e -- --project=chromium frontend/e2e/judge-demo.spec.ts` | PASS: 1 test |
| Backend full | `python -m pytest backend/tests -q -rA` | PASS: 147 tests |
| ML full | `python -m pytest tests/ml -q -ra` | PASS: 220 tests; 14 skips because prepared evaluation samples are absent |
| ML fast | `python -m pytest tests/ml -m "not integration" -q -rA` | PASS: 203 tests |
| ML asset verification | `python ml/scripts/setup_aasist.py --verify-only`; `python ml/scripts/setup_ecapa.py --verify-only` | PASS: approved AASIST SHA and ECAPA revision verified |
| Hosted E2E | Vercel/Render public path | NOT VALIDATED YET: no authenticated deployment or public URLs |

The ML full-suite skips are explicit fixture skips, not failures. The local stack loaded AASIST and ECAPA and produced LIVE model evidence through the browser-controlled audio path.

The recorded local inference host was an Intel 12th Gen Core i5-12450H, 15.7 GB RAM, Intel UHD Graphics, Windows 11 Home Single Language 64-bit. This does not establish hosted compute characteristics.

## Regression fixes

- LIVE selectors are scoped to the accessible `Call context` region rather than relying on a page-wide exact-text match.
- ECAPA UI assertions compare the raw value to the documented two-decimal presentation.
- LIVE timer presentation subtracts the first event timestamp, so epoch evidence timestamps cannot render as multi-million-second durations.
- `VITE_API_WS_URL` is supported independently from `VITE_API_BASE_URL` for production WSS routing.

## Required acceptance cases

DEMO is verified. JASH-005 controlled microphone E2E, JASH-006 live fusion E2E, the legacy controlled LIVE E2E, a 68.125-second LIVE soak, second-session cleanup, and local failure-mode checks passed. Hosted LIVE assertions remain **NOT VALIDATED YET** because provider authentication and public URLs are unavailable in this workspace.

## Failure-mode expectations

The code and tests cover malformed VXAF, sequence gaps/overlap, duplicate producer ownership, bounded queues, missing references, unavailable models, stale/out-of-order speaker evidence, disconnects, Stop, Reset, and lifecycle cleanup. Runtime soak evidence: 3,047 frames produced/sent, 0 drops, 0 transport gaps, 50 inference observations, and no browser console errors.

## Integrity statement

No fabricated LIVE model output, hosted URL, latency, FAR/FRR/EER, physical microphone claim, Indian-language claim, telecom claim, or blockchain claim is included in this report.

## Hosted deployment attempt

Vercel `https://voxsentinel.vercel.app`, Render backend `https://voxsentinel-backend.onrender.com`, and Render ML `https://voxsentinel-ml.onrender.com` were created from the deployment branch. Public HTTPS, explicit CORS preflight, and both WSS routes were observed. ML health reported AASIST and ECAPA readiness, and a direct ECAPA embedding request returned the pinned revision. The free ML service then restarted repeatedly during hosted inference; therefore hosted AASIST/ECAPA evidence, hosted latency, hosted soak, and hosted second-session acceptance are **NOT VALIDATED YET**. Local controlled LIVE remains the verified fallback.
