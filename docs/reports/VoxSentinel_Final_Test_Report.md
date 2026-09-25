# VoxSentinel Final Test Report

Commit under test: `edcb1ee835fc065f9023161bed1abab2e2e8c4cc`

## Suite summary

| Component | Suite | Passed | Failed | Skipped | Result |
|---|---|---:|---:|---:|---|
| Backend | `backend/.venv-jash005-review/Scripts/python.exe -m pytest backend/tests -q` | 147 | 0 | 0 | PASS |
| ML | `ml/.venv-verification/Scripts/python.exe -m pytest tests/ml -q` | 220 | 0 | 14 | PASS with prepared-evaluation skips |
| ML fast | `pytest tests/ml -m "not integration"` | 203 | 0 | 0 | PASS |
| Frontend | `npm.cmd test -- --run` | 65 | 0 | 0 | PASS |
| Frontend lint | `npm.cmd run lint` | — | 0 | — | PASS |
| Frontend build | `npm.cmd run build` | — | 0 | — | PASS |
| DEMO E2E | `judge-demo.spec.ts` | 1 | 0 | 0 | PASS |
| Browser LIVE | `microphone-live.spec.ts` | functional path verified; stale cosine assertion | 1 harness assertion | — | TEST MAINTENANCE NEEDED |

The failed microphone E2E assertion compared a two-decimal UI rendering against a full-precision backend cosine. The product path produced real ECAPA evidence; the assertion requires rounding-aware comparison.

## Controlled LIVE evidence

- Call ID: `52a2f72ec1a6`
- Sample rate: 48 kHz
- Channels: 1
- Browser frames: 1,414 produced and sent
- Browser drops: 0
- Transport gaps: 0
- Fused observations: 50
- AASIST evidence: present
- ECAPA cosine: present, including negative values
- Stop cleanup: tracks, AudioContext, and audio socket inactive
- Console errors: none

## Stress scenario

- Duration: 61.1 seconds
- Frames produced/sent: 2,832 / 2,832
- Drops: 0
- Transport gaps: 0
- Inferences observed: 25
- Cleanup: PASS
- Console errors: none

## Failure-path coverage

Backend and ML unit/integration suites cover malformed VXAF input, finite-value validation, source gaps/overlap, canonicalization, scheduling bounds, duplicate ownership, session lifecycle, telemetry protection, profile isolation, missing/insufficient speaker evidence, model-unavailable handling, and reset/cleanup behavior. No new production failure was found.

## Limitations

The current repository does not expose every backend queue counter in the browser acceptance snapshot, and the separate T1/T2/T3 stage timestamps are not individually exposed. Those values are reported as unavailable rather than inferred.
