# JASH-006 Implementation Plan

Base: `9e4f556464b0e934f0ed1c5df66d89c69d93d8ab`
Branch: `feat/jash006-final-intelligence-integration`

Execution is TDD-first. Each task writes focused RED tests, confirms RED, implements the smallest behavior, confirms GREEN, runs nearby regressions, reviews the diff, and commits one logical change. Backend remains Torch-free. No raw live audio is persisted.

## Task 1 — Baseline and contracts

Files: new backend speaker contract/profile/fusion modules and tests; ML runtime contracts/tests; risk model updates. Define typed availability/state/evidence objects, profile ids, model metadata, and optional LiveRiskEvent fields without changing MockRiskProvider. Run backend, ML fast, and frontend baselines. Commit `feat: add speaker evidence and fusion contracts`.

## Task 2 — Extend ML runtime for ECAPA

Files: `ml/runtime/contracts.py`, `ml/runtime/inference.py`, `ml/runtime/server.py`, new runtime speaker adapter/tests, `ml/requirements.txt` only if the existing pinned dependency is insufficient. Reuse `ECAPASpeakerVerifier`, verify the pinned checkpoint, validate mono 16 kHz float32 input, gate ECAPA forwards at one, add readiness and embed/verify endpoints. RED covers malformed, short, unavailable, timeout, similarity range, and model metadata. Run model-free tests and real integration tests when checkpoint is installed. Commit `feat: expose verified ECAPA runtime endpoints`.

## Task 3 — Profile enrollment

Files: backend profile registry/service/routes/tests, ML client methods, documentation. Add in-memory profile creation from complete audio containers; call ECAPA embed; store embedding and provenance only; discard PCM. Test profile isolation, duplicate ids, no-reference, malformed/short audio, and model mismatch. Commit `feat: add ephemeral speaker profile enrollment`.

## Task 4 — Speaker scheduling and shared session

Files: `backend/app/services/async_session.py`, new bounded speaker scheduler, audio stream/session tests. Add canonical-audio speaker accumulation at 32,000 minimum and 16,000 hop, one in-flight request, bounded newest-data policy, source/window metadata, and reset cleanup. Preserve AASIST queue and duplicate-producer behavior. Commit `feat: schedule bounded speaker probes`.

## Task 5 — Backend speaker evidence provider

Files: MLRiskProvider, ML client, evidence/fusion tests. Resolve the call profile once, consume scheduler observations, map failures to explicit availability, and retain uncalibrated cosine/model/latency. Verify no speaker value is fabricated when reference or model is unavailable. Commit `feat: emit live speaker evidence`.

## Task 6 — Deterministic temporal fusion

Files: new `fusion_policy.py`, session state, tests. Implement the approved matrix, two-observation persistence/demotion, explicit reason codes, operational constants, and no score averaging. Test every matrix cell, single-spike stability, reset isolation, missing evidence, and frozen mock/real AASIST policy constraints. Commit `feat: add deterministic spoof-speaker fusion`.

## Task 7 — Live event and telemetry integration

Files: risk model, provider, acceptance telemetry, tests. Add optional speaker/fusion metadata and bounded telemetry while preserving compatibility with existing consumers. Ensure unsupported replay/prosody/context remain NOT_EVALUATED and do not enter fusion. Commit `feat: expose fused live evidence telemetry`.

## Task 8 — Frontend evidence integration

Files: risk domain, LiveCallRiskStreamSource, SecurityConsole, focused tests only. Render speaker state/cosine as uncalibrated similarity, profile identity state, fusion action, and N/A unavailable evidence without redesigning layout. Ensure mock mode remains identical and live selection never silently falls back. Commit `feat: render live speaker and fusion evidence`.

## Task 9 — Lifecycle and degraded paths

Files: calls API, session cleanup, ML client/provider tests and integration tests. Cover missing reference, ECAPA timeout, AASIST unavailable, risk/audio disconnect, Stop, Reset, second session, and duplicate producer. No late event may cross generations; no error may produce demo values. Commit `fix: harden fused session lifecycle`.

## Task 10 — Controlled live acceptance

Files: integration/browser tests and docs only. Create a controlled profile from an approved genuine fixture, run expected-speaker and different-speaker input through getUserMedia → VXAF → backend → both ML detectors → fusion → risk WebSocket → UI. Run no-reference, insufficient, unavailable, reset, and mock cases. Record exact evidence and latency. If clone assets are present, run them; otherwise record CONTROLLED_CLONE_FIXTURE_NOT_AVAILABLE. Commit `test: add JASH-006 live fusion acceptance`.

## Task 11 — Review and final verification

Run backend full pytest; ML full and fast suites; frontend tests/lint/build; mock, JASH-005 live, and JASH-006 controlled E2E; repository hygiene and diff checks. Review `main...HEAD` for semantics, privacy, bounds, lifecycle, and UI wording. Fix Critical/Important findings with RED tests and rerun. Commit documentation/acceptance updates only when required.

## Task 12 — Physical gate and PR

Attempt physical microphone only when honest hardware access exists. If unavailable, record `PHYSICAL_MIC=PENDING_HUMAN` and continue. Verify no tracked weights/audio/secrets, push `feat/jash006-final-intelligence-integration`, and open a PR against main. Do not merge and do not start another engineering task.

## Required commands

Backend: `backend/.venv-jash005/Scripts/python.exe -m pytest backend/tests -q`

ML fast: `ml/.venv-verification/Scripts/python.exe -m pytest tests/ml -m "not integration" -q`

ML full: `ml/.venv-verification/Scripts/python.exe -m pytest tests/ml -q`

Frontend: `npm test -- --run`, `npm run lint`, `npm run build`

E2E: mock, JASH-005 controlled live, and JASH-006 expected/different/degraded flows.

Final state requires `git diff --check`, clean working tree, exact score semantics, bounded resources, and a PR only after automated acceptance. Physical acceptance may remain PENDING_HUMAN.
