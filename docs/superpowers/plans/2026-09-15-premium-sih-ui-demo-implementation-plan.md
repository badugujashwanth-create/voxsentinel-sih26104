# JASH-002 Premium SIH UI & Judge Demo Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, judge-ready React security console on `feat/person2-ui-demo` that demonstrates risk escalation, explainability, prevention, and secondary-channel resolution without backend or ML dependencies.

**Architecture:** A typed domain layer owns risk boundaries, event validation, three-state security semantics, scenario fixtures, and verification transitions. A `RiskStreamSource` service boundary hides timers and WebSockets from React; hooks/controllers translate source callbacks into UI state; focused console components render the same experience for mock and future live modes.

**Tech Stack:** React, TypeScript, Vite, CSS modules or scoped CSS, Framer Motion for restrained transitions, Recharts for the risk timeline, Lucide React for semantic icons, Vitest, React Testing Library, and Playwright.

**Spec:** `docs/superpowers/specs/2026-09-15-premium-sih-ui-demo-design.md`

## Global Constraints

- Work only on `feat/person2-ui-demo`; do not modify `main`, `backend/`, or `ml/`.
- Keep `VITE_DEMO_MODE=true` fully offline and deterministic.
- Preserve the exact `LiveRiskEvent` field names and enum values from the spec.
- Keep voice authenticity risk, claimed-identity verification, and protected-action authorization as separate state values.
- Keep claimed identity/request, live state, overall risk, major evidence, and current action visible above the fold at 1920×1080 and 1366×768.
- Do not create timer logic in React presentation components.
- Do not commit `.env`, generated output, `node_modules`, secrets, models, or backend/ML changes.
- Use inclusive risk boundaries: 0–29 LOW, 30–59 MEDIUM, 60–79 HIGH, 80–100 CRITICAL.

---

### Task 1: Scaffold the frontend toolchain

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/app/App.tsx`
- Create: `frontend/src/styles/global.css`
- Create: `frontend/.env.example`
- Create: `frontend/tests/setup.ts`
- Create: `frontend/playwright.config.ts`

**Interfaces:**
- Produces a runnable Vite entrypoint and test/build scripts for later tasks.

- [ ] **Step 1: Write the failing scaffold smoke test**

Create `frontend/src/app/App.test.tsx` with:

```tsx
import { render, screen } from "@testing-library/react";
import { App } from "./App";

it("renders the VoxSentinel launcher", () => {
  render(<App />);
  expect(screen.getByRole("heading", { name: /VoxSentinel/i })).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `cd frontend; npm test -- --run src/app/App.test.tsx`

Expected: FAIL because the Vite/React test scaffold does not exist yet.

- [ ] **Step 3: Add the minimal Vite and Vitest scaffold**

Use scripts `dev`, `build`, `lint`, `test`, and `test:e2e`. Configure Vitest with `jsdom`, `frontend/tests/setup.ts`, and the React Testing Library jest-dom matcher. Render a temporary launcher heading from `App.tsx`; use `main.tsx` to mount it.

- [ ] **Step 4: Run the focused test and build**

Run: `cd frontend; npm test -- --run src/app/App.test.tsx; npm run build`

Expected: PASS and a successful Vite production build.

- [ ] **Step 5: Commit**

```bash
git add frontend
git commit -m "feat: scaffold VoxSentinel demo frontend"
```

### Task 2: Define risk domain semantics and verification state

**Files:**
- Create: `frontend/src/domain/risk.ts`
- Create: `frontend/src/domain/risk.test.ts`
- Create: `frontend/src/domain/verification.ts`
- Create: `frontend/src/domain/verification.test.ts`
- Create: `frontend/src/domain/call.ts`

**Interfaces:**
- Produces `RiskLevel`, `RecommendedAction`, `LiveRiskEvent`, `RiskStreamStatus`, `getRiskLevel(score: number): RiskLevel`, `validateLiveRiskEvent(value: unknown): LiveRiskEvent`, and verification transitions consumed by sources and hooks.

- [ ] **Step 1: Write failing risk-boundary and event-validation tests**

```ts
it.each([[29, "LOW"], [30, "MEDIUM"], [59, "MEDIUM"], [60, "HIGH"], [79, "HIGH"], [80, "CRITICAL"], [100, "CRITICAL"]])(
  "maps %i to %s",
  (score, level) => expect(getRiskLevel(score)).toBe(level),
);

it("rejects an event with a probability outside 0..1", () => {
  expect(() => validateLiveRiskEvent({ synthetic_probability: 2 })).toThrow();
});
```

Add verification tests for `idle → pending → verified`, `pending → failed`, and reset returning to idle without changing the last voice-risk event.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm test -- --run src/domain/risk.test.ts src/domain/verification.test.ts`

Expected: FAIL because the domain functions and types are absent.

- [ ] **Step 3: Implement the smallest typed domain layer**

Implement the exact event contract from the spec, inclusive boundary mapping, numeric/range/enum validation, and a verification state reducer whose actions are `REQUEST`, `SENT`, `START`, `VERIFY`, `FAIL`, and `RESET`. Keep the voice event outside the verification reducer.

- [ ] **Step 4: Run tests and typecheck**

Run: `cd frontend; npm test -- --run src/domain; npx tsc --noEmit`

Expected: PASS with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/domain
git commit -m "feat: define risk and verification domain states"
```

### Task 3: Add deterministic scenarios and mock stream source

**Files:**
- Create: `frontend/src/scenarios/scenarios.ts`
- Create: `frontend/src/scenarios/scenarios.test.ts`
- Create: `frontend/src/services/risk-stream/RiskStreamSource.ts`
- Create: `frontend/src/services/risk-stream/MockRiskStreamSource.ts`
- Create: `frontend/src/services/risk-stream/MockRiskStreamSource.test.ts`

**Interfaces:**
- Consumes: domain event types and validation from Task 2.
- Produces: `ScenarioId`, `ScenarioDefinition`, `SCENARIOS`, and a `MockRiskStreamSource` implementing `RiskStreamSource` with `start`, `stop`, `pause`, `resume`, `reset`, and `dispose`.

- [ ] **Step 1: Write failing fixture and source tests**

Test exact risk arrays for all four scenarios, final high-value event `92 / 100`, final `CRITICAL`, final `BLOCK_ACTION`, and event sequence `[1, 2, ...]`. Use Vitest fake timers to assert `start()` emits the first event, `pause()` stops emissions, `resume()` continues, and `dispose()` clears scheduling.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm test -- --run src/scenarios src/services/risk-stream/MockRiskStreamSource.test.ts`

Expected: FAIL because fixtures and source are absent.

- [ ] **Step 3: Implement deterministic fixture data and source**

Store scenario metadata and event arrays in `scenarios.ts`. The source owns its interval handle and cursor, emits validated events in order, reports `CONNECTING`, `LIVE`, `PAUSED`, `DISCONNECTED`, and `IDLE`, and clears all listeners/timers in `dispose()`. Components receive only handlers, never timer callbacks.

- [ ] **Step 4: Run source tests**

Run: `cd frontend; npm test -- --run src/scenarios src/services/risk-stream/MockRiskStreamSource.test.ts`

Expected: PASS with deterministic output for every scenario.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/scenarios frontend/src/services/risk-stream
git commit -m "feat: add deterministic SIH demo scenarios"
```

### Task 4: Add the validated WebSocket source

**Files:**
- Create: `frontend/src/services/risk-stream/WebSocketRiskStreamSource.ts`
- Create: `frontend/src/services/risk-stream/WebSocketRiskStreamSource.test.ts`

**Interfaces:**
- Consumes: `RiskStreamSource`, event validation, and `VITE_API_BASE_URL`.
- Produces: a source that connects to `/api/v1/calls/{call_id}/risk-stream`, delivers validated events, reports errors/disconnects, and supports reset/disposal.

- [ ] **Step 1: Write failing WebSocket adapter tests**

Mock `WebSocket` and test URL construction, `onmessage` event delivery, out-of-order rejection, malformed JSON error state, `onclose` disconnect state, `stop()` close behavior, and `dispose()` listener cleanup.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm test -- --run src/services/risk-stream/WebSocketRiskStreamSource.test.ts`

Expected: FAIL because the adapter is absent.

- [ ] **Step 3: Implement the adapter**

Resolve `VITE_API_BASE_URL` from `import.meta.env`, normalize trailing slashes, append `/api/v1/calls/${callId}/risk-stream`, parse messages, validate events, enforce increasing sequence, and route failures through `onError` plus `ERROR` status. Do not add a fallback to mock mode inside this class.

- [ ] **Step 4: Run adapter tests and typecheck**

Run: `cd frontend; npm test -- --run src/services/risk-stream/WebSocketRiskStreamSource.test.ts; npx tsc --noEmit`

Expected: PASS with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/services/risk-stream/WebSocketRiskStreamSource.ts frontend/src/services/risk-stream/WebSocketRiskStreamSource.test.ts
git commit -m "feat: add future WebSocket risk stream adapter"
```

### Task 5: Build the demo controller and state-preserving stream hook

**Files:**
- Create: `frontend/src/hooks/useRiskStream.ts`
- Create: `frontend/src/hooks/useDemoController.ts`
- Create: `frontend/src/hooks/useRiskStream.test.ts`
- Create: `frontend/src/hooks/useDemoController.test.ts`
- Modify: `frontend/src/app/App.tsx`

**Interfaces:**
- Consumes: both stream implementations and the domain verification reducer.
- Produces: controller state containing `voiceRisk`, `identityVerification`, `protectedAction`, `streamStatus`, current event, ordered timeline, selected scenario, and controls.

- [ ] **Step 1: Write failing hook tests**

Test selecting a scenario, starting to `LIVE`, applying events in order, reaching CRITICAL/blocked, resetting, and completing callback verification while asserting the current CRITICAL event and reasons remain unchanged. Test `VITE_DEMO_MODE=true` chooses mock source and the alternate mode chooses WebSocket source.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm test -- --run src/hooks`

Expected: FAIL because the hooks and controller state are absent.

- [ ] **Step 3: Implement the hooks**

Create the source once per selected call, subscribe through handlers, reduce events into ordered timeline state, derive action state without mutating voice risk, and dispose the source in effect cleanup. Provide `start`, `pause`, `resume`, `reset`, `selectScenario`, `beginVerification`, `completeVerification`, and `failVerification` actions.

- [ ] **Step 4: Run tests and typecheck**

Run: `cd frontend; npm test -- --run src/hooks; npx tsc --noEmit`

Expected: PASS with the three security states demonstrably separate.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks frontend/src/app/App.tsx
git commit -m "feat: connect demo controller to risk streams"
```

### Task 6: Implement the launcher and above-the-fold console shell

**Files:**
- Create: `frontend/src/components/demo/ScenarioLauncher.tsx`
- Create: `frontend/src/components/demo/OperatorDrawer.tsx`
- Create: `frontend/src/components/call/CallIdentityRail.tsx`
- Create: `frontend/src/components/call/CallStatusBar.tsx`
- Create: `frontend/src/components/risk/RiskWorkspace.tsx`
- Create: `frontend/src/components/risk/LiveWaveform.tsx`
- Create: `frontend/src/components/evidence/EvidenceRail.tsx`
- Create: `frontend/src/components/risk/ProtectionState.tsx`
- Modify: `frontend/src/app/App.tsx`
- Modify: `frontend/src/styles/global.css`

**Interfaces:**
- Consumes: controller state and actions from Task 5.
- Produces: launcher-to-console navigation with the five critical information groups visible without scrolling.

- [ ] **Step 1: Write failing component tests**

Assert launcher buttons for all four scenarios, a persistent `DEMO MODE` indicator, `LIVE` state after start, caller/request/risk/evidence/protection labels, and operator controls hidden from the primary hierarchy until the drawer opens.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm test -- --run src/components`

Expected: FAIL because the console components are absent.

- [ ] **Step 3: Implement the visual shell**

Use a semantic layout with left context rail, flexible center workspace, right evidence rail, and a compact lower band. Keep each metric in the evidence rail as a row/divider treatment rather than independent floating cards. Build the waveform procedurally from deterministic local data and label it `LIVE AUDIO ANALYSIS`; it must not imply real ML. Use CSS `min-height`/grid constraints and responsive compression so identity, live state, risk, evidence, and protection remain above the fold at both required viewport sizes.

- [ ] **Step 4: Run component tests and production build**

Run: `cd frontend; npm test -- --run src/components; npm run build`

Expected: PASS and a successful production build.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app frontend/src/components frontend/src/styles
git commit -m "feat: add premium live security console shell"
```

### Task 7: Add evidence explanation, timeline, prevention, and verification flows

**Files:**
- Create: `frontend/src/components/evidence/ReasonList.tsx`
- Create: `frontend/src/components/risk/RiskTimeline.tsx`
- Create: `frontend/src/components/verification/VerificationDrawer.tsx`
- Create: `frontend/src/components/verification/VerificationMethod.tsx`
- Create: `frontend/src/components/verification/VerificationDrawer.test.tsx`
- Modify: `frontend/src/components/risk/ProtectionState.tsx`
- Modify: `frontend/src/app/App.tsx`
- Modify: `frontend/src/styles/global.css`

**Interfaces:**
- Consumes: ordered events and three-state controller output from Task 5.
- Produces: dynamic reasons, current-call chart, blocked transfer state, four simulated verification methods, and state-preserving resolution.

- [ ] **Step 1: Write failing verification and rendering tests**

Test that CRITICAL transfer state renders `TRANSFER TEMPORARILY BLOCKED`, reasons include synthetic speech/speaker mismatch/high-value context, callback progresses requested → in progress → verified, and final copy includes `IDENTITY VERIFIED THROUGH SECONDARY CHANNEL` while still rendering `Voice Risk: CRITICAL` and the original reasons. Test failed verification leaves the action blocked.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend; npm test -- --run src/components/verification/VerificationDrawer.test.tsx`

Expected: FAIL because the verification drawer and integrated prevention state are absent.

- [ ] **Step 3: Implement the focused interaction surfaces**

Use Recharts or an equivalent small SVG chart for the current-call timeline, with threshold regions and readable labels. Render reasons as an updating list. Implement OTP, Voice Challenge with phrase `BLUE 47 MANGO`, Verified Callback, and Supervisor Approval through the existing lightweight state reducer; do not add role-management or external workflow infrastructure.

- [ ] **Step 4: Run component tests and typecheck**

Run: `cd frontend; npm test -- --run src/components; npx tsc --noEmit`

Expected: PASS with voice risk preserved through every verification outcome.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components frontend/src/app frontend/src/styles
git commit -m "feat: add prevention and verification experience"
```

### Task 8: Add the canonical Playwright acceptance flow and frontend documentation

**Files:**
- Create: `frontend/e2e/judge-demo.spec.ts`
- Modify: `frontend/README.md`
- Modify: `frontend/playwright.config.ts`

**Interfaces:**
- Consumes: the completed launcher, controller, console, and verification experience.
- Produces: an automated acceptance test and operator-facing setup documentation.

- [ ] **Step 1: Write the Playwright acceptance test**

Automate: open app → choose `HIGH_VALUE_TRANSFER_ATTACK` → start → assert `LIVE` → wait for `92` and `CRITICAL` → assert reasons and blocked transfer → open verification → choose Verified Callback → complete it → assert secondary-channel resolution, `Voice Risk: CRITICAL`, and policy eligibility.

- [ ] **Step 2: Run the E2E test and verify it fails if integration is incomplete**

Run: `cd frontend; npm run test:e2e -- --project=chromium`

Expected: the test either passes if the prior tasks are complete or identifies the first missing integration state; fix only the implementation required by the failure.

- [ ] **Step 3: Document usage**

Update `frontend/README.md` with install, development, test, lint, build, `VITE_DEMO_MODE`, `VITE_API_BASE_URL`, scenario selection, WebSocket mode, and current demo-only limitations. Do not claim real ML or external verification.

- [ ] **Step 4: Run the full frontend checks**

Run: `cd frontend; npm test -- --run; npm run lint; npm run build; npm run test:e2e -- --project=chromium`

Expected: all commands exit 0.

- [ ] **Step 5: Commit**

```bash
git add frontend/e2e frontend/README.md frontend/playwright.config.ts
git commit -m "test: cover judge demo workflow"
```

### Task 9: Perform visual QA, scope review, and handoff

**Files:**
- Modify: only files required to correct verified visual or accessibility defects.
- Review: `git diff origin/main...HEAD --stat`, `git diff origin/main...HEAD`, and all changed frontend files.

**Interfaces:**
- Consumes: complete frontend implementation and acceptance test.
- Produces: evidence for the final PR; no merge into `main`.

- [ ] **Step 1: Run the app for browser inspection**

Run: `cd frontend; npm run dev -- --host 127.0.0.1`

Use Playwright/browser tooling at 1920×1080 and 1366×768. Capture evidence of launcher, LIVE call, escalating risk, critical block, callback verification, and final state.

- [ ] **Step 2: Inspect required visual conditions**

Check above-the-fold visibility of identity/request, live state, risk, evidence, and protection; inspect spacing, typography, chart labels, drawer focus, reduced motion, overflow, and console errors. Correct only observed defects, then rerun affected tests.

- [ ] **Step 3: Review scope and diff**

Run:

```bash
git status
git diff origin/main...HEAD --stat
git diff origin/main...HEAD
git diff --name-only origin/main...HEAD -- backend ml
```

Expected: clean final tree after evidence capture, frontend-only changed paths, and no backend/ML output.

- [ ] **Step 4: Request independent code review**

Use `superpowers:requesting-code-review` against bootstrap base `4281dcf4bffe92cee2321b82858dbf1d3e10251c`. Address all Critical and valid Important findings; record Minor findings honestly.

- [ ] **Step 5: Push and open, but do not merge, the PR**

```bash
git push -u origin feat/person2-ui-demo
gh pr create --base main --head feat/person2-ui-demo --title "JASH-002: Build premium VoxSentinel SIH demo experience" --body-file <reviewed-pr-body>
```

The PR must remain open for Brain review. Do not merge it or modify Rohan’s branch.
