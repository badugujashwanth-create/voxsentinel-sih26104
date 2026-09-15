# JASH-002 Premium SIH UI & Judge Demo Experience

## Status

Design specification for review. This document describes the frontend-only implementation planned for `feat/person2-ui-demo`.

## Goal

Build one polished, deterministic VoxSentinel security console that lets a judge understand a live sensitive call through four progressive investigation states: calm monitoring, emerging threat, critical intervention, and secondary-channel resolution.

## Scope and non-goals

The deliverable is a React/TypeScript/Vite frontend that runs independently in deterministic demo mode. It includes four scenario fixtures, a typed stream-source boundary, a mock source, an unused WebSocket source, risk/evidence visualization, prevention states, and lightweight simulated verification.

The deliverable does not include FastAPI, ML inference, AASIST, RawNet2, WavLM, training, Twilio, SMS, databases, authentication, Docker, Kubernetes, or changes to `backend/` or `ml/`.

## Visual thesis

VoxSentinel is a premium cybersecurity operations console: deep graphite/navy surfaces, off-white typography, restrained electric-cyan analysis accents, and amber/red severity states. The composition is calm and card-light, with a central dominant risk workspace, a left identity/context rail, a right evidence rail, an integrated compact timeline, and a prevention state treated as part of the workspace rather than a separate dashboard card.

The primary display is optimized for 16:9 presentation. Typography, spacing, contrast, and dividers carry hierarchy before decorative effects. `DEMO MODE` or `SIMULATION MODE` remains a small persistent indicator so mock values cannot be mistaken for real inference.

## Primary information hierarchy

At both 1920×1080 and 1366×768, without scrolling, the first viewport must show:

1. claimed identity and sensitive request;
2. live call state, scenario, and elapsed time;
3. overall impersonation risk and severity;
4. major evidence signals, with speaker identity presented as `SPEAKER MATCH` first and `identity mismatch` as secondary interpretation;
5. current protection/action state.

The timeline, explanation details, and verification controls may compress or move into the lower workspace/drawer only after these five items remain visible. No primary viewport layout may require vertical scrolling for the canonical scenario.

## Screen flow

```text
Scenario Launcher
    ↓
Live Security Console
    ↓
LIVE CALL + streaming evidence
    ↓
Risk escalation
    ↓
92 / 100 — CRITICAL
    ↓
TRANSFER TEMPORARILY BLOCKED
    ↓
Dynamic reasons displayed
    ↓
Verified Callback or Voice Challenge
    ↓
IDENTITY VERIFIED THROUGH SECONDARY CHANNEL
    ↓
Protected action is now eligible to proceed according to policy.
```

The canonical `HIGH_VALUE_TRANSFER_ATTACK` path uses Verified Callback. The operator can restart, reset, pause when useful, or choose another scenario from a discreet operator drawer; these controls do not compete with the judge-facing hierarchy.

## Three distinct security states

The UI and domain state must keep these concepts separate:

| State | Meaning | Example |
| --- | --- | --- |
| Voice authenticity risk | Suspicion that the audio is synthetic, replayed, or otherwise anomalous | `CRITICAL` |
| Claimed-identity verification | Whether the claimed person has been verified through a secondary channel | `Verified via Callback` |
| Protected-action authorization | Whether policy permits the requested sensitive action | `Eligible to Proceed` or `Blocked` |

Secondary verification must never rewrite the voice-risk conclusion. After a CRITICAL voice-risk event, successful OTP, callback, challenge, or supervisor approval preserves the CRITICAL risk score, original evidence, reasons, and incident timeline. It changes only identity verification and the policy-derived action state. The resolved presentation must say `IDENTITY VERIFIED THROUGH SECONDARY CHANNEL`, followed by `Protected action is now eligible to proceed according to policy.`

## Live Security Console

The console uses a compact top bar plus a three-region workspace whose disclosure level changes with the investigation state:

- Left identity/context rail: claimed caller, role, call status, scenario, elapsed time, and sensitive request such as `₹25,00,000 vendor transfer`.
- Center risk workspace: dominant animated `Impersonation Risk` score, risk label, live procedural waveform labelled `LIVE AUDIO ANALYSIS`, current incident statement, and the integrated protection state.
- Right evidence rail: Synthetic Voice, Speaker Match, Prosody Anomaly, Replay Risk, and Contextual Risk. Each metric has a value, label, and short interpretation; bars are not the only meaning-bearing element.

In CALM, the dominant live signal and caller/request context lead while deep forensic detail remains hidden. In THREAT EMERGING, the same composition gains a small number of waveform annotations and the strongest supporting evidence. In CRITICAL, the console promotes the intervention decision, reasons, and next action. In SECONDARY VERIFICATION, the console promotes the callback progression and the three separate security dimensions while preserving the original voice-risk evidence.

The lower portion contains a compact current-call risk timeline with 0–100 scale and LOW/MEDIUM/HIGH/CRITICAL threshold bands, plus a dynamic explanation section headed `WHY THIS CALL WAS FLAGGED`. The timeline and secondary telemetry compress before claimed identity/request, live state, risk, major evidence, or current action move below the fold. The critical action state stays visually adjacent to the main risk workspace and becomes unmistakable at `BLOCK_ACTION`.

## Risk levels and event contract

Risk levels use inclusive boundaries:

```text
0–29   LOW
30–59  MEDIUM
60–79  HIGH
80–100 CRITICAL
```

The frontend uses this exact event shape without casually renaming fields:

```typescript
interface LiveRiskEvent {
  call_id: string;
  sequence: number;
  timestamp_ms: number;

  synthetic_probability: number;

  speaker_match_score: number;
  speaker_mismatch_score: number;

  prosody_anomaly_score: number;
  replay_risk_score: number;
  context_risk_score: number;

  overall_risk_score: number;

  risk_level:
    | "LOW"
    | "MEDIUM"
    | "HIGH"
    | "CRITICAL";

  reasons: string[];

  recommended_action:
    | "NONE"
    | "MONITOR"
    | "REQUIRE_OTP"
    | "REQUIRE_CALLBACK"
    | "REQUIRE_VOICE_CHALLENGE"
    | "REQUIRE_SUPERVISOR"
    | "BLOCK_ACTION";
}
```

Scores labelled probability use the inclusive range `0.0–1.0`; `overall_risk_score` uses `0–100`. Fixture validation rejects malformed values, invalid levels/actions, empty call IDs, non-positive sequences, and out-of-order events.

## Risk stream source boundary

Timer and socket logic live only in services. React presentation components consume controller state and never create scenario timers directly.

```typescript
type RiskStreamStatus =
  | "IDLE"
  | "CONNECTING"
  | "LIVE"
  | "PAUSED"
  | "DISCONNECTED"
  | "ERROR";

interface RiskStreamHandlers {
  onEvent(event: LiveRiskEvent): void;
  onStatusChange(status: RiskStreamStatus): void;
  onError(error: Error): void;
}

interface RiskStreamSource {
  start(handlers: RiskStreamHandlers): Promise<void>;
  stop(): Promise<void>;
  pause(): void;
  resume(): void;
  reset(): void;
  dispose(): void;
}
```

`MockRiskStreamSource` implements the contract with deterministic scenario fixtures, controlled interval scheduling, sequence validation, pause/resume, and cleanup. `WebSocketRiskStreamSource` implements the same contract for `/api/v1/calls/{call_id}/risk-stream`, derives its URL from `VITE_API_BASE_URL`, parses and validates incoming JSON, reports malformed messages and disconnects, and performs no backend call in demo mode.

## Scenario fixtures

The source emits fixed, repeatable risk progressions and the console derives a presentation state from them:

| Scenario | Risk progression | Expected final state |
| --- | --- | --- |
| `GENUINE` | `12, 10, 14, 11, 13` | LOW, monitor |
| `HUMAN_IMPOSTOR` | `18, 26, 39, 54, 67, 76` | HIGH, identity mismatch emphasized |
| `AI_CLONE` | `20, 31, 46, 63, 78, 88` | CRITICAL, synthetic evidence emphasized |
| `HIGH_VALUE_TRANSFER_ATTACK` | `18, 27, 43, 61, 79, 92` | CRITICAL, `BLOCK_ACTION` |

Presentation states are `CALM` at the initial low-risk event, `THREAT_EMERGING` once anomaly evidence is introduced, `CRITICAL_INTERVENTION` at critical action blocking, and `SECONDARY_VERIFICATION` while a verification flow is active or resolved. The state changes disclosure and emphasis without replacing the entire page or changing the event contract.

Each fixture defines caller context, request, evidence scores, reasons, timestamps, risk levels, and recommended actions. The high-value transfer story is a senior executive impersonation requesting `₹25,00,000` urgently. Its reason list evolves from monitoring context to synthetic speech characteristics, speaker identity mismatch, and high-value transfer context.

## Prevention and verification

At MEDIUM/HIGH the console communicates monitoring or warning. At CRITICAL with `BLOCK_ACTION`, it shows:

```text
TRANSFER TEMPORARILY BLOCKED
Risk score: 92 / 100
Secondary verification required before this action can proceed.
```

All four lightweight simulated methods are available:

- OTP: requested → sent → pending → verified/failed.
- Voice Challenge: phrase shown → passed/failed.
- Verified Callback: requested → in progress → verified.
- Supervisor Approval: awaiting → approved/rejected.

The canonical transfer flow opens Verified Callback. A success produces the secondary-channel language above and makes the action policy-eligible; it does not relabel the voice as genuine. A failure keeps the action blocked and says `VERIFICATION FAILED`. The final screen must visibly retain `VOICE AUTHENTICITY — CRITICAL`, `IDENTITY — VERIFIED VIA CALLBACK`, and `ACTION — ELIGIBLE TO PROCEED` as separate values.

## Motion and accessibility

Motion is limited to risk-score interpolation, evidence/timeline updates, alert transitions, drawer/modal presence, and verification state changes. `prefers-reduced-motion` disables interpolation and nonessential transitions. Primary controls are keyboard reachable with visible focus, use labels and descriptive status text, maintain readable contrast, and never communicate state through color alone.

## Validation and acceptance

The implementation is accepted only when:

- unit/component tests cover risk boundaries, fixtures, adapters, state separation, and verification reset/success/failure;
- Playwright completes the canonical launcher → LIVE CALL → 92 CRITICAL → blocked → callback → verified flow;
- `npm test`, `npm run lint`, and `npm run build` pass;
- browser inspection shows the five critical information groups above the fold at 1920×1080 and 1366×768;
- no browser console errors, clipping, broken chart labels, or backend/ML scope changes are present;
- the UI visibly retains CRITICAL voice evidence after secondary verification.
- the four imported visual states read as one progressive sequence: CALM → THREAT EMERGING → CRITICAL INTERVENTION → SECONDARY VERIFICATION.

## Environment and limitations

`VITE_DEMO_MODE=true` selects the offline mock source. `VITE_API_BASE_URL` supplies the backend origin for future WebSocket mode; no production URL is hardcoded. Demo values are simulated and are labelled as such. There is no real SMS, callback, voice challenge engine, supervisor system, or ML inference in this task.
