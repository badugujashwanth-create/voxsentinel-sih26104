# Stitch Implementation Notes

## 1. Four-state UX story

The exports form a clear progressive investigation sequence rather than four unrelated pages:

```text
CALM
  caller known · live signal · 18 LOW
      ↓
THREAT EMERGING
  waveform discontinuity · synthesis artifact · identity drift · 79 ELEVATED
      ↓
CRITICAL INTERVENTION
  92 CRITICAL · ₹25,00,000 transfer blocked · reasons and next action
      ↓
SECONDARY VERIFICATION
  voice remains CRITICAL · callback verifies identity · action becomes policy-eligible
```

State 1 is intentionally spacious and waveform-led. State 2 keeps the same forensic surface but introduces annotations and only the strongest evidence. State 3 changes the decision emphasis to intervention. State 4 changes the operator task to out-of-band resolution while retaining the original voice-risk conclusion.

## 2. Visual source of truth

The PNGs are the visual reference for composition, tone, density, typography, signal motif, and state transitions. The generated HTML files are reference implementations for inspection, not copy-paste production code. The design system in `design-system/DESIGN.md` is the source for exact palette, typography, radius, spacing, and acoustic motif values.

## 3. Preserve

- The architectural dark canvas with edge-to-edge structural planes and hairline separators.
- VoxSentinel wordmark with a small acoustic waveform mark.
- Geist for product copy and JetBrains Mono for telemetry, labels, timestamps, and numeric readouts.
- The asymmetric investigation composition: identity/request context at the top or left, dominant acoustic signal in the center, evidence revealed progressively, and intervention integrated into the main surface.
- The cyan live-signal trace, confidence band, grid/ruler treatment, and precise vertical anomaly markers.
- Calm cyan/green State 1, amber State 2, crimson State 3, and green/cyan State 4 semantic progression.
- State 3’s immediate `92`, `CRITICAL`, `₹25,00,000`, and `TRANSFER BLOCKED` hierarchy.
- State 4’s three-pillar separation: voice authenticity, identity resolution, and transaction/action status.
- The restrained rectangular geometry: 4px operational radius, 8px maximum module radius, and zero-radius signal viewports.
- Progressive disclosure as the core product behavior, not just a visual variation.

## 4. Adapt

- Replace the generated top navigation and incidental links with the single judge-facing console plus discreet operator drawer; keep only navigation that supports the demonstration.
- Preserve the State 1 waveform as the dominant visual, but use deterministic mock data and honest `SIMULATION MODE` copy instead of claims such as `Neural Verification Operational` or `Voiceprint Verified`.
- Keep State 2 annotations near the waveform, but reveal only the strongest evidence based on the current event. Move detailed telemetry into an inspectable secondary region.
- Keep State 3’s large intervention banner and three reason groups, but use the functional copy `Synthetic speech characteristics detected`, `Speaker identity mismatch`, and `High-value financial request` without invented model/vendor/policy claims.
- Keep State 4’s three security dimensions, but explicitly label the final action as `Eligible to Proceed according to policy`; callback verification must not turn the voice state green or erase the CRITICAL incident history.
- Preserve the screen rhythm while enforcing the JASH-002 above-the-fold requirement at 1920×1080 and 1366×768. Compress the lower event trace before allowing identity, risk, evidence, or action state to fall below the fold.
- Replace Material Symbols and Tailwind CDN assumptions with a small local icon library and repository-owned CSS tokens.
- Keep only one polished canonical verification path—Verified Callback—visually primary; expose OTP, Voice Challenge, and Supervisor Approval as lightweight alternatives.

## 5. Reject

- Wholesale generated HTML insertion into `frontend/src`; it is duplicated, single-screen markup with inline Tailwind classes and no typed domain boundary.
- External CDN runtime dependencies, Google Fonts runtime dependence, and Material Symbols as a production requirement.
- Unsupported claims in the exports such as HSM sealing, cryptographic session hashes, ElevenLabs/X TTS model identification, exact enterprise policy IDs, FIDO hardware facts, database indexing, or real neural verification. These are visual copy references only and cannot be presented as implemented functionality.
- State 2’s equal-weight metric cards as the permanent layout. They flatten the progressive story and create a generic dashboard impression.
- Excessive rounded containers, pills, badges, and shadowed modules where a divider or plain layout is enough.
- Tiny telemetry that cannot be read at presentation distance and decorative labels that do not explain a security decision.
- Constant `animate-pulse`/`animate-ping` activity, ambient glow, gradient-heavy surfaces, and terminal-like copy without operational meaning.
- State 4 language that implies the callback authenticated the original audio, such as “voiceprint threat database” or “cryptographically signed vocal tokens.”

## 6. Design tokens

### Typography

- UI and editorial family: Geist.
- Telemetry family: JetBrains Mono.
- Headline XL: 44px / 52px / 600 / -0.03em; mobile 30px / 38px.
- Headline LG: 32px / 40px / 600 / -0.025em; mobile 24px / 32px.
- Headline MD: 22px / 30px / 500.
- Body LG: 16px / 24px; body MD 14px / 20px; body SM 12px / 18px.
- Labels: JetBrains Mono, 13px, 11px, and 10px scales with expanded tracking.
- Mono metric: JetBrains Mono, 26px / 32px / 600 / -0.02em with tabular numeric treatment.

### Colors

| Token | Value | Use |
| --- | --- | --- |
| Canvas base | `#080a0f` | page substrate |
| Surface 1 | `#0f121a` | primary monitoring planes |
| Surface 2 | `#151a24` | nested/active telemetry |
| Hairline | `#1e2533` | separators and structural borders |
| Hairline strong | `#283244` | focused or elevated borders |
| Primary text | `#f8fafc` | metrics and status headlines |
| Secondary text | `#cbd5e1` | context and descriptions |
| Muted text | `#8892b0` | metadata and inactive labels |
| Live analysis | `#06b6d4` / `#38bdf8` | waveform and live telemetry |
| Emerging anomaly | `#f59e0b` / `#fbbf24` | warning and investigation |
| Critical | `#ef4444` / `#dc2626` | intervention and blocked action |
| Verified | `#10b981` | secondary-channel confirmation |

The supplied semantic Material-style tokens additionally include `#111319` surface, `#e2e2e9` on-surface, `#bcc9cd` on-surface-variant, `#4cd7f6` primary, `#4edea3` secondary, and `#ffb95f` tertiary. Production tokens should normalize to one coherent set while preserving these visual values.

### Spacing, radius, borders, and depth

- Base rhythm is 0.25rem / 0.5rem / 1rem / 1.5rem / 2.5rem / 4rem.
- Desktop gutter is 1.5rem; desktop margin is 2.5rem.
- Operational radius is 0.125rem–0.25rem; standard modules max out at 0.5rem; use 0px for signal ports and timeline surfaces.
- Borders are 1px hairlines; avoid thick outlines and alternating table fills.
- Depth comes from tonal surfaces and borders, not drop-shadow stacks. A controlled amber or red perimeter is sufficient for escalation.

## 7. Responsive behavior

At large desktop, preserve the asymmetric forensic runway. At 1366×768, collapse secondary telemetry and shorten the lower event trace before shrinking the caller, request, risk, evidence, and action content below readable sizes. At tablet/mobile, stack context above the signal and move detailed evidence into a prioritized feed or drawer; retain the same state model and semantic ordering.

## 8. Motion strategy

- Calm → threat: reveal anomaly markers and evidence with short opacity/position transitions tied to received events.
- Risk escalation: interpolate the score and trace only when the value changes; use reduced-motion fallback for direct updates.
- Critical intervention: one deliberate perimeter/status transition and a single blocked-action reveal; no ambient page shake or perpetual glow.
- Verification: drawer/modal presence, step progression, and success/failure state transitions.
- Resolution: update identity and authorization regions while keeping the voice-risk region visually unchanged.

## 9. Accessibility corrections

- Use semantic headings, landmarks, buttons, lists, and status regions rather than div-only generated markup.
- Provide accessible labels for waveform, score, evidence rows, and action state; expose numeric values and text interpretations alongside color.
- Preserve visible keyboard focus and logical drawer/modal focus management.
- Do not depend on hover-only annotations or tiny monospace text for essential decisions.
- Respect `prefers-reduced-motion` and maintain contrast for cyan, amber, crimson, and green against the dark canvas.

## 10. Anti-AI-generated UI rules

The production reconstruction should not look AI-generated because hierarchy is stateful and intentional: asymmetry supports forensic attention; typography separates product meaning from telemetry; information density rises only when risk rises; component variation reflects the operator’s decision; colors are semantic and scarce; generic cards are replaced by structural planes and dividers; the audio waveform and anomaly markers are voice-specific rather than decorative; motion corresponds to evidence and intervention; copy describes realistic operator decisions; spacing follows a visible rhythm; and controls stay subordinate to the product surface.

## 11. Differences from Stitch generated code

- Stitch outputs are standalone HTML documents, not React/TypeScript modules.
- They load Tailwind from a CDN and configure tokens inline; production will use local typed styles and a build-time dependency.
- They repeat full headers, shells, and large markup trees per state; production will share components and vary state through domain data.
- They use Material Symbols and external Google Fonts; production must provide an intentional dependency strategy and accessible icon labels.
- They contain hardcoded telemetry, timers, invented operational claims, and screen-specific annotations; production will bind only to validated `LiveRiskEvent` data and deterministic fixtures.
- The exports emphasize different layouts in different states; production will preserve the same visual evolution through one state-aware console architecture.

## 12. Production implementation recommendations

Implement `ScenarioLauncher`, a shared state-aware `LiveSecurityConsole`, `RiskStreamSource` adapters, and focused regions for identity, signal, evidence, intervention, timeline, and verification. Use the existing JASH-002 domain/event contract. Add a `presentationState` derived from the current event and verification state so progressive disclosure is explicit: CALM, EMERGING, CRITICAL, and SECONDARY_VERIFICATION. Keep the voice event and incident history immutable after verification; derive action eligibility separately. Validate visual behavior with Playwright screenshots at 1920×1080 and 1366×768 before the implementation PR.
