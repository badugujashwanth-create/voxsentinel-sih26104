import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  AudioLines,
  Check,
  ChevronRight,
  Fingerprint,
  LockKeyhole,
  PhoneCall,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  UserRound,
  X,
} from "lucide-react";
import type { DemoController } from "../../hooks/useDemoController";
import { ShinyText } from "../ui/ShinyText";
import type { VerificationMethod } from "../../domain/verification";
import { SCENARIOS, type ScenarioId } from "../../scenarios/scenarios";

interface SecurityConsoleProps {
  controller: DemoController;
  verificationOpen: boolean;
  onOpenVerification(): void;
  onCloseVerification(): void;
}

const VERIFICATION_LABELS: Record<VerificationMethod, string> = {
  OTP: "Require OTP",
  VOICE_CHALLENGE: "Voice Challenge",
  VERIFIED_CALLBACK: "Verified Callback",
  SUPERVISOR_APPROVAL: "Supervisor Approval",
};

/** Renders the evolving forensic console and its verification drawer. */
export function SecurityConsole({ controller, verificationOpen, onOpenVerification, onCloseVerification }: SecurityConsoleProps) {
  const { context, currentEvent, selectedScenario, presentationState, voiceRisk, protectedAction, identityVerification, streamError } = controller;
  const scenario = selectedScenario ? SCENARIOS[selectedScenario] : null;
  const microphoneReady = !controller.liveMode || controller.microphoneState === "STREAMING";
  const isLive = microphoneReady && (controller.streamStatus === "LIVE" || (controller.timeline.length > 0 && !streamError && controller.streamStatus !== "ERROR"));
  const isCritical = voiceRisk.level === "CRITICAL";
  const isVerified = identityVerification.status === "VERIFIED";

  return (
    <main className={`console-page console-page--${presentationState.toLowerCase()}`}>
      <ConsoleHeader isLive={isLive} incidentId={controller.callId} liveMode={controller.liveMode} microphoneState={controller.microphoneState} onReset={controller.reset} />
      <div className="console-body">
        <CallContextStrip context={context} scenarioLabel={scenario?.label} isLive={isLive} score={voiceRisk.score} level={voiceRisk.level} timestamp={currentEvent?.timestamp_ms ?? 0} baselineTimestamp={controller.timeline[0]?.timestamp_ms ?? 0} />
        <section className="console-grid">
          <IdentityRail context={context} action={protectedAction.status} isVerified={isVerified} />
          <div className="signal-column">
            <SignalHeading state={presentationState} />
            <ForensicWaveform risk={voiceRisk.score} state={presentationState} />
            <SignalCaption />
            <SignalAlert state={presentationState} event={currentEvent} isCritical={isCritical} error={streamError} />
            <ProtectionState status={protectedAction.status} isCritical={isCritical} isVerified={isVerified} reasons={voiceRisk.reasons} onVerify={onOpenVerification} />
          </div>
          <EvidenceRail event={currentEvent} level={voiceRisk.level} state={presentationState} />
        </section>
        <section className="lower-band">
          <RiskTimeline events={controller.timeline} currentScore={voiceRisk.score} />
          <ReasonPanel reasons={voiceRisk.reasons} eventCount={controller.timeline.length} isCritical={isCritical} />
        </section>
        <ConsoleFooter controller={controller} selectedScenario={selectedScenario} />
      </div>
      <AnimatePresence>
        {verificationOpen && <VerificationDrawer state={identityVerification} onClose={onCloseVerification} onBegin={controller.beginVerification} onComplete={controller.completeVerification} onFail={controller.failVerification} />}
      </AnimatePresence>
    </main>
  );
}

/** Renders the persistent product and demo-mode header. */
function ConsoleHeader({ isLive, incidentId, liveMode, microphoneState, onReset }: { isLive: boolean; incidentId: string | null; liveMode: boolean; microphoneState: string; onReset(): void }) {
  const incidentLabel = incidentId ? `Incident #${incidentId}` : "Session pending";
  return <header className="brand-header console-header"><div className="brand-lockup"><AudioLines size={21} /><ShinyText text="VoxSentinel" /><span className="brand-slash">/</span><small>Forensic Command Workspace</small></div><div className="header-center"><span className="incident-tag"><span className={`status-dot ${isLive ? "status-dot--green" : "status-dot--muted"}`} /> {incidentLabel}</span><span className="header-separator">·</span><span>{isLive ? "Live call monitored" : liveMode ? "Awaiting backend call" : "Awaiting call"}</span>{liveMode && <><span className="mono-label">MIC {microphoneState}</span><span className="mono-label">RAW AUDIO NOT RETAINED</span></>}</div><div className="header-actions"><span className="simulation-badge"><span className={`status-dot ${liveMode ? "status-dot--green" : "status-dot--cyan"}`} /> {liveMode ? "LIVE MODE" : "DEMO MODE"}</span><button className="icon-button" aria-label="Reset scenario" onClick={onReset}><RotateCcw size={16} /></button></div></header>;
}

/** Renders the above-fold identity, request, call, and risk context. */
function CallContextStrip({ context, scenarioLabel, isLive, score, level, timestamp, baselineTimestamp }: { context: DemoController["context"]; scenarioLabel?: string; isLive: boolean; score: number; level: string; timestamp: number; baselineTimestamp: number }) {
  return <section className="context-strip" aria-label="Call context"><div className="context-cell context-cell--caller"><div className="context-icon"><UserRound size={21} /></div><div><span className="eyebrow">Claimed caller</span><strong>{context?.caller}</strong><small>{context?.role}</small></div></div><div className="context-cell context-cell--request"><div><span className="eyebrow">Sensitive request</span><strong>{context?.request}</strong><small>Urgent vendor transfer</small></div></div><div className="context-cell context-cell--call"><div className="live-label"><span className={`status-dot ${isLive ? "status-dot--green" : "status-dot--muted"}`} /> {isLive ? "LIVE CALL" : "CALL READY"}</div><strong className="mono-metric">{isLive ? formatDuration(timestamp, baselineTimestamp) : "00:00.00"}</strong><small>{scenarioLabel}</small></div><div className={`context-risk context-risk--${level.toLowerCase()}`}><span className="eyebrow">Impersonation risk</span><div><strong>{score}</strong><span>/100</span></div><b>{level}</b></div></section>;
}

/** Renders claimed identity and protected-action context. */
function IdentityRail({ context, action, isVerified }: { context: DemoController["context"]; action: string; isVerified: boolean }) {
  if (isVerified) return <aside className="identity-rail identity-rail--resolved" aria-label="Resolved security state"><div className="rail-section"><span className="eyebrow eyebrow--green">Secondary channel</span><strong className="resolved-rail-title">Callback verified</strong><p className="resolved-rail-copy">Claimed identity confirmed independently.</p></div><div className="rail-rule" /><div className="rail-section rail-section--small"><span className="eyebrow">Protected action</span><div className="action-chip action-chip--eligible"><span className="status-dot" />ELIGIBLE TO PROCEED</div><p>Independent confirmation recorded.</p></div></aside>;
  return <aside className="identity-rail" aria-label="Identity and action context"><div className="rail-section"><span className="eyebrow">Call identity</span><div className="identity-name"><div className="avatar-mark"><UserRound size={23} /></div><div><strong>{context?.caller}</strong><small>{context?.role}</small></div></div></div><div className="rail-rule" /><div className="rail-section"><span className="eyebrow">Protected action</span><div className="request-amount">₹25,00,000</div><div className="request-label">Vendor transfer</div><div className={`action-chip action-chip--${action.toLowerCase()}`}><span className="status-dot" />{action === "ELIGIBLE" ? "ELIGIBLE TO PROCEED" : action}</div></div><div className="rail-rule" /><div className="rail-section rail-section--small"><span className="eyebrow">Operator note</span><p>{isVerified ? "Identity was confirmed through an independent channel. Voice-risk evidence remains preserved." : "Monitor the voice signal and intervene when the protected action crosses policy threshold."}</p></div></aside>;
}

/** Renders the signal title for each progressive investigation state. */
function SignalHeading({ state }: { state: string }) {
  const title = state === "CALM" ? "Harmonic confidence signal" : state === "THREAT_EMERGING" ? "Forensic signal under review" : "Signal evidence preserved";
  return <div className="signal-heading"><div><span className="eyebrow eyebrow--cyan"><Activity size={13} /> Live voice forensics</span><h1>{title}</h1></div><span className="signal-spec mono-label">48.0 kHz · PCM</span></div>;
}

/** Renders a purposeful signal grid and deterministic harmonic trace. */
function ForensicWaveform({ risk, state }: { risk: number; state: string }) {
  const points = buildWaveform(risk);
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point[0]} ${point[1]}`).join(" ");
  const showMarkers = risk >= 30;
  const critical = risk >= 80;
  const lineColor = critical ? "#ef4444" : showMarkers ? "#f59e0b" : "#06b6d4";
  return <div className={`waveform ${critical ? "waveform--critical" : showMarkers ? "waveform--warning" : ""}`} aria-label={`Live harmonic voice signal, risk ${risk} out of 100`}><div className="waveform-axis waveform-axis--top mono-label">8.0 kHz</div><div className="waveform-axis waveform-axis--mid mono-label">3.4 kHz</div><div className="waveform-axis waveform-axis--bottom mono-label">80 Hz</div><svg viewBox="0 0 1000 300" preserveAspectRatio="none" role="img"><defs><pattern id="wave-grid" width="80" height="50" patternUnits="userSpaceOnUse"><path d="M 80 0 L 0 0 0 50" fill="none" stroke="rgba(134,147,151,.12)" strokeWidth="1" /></pattern></defs><rect width="1000" height="300" fill="url(#wave-grid)" /><path d={path} fill="none" stroke={lineColor} strokeWidth="2.2" vectorEffect="non-scaling-stroke" /><path d={path} fill="none" stroke={critical ? "#ffb4ab" : showMarkers ? "#fbbf24" : "#38bdf8"} strokeWidth="1" vectorEffect="non-scaling-stroke" />{showMarkers && <><line x1="686" y1="33" x2="686" y2="258" stroke={lineColor} strokeDasharray="4 5" /><circle cx="686" cy="145" r="5" fill={lineColor} /><text x="700" y="52" className="wave-annotation">{critical ? "CRITICAL DISCONTINUITY" : "SYNTHESIS ARTIFACT"}</text></>}{state === "THREAT_EMERGING" && <text x="780" y="270" className="wave-annotation wave-annotation--muted">IDENTITY DRIFT</text>}</svg><div className="waveform-footer"><span className="mono-label">FUNDAMENTAL / HARMONIC RESONANCE</span><span className={`mono-label ${critical ? "text-critical" : showMarkers ? "text-warning" : "text-cyan"}`}>{critical ? "VOICE RISK CRITICAL" : showMarkers ? "ANOMALY REVIEW ACTIVE" : "BASELINE STABLE"}</span></div></div>;
}

/** Renders signal annotations without implying real model inference. */
function SignalCaption() { return <div className="signal-caption"><span><span className="status-dot status-dot--cyan" /> LIVE AUDIO ANALYSIS</span><span className="mono-label">SIGNAL BAND ACTIVE</span><span className="mono-label">F1—F4 REFERENCE</span></div>; }

/** Renders progressive warning and critical forensic conclusions. */
function SignalAlert({ state, event, isCritical, error }: { state: string; event: DemoController["currentEvent"]; isCritical: boolean; error: Error | null }) {
  const prosodyUnavailable = event?.evidence_availability?.prosody_anomaly_score === "NOT_EVALUATED";
  return <AnimatePresence mode="wait">{error && <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="signal-alert signal-alert--critical"><ShieldAlert size={17} /><div><strong>Live connection unavailable</strong><span>{error.message}</span></div><span className="mono-label">CHECK BACKEND</span></motion.div>}{!error && state === "THREAT_EMERGING" && <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="signal-alert signal-alert--warning"><AlertTriangle size={16} /><div><strong>Synthesis artifact under review</strong><span>Signal continuity requires further authenticity review.</span></div><span className="mono-label">{prosodyUnavailable ? "ANOMALY N/A" : `ANOMALY ${Math.round((event?.prosody_anomaly_score ?? 0) * 100)}%`}</span></motion.div>}{!error && isCritical && <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="signal-alert signal-alert--critical"><ShieldAlert size={17} /><div><strong>Critical voice-risk conclusion</strong><span>Secondary verification cannot change the original audio assessment.</span></div><span className="mono-label">PRESERVED</span></motion.div>}</AnimatePresence>;
}

/** Renders the intervention state and critical reasons above the fold. */
function ProtectionState({ status, isCritical, isVerified, reasons, onVerify }: { status: string; isCritical: boolean; isVerified: boolean; reasons: string[]; onVerify(): void }) {
  if (isVerified) return <motion.div layout className="protection-state protection-state--eligible"><div className="protection-icon"><ShieldCheck size={22} /></div><div><span className="eyebrow eyebrow--green">Identity verified through secondary channel</span><strong>Protected action is now eligible to proceed according to policy.</strong><small>Voice authenticity remains <b className="text-critical">CRITICAL</b>. Original evidence preserved.</small><div className="security-triad"><span><b>VOICE AUTHENTICITY</b> CRITICAL</span><span><b>IDENTITY</b> VERIFIED VIA CALLBACK</span><span><b>ACTION</b> ELIGIBLE TO PROCEED</span></div></div></motion.div>;
  if (isCritical) return <motion.div layout className="protection-state protection-state--blocked"><div className="protection-icon"><ShieldAlert size={22} /></div><div><span className="eyebrow eyebrow--critical">Sensitive action protected</span><strong>TRANSFER TEMPORARILY BLOCKED</strong><small>Secondary verification required before this action can proceed.</small><div className="protection-reasons">{reasons.map((reason) => <span key={reason}>• {reason}</span>)}</div></div><button className="button button--critical" onClick={onVerify}>VERIFY IDENTITY <ChevronRight size={15} /></button></motion.div>;
  return <motion.div layout className="protection-state"><div className="protection-icon"><LockKeyhole size={19} /></div><div><span className="eyebrow">Transaction status</span><strong>{status === "WARNING" ? "Operator review recommended" : "TRANSACTION MONITORING"}</strong><small>{status === "WARNING" ? "Risk is elevated; prepare a secondary verification." : "No prevention action has been taken."}</small></div></motion.div>;
}

/** Renders the compact accumulating current-call risk chart. */
function RiskTimeline({ events, currentScore }: { events: DemoController["timeline"]; currentScore: number }) {
  const scores = events.map((event) => event.overall_risk_score);
  const points = scores.length > 1 ? scores.map((score, index) => `${(index / (scores.length - 1)) * 100},${100 - score}`).join(" ") : `0,${100 - currentScore}`;
  return <div className="timeline-panel"><div className="section-heading"><span className="eyebrow">Live risk timeline</span><span className="mono-label">CURRENT CALL · 0—100</span></div><div className="timeline-chart"><div className="timeline-threshold timeline-threshold--critical" /><div className="timeline-threshold timeline-threshold--high" /><div className="timeline-threshold-labels"><span>CRITICAL</span><span>HIGH</span><span>LOW</span></div><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Risk score over current call"><polyline points={points} fill="none" stroke={currentScore >= 80 ? "#ef4444" : currentScore >= 30 ? "#f59e0b" : "#06b6d4"} strokeWidth="1.7" vectorEffect="non-scaling-stroke" />{scores.map((score, index) => <circle key={`${score}-${index}`} cx={scores.length > 1 ? (index / (scores.length - 1)) * 100 : 0} cy={100 - score} r="2" fill={score >= 80 ? "#ef4444" : score >= 30 ? "#f59e0b" : "#06b6d4"} />)}</svg></div><div className="timeline-values"><span>START</span>{scores.slice(-3).map((score, index) => <strong key={`${score}-${index}`}>{score}</strong>)}<span>NOW</span></div></div>;
}

/** Renders plain-language reasons without competing with the intervention. */
function ReasonPanel({ reasons, eventCount, isCritical }: { reasons: string[]; eventCount: number; isCritical: boolean }) { return <div className="reason-panel"><div className="section-heading"><span className="eyebrow">Why this call is being evaluated</span><span className="mono-label">{eventCount ? `${eventCount.toString().padStart(2, "0")} EVENTS` : "BASELINE"}</span></div><div className="reason-list">{reasons.map((reason) => <div className="reason-item" key={reason}><span className={`reason-marker ${isCritical ? "reason-marker--critical" : ""}`} />{reason}</div>)}</div></div>; }

/** Renders discreet operator controls below the judge-facing workspace. */
function ConsoleFooter({ controller, selectedScenario }: { controller: DemoController; selectedScenario: ScenarioId | null }) { const modeCopy = controller.liveMode ? "LIVE BACKEND MODE · Backend stream data" : "SIMULATION MODE · Values are deterministic demo data"; const activeControl = controller.liveMode && controller.microphoneState === "STREAMING" ? <button className="text-button text-button--danger" onClick={() => void controller.stop()}>Stop analysis</button> : controller.streamStatus === "LIVE" ? <button className="text-button" onClick={controller.pause}>Pause</button> : <button className="text-button text-button--accent" onClick={() => void controller.start()}>{controller.timeline.length ? "Resume" : "Start analysis"}<ChevronRight size={14} /></button>; return <footer className="console-footer"><span><LockKeyhole size={13} /> {modeCopy}</span><span className="mono-label">STREAM ADAPTER: {controller.liveMode ? "WEBSOCKET" : "MOCK"}</span><div className="footer-controls"><button className="text-button" onClick={() => selectedScenario && controller.selectScenario(selectedScenario)}><ArrowLeft size={14} /> Change scenario</button><button className="text-button" onClick={controller.reset}><RotateCcw size={14} /> Reset</button>{activeControl}</div></footer>; }

/** Renders independent verification methods and their simulated progress. */
function VerificationDrawer({ state, onClose, onBegin, onComplete, onFail }: { state: DemoController["identityVerification"]; onClose(): void; onBegin(method: VerificationMethod): void; onComplete(): void; onFail(): void }) {
  const methods: VerificationMethod[] = ["VERIFIED_CALLBACK", "VOICE_CHALLENGE", "OTP", "SUPERVISOR_APPROVAL"];
  const activeLabel = state.method ? VERIFICATION_LABELS[state.method] : null;
  return <motion.aside role="dialog" aria-modal="true" className="verification-drawer" initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }} transition={{ duration: 0.25 }} aria-label="Secondary verification"><div className="drawer-header"><div><span className="eyebrow eyebrow--green">Out-of-band verification</span><h2>Verify identity</h2></div><button className="icon-button" aria-label="Close verification" onClick={onClose}><X size={17} /></button></div><div className="drawer-callout"><PhoneCall size={18} /><div><strong>Voice risk remains CRITICAL</strong><small>Choose an independent channel to verify the claimed identity.</small></div></div>{state.status === "IDLE" ? <div className="verification-methods">{methods.map((method, index) => <button className={`verification-method ${index === 0 ? "verification-method--primary" : ""}`} key={method} onClick={() => onBegin(method)}><span className="verification-step">0{index + 1}</span><span><strong>{VERIFICATION_LABELS[method]}</strong><small>{getMethodDescription(method)}</small></span><ChevronRight size={15} /></button>)}</div> : <div className="verification-progress"><div className="progress-status"><div className="progress-icon">{state.status === "VERIFIED" ? <Check size={22} /> : state.status === "FAILED" ? <X size={22} /> : <Activity size={22} />}</div><span className={`eyebrow eyebrow--${state.status === "VERIFIED" ? "green" : state.status === "FAILED" ? "critical" : "cyan"}`}>{state.status === "VERIFIED" ? "Identity confirmed" : state.status === "FAILED" ? "Verification failed" : activeLabel}</span><h3>{state.status === "VERIFIED" ? "IDENTITY VERIFIED THROUGH SECONDARY CHANNEL" : state.status === "FAILED" ? "ACTION REMAINS BLOCKED" : state.step}</h3><p>{state.status === "VERIFIED" ? "Protected action is now eligible to proceed according to policy." : state.status === "FAILED" ? "The original voice-risk evidence is preserved. Try another method or keep the action blocked." : "Complete the simulated verification step to continue."}</p></div>{state.status === "PENDING" && <>{state.method === "VERIFIED_CALLBACK" && <CallbackSteps step={state.step} />}<div className="progress-actions"><button className="button button--primary" onClick={onComplete}>Simulate success <Check size={15} /></button><button className="text-button text-button--danger" onClick={onFail}>Simulate failure</button></div></>}{state.status !== "PENDING" && <button className="button button--ghost" onClick={onClose}>Return to console <ChevronRight size={15} /></button>}</div>}<div className="drawer-footer"><ShieldCheck size={14} /><span>Demo verification · No SMS or telephony integration</span></div></motion.aside>;
}

/** Shows the canonical callback sequence without changing the voice-risk conclusion. */
function CallbackSteps({ step }: { step: string }) { const stages = ["CALLBACK INITIATED", "SECURE CHANNEL ESTABLISHED", "IDENTITY CONFIRMED"]; const activeIndex = step === "Secure channel established" ? 1 : 0; return <div className="callback-steps">{stages.map((stage, index) => <span className={index <= activeIndex ? "callback-step callback-step--active" : "callback-step"} key={stage}><b>{index + 1}</b>{stage}</span>)}</div>; }

/** Returns concise copy for a simulated verification method. */
function getMethodDescription(method: VerificationMethod): string { if (method === "VERIFIED_CALLBACK") return "Confirm through a separate trusted channel"; if (method === "VOICE_CHALLENGE") return "Ask for a one-time phrase response"; if (method === "SUPERVISOR_APPROVAL") return "Request a policy-owner decision"; return "Send a simulated one-time passcode"; }

/** Builds repeatable harmonic points from the current risk score. */
function buildWaveform(risk: number): Array<[number, number]> { const pointCount = 96; const startX = 42; const endX = 958; const amplitude = 42 + Math.min(risk, 100) * 0.12; return Array.from({ length: pointCount }, (_, index) => { const progress = index / (pointCount - 1); const x = startX + (endX - startX) * progress; const harmonic = Math.sin(progress * Math.PI * 12) * amplitude; const overtone = Math.sin(progress * Math.PI * 28 + 0.4) * (8 + risk * 0.12); const discontinuity = risk >= 30 && progress > 0.72 ? Math.sin(progress * Math.PI * 70) * (risk - 20) * 0.18 : 0; const microVariation = Math.sin(progress * Math.PI * 37 + risk * 0.03) * (risk >= 30 ? 4 : 1.5); return [x, 150 - harmonic - overtone - discontinuity - microVariation]; }); }

/** Formats deterministic demo elapsed time as mm:ss.cc. */
function formatDuration(timestampMs: number, baselineTimestampMs: number): string { const elapsedMs = Math.max(0, timestampMs - baselineTimestampMs); const seconds = Math.floor(elapsedMs / 1000); const milliseconds = elapsedMs % 1000; return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}.${milliseconds.toString().padStart(3, "0").slice(0, 2)}`; }

/** Renders the strongest supporting evidence channels. */
function EvidenceRail({ event, level, state }: { event: DemoController["currentEvent"]; level: string; state: string }) {
  const calm = state === "CALM";
  const availability = (field: string) => event?.evidence_availability?.[field];
  const isUnavailable = (field: string) => {
    const value = availability(field);
    return value !== undefined && value !== "MEASURED" && value !== "EVALUATED";
  };
  const evidence = event ? [
    { field: "synthetic_probability", label: "Synthetic voice", value: event.synthetic_probability, interpretation: event.synthetic_probability > 0.6 ? "Characteristics detected" : "Within baseline", tone: event.synthetic_probability > 0.6 ? "critical" : "cyan", semantics: event.synthetic_score_semantics === "uncalibrated" ? "Uncalibrated spoof evidence" : undefined },
    { field: "speaker_match_score", label: "Speaker identity", value: event.speaker_similarity ?? event.speaker_match_score, interpretation: event.speaker_state === "INCONSISTENT" ? "Speaker mismatch detected" : event.speaker_state === "CONSISTENT" ? "Expected speaker consistent" : "No enrolled reference", tone: event.speaker_state === "INCONSISTENT" ? "warning" : "cyan", semantics: event.speaker_score_semantics === "uncalibrated_similarity" ? "Uncalibrated speaker similarity" : undefined },
    { field: "context_risk_score", label: "Context risk", value: event.context_risk_score, interpretation: event.context_risk_score > 0.6 ? "High-value request" : "Request monitored", tone: event.context_risk_score > 0.6 ? "warning" : "cyan", semantics: undefined },
  ] : [];
  return <aside className="evidence-rail" aria-label="Forensic evidence"><div className="section-heading"><span className="eyebrow">Evidence signals</span><span className="mono-label">{calm ? "BASELINE" : level}</span></div>{calm ? <div className="evidence-calm"><div className="evidence-calm__icon"><Fingerprint size={18} /></div><strong>Signal baseline forming</strong><p>Forensic evidence will appear as the call develops.</p></div> : <div className="evidence-list">{evidence.map((item) => { const unavailable = isUnavailable(item.field); const speakerSimilarity = item.field === "speaker_match_score" && event?.speaker_score_semantics === "uncalibrated_similarity"; const trackValue = speakerSimilarity ? (item.value + 1) / 2 : item.value; const displayValue = unavailable ? "N/A" : speakerSimilarity ? item.value.toFixed(2) : item.field === "synthetic_probability" && event?.synthetic_score_semantics === "uncalibrated" ? item.value.toFixed(2) : `${Math.round(item.value * 100)}%`; return <div className="evidence-row" key={item.label}><div className="evidence-row__top"><span>{item.label}</span><strong className={`text-${item.tone}`}>{displayValue}</strong></div><div className="evidence-track"><span className={`evidence-fill evidence-fill--${item.tone}`} style={{ width: unavailable ? "0%" : `${Math.round(trackValue * 100)}%` }} /></div><small>{unavailable ? "Not evaluated" : item.semantics ?? item.interpretation}</small></div>; })}</div>}</aside>;
}
