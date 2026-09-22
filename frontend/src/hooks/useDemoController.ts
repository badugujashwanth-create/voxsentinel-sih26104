import { useCallback, useMemo, useState } from "react";
import { getPresentationState, getProtectedActionState, type CallContext, type PresentationState, type ProtectedActionState, type VoiceRiskState } from "../domain/call";
import { initialVerificationState, verificationReducer, type VerificationMethod, type VerificationState } from "../domain/verification";
import { SCENARIOS, type ScenarioId } from "../scenarios/scenarios";
import { MockRiskStreamSource } from "../services/risk-stream/MockRiskStreamSource";
import type { RiskStreamSource } from "../services/risk-stream/RiskStreamSource";
import { LiveCallRiskStreamSource } from "../services/risk-stream/LiveCallRiskStreamSource";
import { useRiskStream } from "./useRiskStream";
import { MicrophoneSession, type MicrophoneState } from "../audio/microphone-session";

export type DemoMode = "DEMO" | "LIVE";

export interface DemoController {
  mode: DemoMode;
  selectedScenario: ScenarioId | null;
  context: CallContext | null;
  presentationState: PresentationState;
  streamStatus: ReturnType<typeof useRiskStream>["status"];
  streamError: ReturnType<typeof useRiskStream>["error"];
  currentEvent: ReturnType<typeof useRiskStream>["events"][number] | null;
  timeline: ReturnType<typeof useRiskStream>["events"];
  voiceRisk: VoiceRiskState;
  identityVerification: VerificationState;
  protectedAction: ProtectedActionState;
  selectScenario(scenarioId: ScenarioId): void;
  selectMode(mode: DemoMode): void;
  start(): Promise<void>;
  stop(): Promise<void>;
  pause(): void;
  resume(): void;
  reset(): void;
  beginVerification(method: VerificationMethod): void;
  completeVerification(): void;
  failVerification(): void;
  callId: string | null;
  liveMode: boolean;
  microphoneState: MicrophoneState;
}

/** Coordinates deterministic demo mode and the future WebSocket mode. */
export function useDemoController(): DemoController {
  const [mode, setMode] = useState<DemoMode>(import.meta.env.VITE_DEMO_MODE === "false" ? "LIVE" : "DEMO");
  const liveMode = mode === "LIVE";
  const [microphoneState, setMicrophoneState] = useState<MicrophoneState>("IDLE");
  const [selectedScenario, setSelectedScenario] = useState<ScenarioId | null>(null);
  const [verification, setVerification] = useState<VerificationState>(initialVerificationState);
  const source = useMemo(() => createSource(selectedScenario, liveMode, setMicrophoneState), [selectedScenario, liveMode]);
  const stream = useRiskStream(source);
  const context = selectedScenario ? SCENARIOS[selectedScenario] : null;
  const currentEvent = stream.events.at(-1) ?? context?.events[0] ?? null;
  const callId = stream.events.at(-1)?.call_id ?? (liveMode ? null : context?.events[0]?.call_id ?? null);
  const presentationState = getPresentationState(currentEvent, verification);
  const protectedAction = getProtectedActionState(currentEvent, verification);
  const voiceRisk: VoiceRiskState = {
    score: currentEvent?.overall_risk_score ?? 0,
    level: currentEvent?.risk_level ?? "LOW",
    reasons: currentEvent?.reasons ?? [],
  };

  const selectScenario = useCallback((scenarioId: ScenarioId) => {
    setSelectedScenario(scenarioId);
    setVerification(initialVerificationState);
  }, []);

  /** Changes the source mode while the call is idle. */
  const selectMode = useCallback((nextMode: DemoMode) => {
    if (stream.status === "CONNECTING" || stream.status === "LIVE" || stream.status === "PAUSED") return;
    setMode(nextMode);
  }, [stream.status]);

  const reset = useCallback(() => {
    stream.reset();
    setVerification(initialVerificationState);
  }, [stream]);

  const beginVerification = useCallback((method: VerificationMethod) => {
    setVerification((current) => verificationReducer(current, { type: "REQUEST", method }));
  }, []);

  const completeVerification = useCallback(() => {
    setVerification((current) => verificationReducer(current, { type: "VERIFY" }));
  }, []);

  const failVerification = useCallback(() => {
    setVerification((current) => verificationReducer(current, { type: "FAIL" }));
  }, []);

  return {
    mode,
    selectedScenario,
    context,
    presentationState,
    streamStatus: stream.status,
    streamError: stream.error,
    currentEvent,
    timeline: stream.events,
    voiceRisk,
    identityVerification: verification,
    protectedAction,
    selectScenario,
    selectMode,
    start: stream.start,
    stop: stream.stop,
    pause: stream.pause,
    resume: stream.resume,
    reset,
    beginVerification,
    completeVerification,
    failVerification,
    callId,
    liveMode,
    microphoneState,
  };
}

/** Creates the configured source for the selected scenario. */
function createSource(scenarioId: ScenarioId | null, liveMode: boolean, onMicrophoneState: (state: MicrophoneState) => void): RiskStreamSource | null {
  if (!scenarioId) return null;
  if (!liveMode) return new MockRiskStreamSource(scenarioId);
  const scenario = SCENARIOS[scenarioId];
  const baseUrl = import.meta.env.VITE_API_BASE_URL;
  const microphone = new MicrophoneSession({ baseUrl }, onMicrophoneState);
  return new LiveCallRiskStreamSource({
    baseUrl,
    request: {
      claimed_identity: scenario.caller,
      scenario: scenario.id,
      transaction_value: 2500000,
      currency: "INR",
      ...(import.meta.env.VITE_SPEAKER_PROFILE_ID ? { speaker_profile_id: import.meta.env.VITE_SPEAKER_PROFILE_ID } : {}),
    },
    microphone,
  });
}
