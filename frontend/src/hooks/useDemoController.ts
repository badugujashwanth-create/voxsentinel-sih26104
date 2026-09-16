import { useCallback, useMemo, useState } from "react";
import { getPresentationState, getProtectedActionState, type CallContext, type PresentationState, type ProtectedActionState, type VoiceRiskState } from "../domain/call";
import { initialVerificationState, verificationReducer, type VerificationMethod, type VerificationState } from "../domain/verification";
import { SCENARIOS, type ScenarioId } from "../scenarios/scenarios";
import { MockRiskStreamSource } from "../services/risk-stream/MockRiskStreamSource";
import type { RiskStreamSource } from "../services/risk-stream/RiskStreamSource";
import { WebSocketRiskStreamSource } from "../services/risk-stream/WebSocketRiskStreamSource";
import { useRiskStream } from "./useRiskStream";

export interface DemoController {
  selectedScenario: ScenarioId | null;
  context: CallContext | null;
  presentationState: PresentationState;
  streamStatus: ReturnType<typeof useRiskStream>["status"];
  currentEvent: ReturnType<typeof useRiskStream>["events"][number] | null;
  timeline: ReturnType<typeof useRiskStream>["events"];
  voiceRisk: VoiceRiskState;
  identityVerification: VerificationState;
  protectedAction: ProtectedActionState;
  selectScenario(scenarioId: ScenarioId): void;
  start(): Promise<void>;
  pause(): void;
  resume(): void;
  reset(): void;
  beginVerification(method: VerificationMethod): void;
  completeVerification(): void;
  failVerification(): void;
}

/** Coordinates deterministic demo mode and the future WebSocket mode. */
export function useDemoController(): DemoController {
  const [selectedScenario, setSelectedScenario] = useState<ScenarioId | null>(null);
  const [verification, setVerification] = useState<VerificationState>(initialVerificationState);
  const source = useMemo(() => createSource(selectedScenario), [selectedScenario]);
  const stream = useRiskStream(source);
  const context = selectedScenario ? SCENARIOS[selectedScenario] : null;
  const currentEvent = stream.events.at(-1) ?? context?.events[0] ?? null;
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
    selectedScenario,
    context,
    presentationState,
    streamStatus: stream.status,
    currentEvent,
    timeline: stream.events,
    voiceRisk,
    identityVerification: verification,
    protectedAction,
    selectScenario,
    start: stream.start,
    pause: stream.pause,
    resume: stream.resume,
    reset,
    beginVerification,
    completeVerification,
    failVerification,
  };
}

/** Creates the configured source for the selected scenario. */
function createSource(scenarioId: ScenarioId | null): RiskStreamSource | null {
  if (!scenarioId) return null;
  const demoMode = import.meta.env.VITE_DEMO_MODE !== "false";
  if (demoMode) return new MockRiskStreamSource(scenarioId);
  return new WebSocketRiskStreamSource(SCENARIOS[scenarioId].events[0].call_id);
}
