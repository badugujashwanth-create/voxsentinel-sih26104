import type { LiveRiskEvent, RiskLevel } from "./risk";
import type { VerificationState } from "./verification";

export type PresentationState = "CALM" | "THREAT_EMERGING" | "CRITICAL_INTERVENTION" | "SECONDARY_VERIFICATION";
export type ProtectedActionStatus = "MONITORING" | "WARNING" | "BLOCKED" | "ELIGIBLE";

export interface VoiceRiskState {
  score: number;
  level: RiskLevel;
  reasons: string[];
}

export interface ProtectedActionState {
  status: ProtectedActionStatus;
  label: string;
}

export interface CallContext {
  caller: string;
  role: string;
  request: string;
}

/** Derives the progressive visual state from voice risk and verification state. */
export function getPresentationState(event: LiveRiskEvent | null, verification: VerificationState): PresentationState {
  if (verification.status !== "IDLE") return "SECONDARY_VERIFICATION";
  if (!event || event.overall_risk_score < 30) return "CALM";
  if (event.overall_risk_score < 80) return "THREAT_EMERGING";
  return "CRITICAL_INTERVENTION";
}

/** Derives protected-action authorization without changing the voice conclusion. */
export function getProtectedActionState(event: LiveRiskEvent | null, verification: VerificationState): ProtectedActionState {
  if (verification.status === "VERIFIED") return { status: "ELIGIBLE", label: "Eligible to proceed according to policy" };
  if (event?.recommended_action === "BLOCK_ACTION") return { status: "BLOCKED", label: "Transfer temporarily blocked" };
  if (event?.overall_risk_score !== undefined && event.overall_risk_score >= 60) return { status: "WARNING", label: "Secondary verification required" };
  return { status: "MONITORING", label: "Transaction monitoring" };
}
