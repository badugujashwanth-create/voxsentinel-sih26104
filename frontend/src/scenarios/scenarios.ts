import { getRiskLevel, type LiveRiskEvent, type RecommendedAction } from "../domain/risk";

export type ScenarioId = "GENUINE" | "HUMAN_IMPOSTOR" | "AI_CLONE" | "HIGH_VALUE_TRANSFER_ATTACK";

export interface ScenarioDefinition {
  id: ScenarioId;
  label: string;
  summary: string;
  caller: string;
  role: string;
  request: string;
  events: LiveRiskEvent[];
}

const CALL_ID = "VX-88914";
const EVENT_INTERVAL_MS = 700;

interface EventProfile {
  synthetic_probability: number;
  speaker_match_score: number;
  speaker_mismatch_score: number;
  prosody_anomaly_score: number;
  replay_risk_score: number;
  context_risk_score: number;
  reasons: string[];
  recommended_action: RecommendedAction;
}

/** Creates a validated deterministic event for one scenario step. */
function createEvent(sequence: number, score: number, profile: EventProfile): LiveRiskEvent {
  return {
    call_id: CALL_ID,
    sequence,
    timestamp_ms: sequence * EVENT_INTERVAL_MS,
    ...profile,
    overall_risk_score: score,
    risk_level: getRiskLevel(score),
  };
}

/** Maps fixture risk scores to their declared product level. */
/** Creates a sequence of fixture events from scores and profiles. */
function createEvents(scores: number[], profiles: EventProfile[]): LiveRiskEvent[] {
  return scores.map((score, index) => createEvent(index + 1, score, profiles[index]));
}

const calmProfile: EventProfile = {
  synthetic_probability: 0.08,
  speaker_match_score: 0.92,
  speaker_mismatch_score: 0.08,
  prosody_anomaly_score: 0.11,
  replay_risk_score: 0.06,
  context_risk_score: 0.12,
  reasons: ["Voice signal remains within the enrolled speaker baseline"],
  recommended_action: "MONITOR",
};

const impostorProfiles: EventProfile[] = [
  calmProfile,
  { ...calmProfile, speaker_match_score: 0.74, speaker_mismatch_score: 0.26, reasons: ["Speaker identity drift is emerging"] },
  { ...calmProfile, speaker_match_score: 0.61, speaker_mismatch_score: 0.39, context_risk_score: 0.4, reasons: ["Speaker identity mismatch is increasing"] },
  { ...calmProfile, speaker_match_score: 0.46, speaker_mismatch_score: 0.54, context_risk_score: 0.62, reasons: ["Speaker identity mismatch", "Sensitive request requires review"] },
  { ...calmProfile, speaker_match_score: 0.33, speaker_mismatch_score: 0.67, context_risk_score: 0.76, reasons: ["Speaker identity mismatch", "High-value financial request"] },
  { ...calmProfile, speaker_match_score: 0.24, speaker_mismatch_score: 0.76, context_risk_score: 0.82, reasons: ["Speaker identity mismatch", "High-value financial request"] },
];

const cloneProfiles: EventProfile[] = [
  calmProfile,
  { ...calmProfile, synthetic_probability: 0.31, prosody_anomaly_score: 0.38, reasons: ["Synthesis artifact is emerging"] },
  { ...calmProfile, synthetic_probability: 0.46, prosody_anomaly_score: 0.55, speaker_match_score: 0.68, speaker_mismatch_score: 0.32, reasons: ["Synthetic speech characteristics detected"] },
  { ...calmProfile, synthetic_probability: 0.63, prosody_anomaly_score: 0.67, speaker_match_score: 0.48, speaker_mismatch_score: 0.52, reasons: ["Synthetic speech characteristics detected", "Prosody anomaly detected"] },
  { ...calmProfile, synthetic_probability: 0.78, prosody_anomaly_score: 0.78, speaker_match_score: 0.34, speaker_mismatch_score: 0.66, reasons: ["Synthetic speech characteristics detected", "Speaker identity mismatch"] },
  { ...calmProfile, synthetic_probability: 0.88, prosody_anomaly_score: 0.86, speaker_match_score: 0.22, speaker_mismatch_score: 0.78, reasons: ["Synthetic speech characteristics detected", "Speaker identity mismatch", "Prosody anomaly detected"], recommended_action: "BLOCK_ACTION" },
];

const transferProfiles: EventProfile[] = [
  { ...calmProfile, context_risk_score: 0.18 },
  { ...calmProfile, context_risk_score: 0.27, reasons: ["Sensitive financial request is being monitored"] },
  { ...calmProfile, synthetic_probability: 0.43, context_risk_score: 0.43, prosody_anomaly_score: 0.36, reasons: ["Synthesis artifact is emerging", "High-value financial request"] },
  { ...calmProfile, synthetic_probability: 0.61, speaker_match_score: 0.48, speaker_mismatch_score: 0.52, context_risk_score: 0.61, prosody_anomaly_score: 0.58, reasons: ["Synthetic speech characteristics detected", "Speaker identity mismatch"] },
  { ...calmProfile, synthetic_probability: 0.79, speaker_match_score: 0.31, speaker_mismatch_score: 0.69, context_risk_score: 0.79, prosody_anomaly_score: 0.72, reasons: ["Synthetic speech characteristics detected", "Speaker identity mismatch", "High-value financial request"], recommended_action: "REQUIRE_CALLBACK" },
  { ...calmProfile, synthetic_probability: 0.91, speaker_match_score: 0.28, speaker_mismatch_score: 0.72, context_risk_score: 0.95, prosody_anomaly_score: 0.74, replay_risk_score: 0.21, reasons: ["Synthetic speech characteristics detected", "Speaker identity mismatch", "High-value financial request"], recommended_action: "BLOCK_ACTION" },
];

export const SCENARIOS: Record<ScenarioId, ScenarioDefinition> = {
  GENUINE: { id: "GENUINE", label: "Genuine Caller", summary: "Known speaker, ordinary request", caller: "Arjun Mehta", role: "Chief Financial Officer", request: "₹25,00,000 Vendor Transfer", events: createEvents([12, 10, 14, 11, 13], [calmProfile, calmProfile, calmProfile, calmProfile, calmProfile]) },
  HUMAN_IMPOSTOR: { id: "HUMAN_IMPOSTOR", label: "Human Impostor", summary: "Human speech with identity mismatch", caller: "Arjun Mehta", role: "Chief Financial Officer", request: "₹25,00,000 Vendor Transfer", events: createEvents([18, 26, 39, 54, 67, 76], impostorProfiles) },
  AI_CLONE: { id: "AI_CLONE", label: "AI Voice Clone", summary: "Synthetic speech characteristics emerge", caller: "Arjun Mehta", role: "Chief Financial Officer", request: "₹25,00,000 Vendor Transfer", events: createEvents([20, 31, 46, 63, 78, 88], cloneProfiles) },
  HIGH_VALUE_TRANSFER_ATTACK: { id: "HIGH_VALUE_TRANSFER_ATTACK", label: "High-Value Transfer Attack", summary: "Urgent executive impersonation attempt", caller: "Arjun Mehta", role: "Chief Financial Officer", request: "₹25,00,000 Vendor Transfer", events: createEvents([18, 27, 43, 61, 79, 92], transferProfiles) },
};

export { EVENT_INTERVAL_MS };
