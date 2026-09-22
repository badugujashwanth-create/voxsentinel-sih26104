import type { MicrophoneState } from "./microphone-session";
import type { RiskLevel, RecommendedAction } from "../domain/risk";

export interface AcceptanceInference {
  audio_window_sequence?: number;
  raw_spoof_score: number;
  aggregate_score: number;
  policy_state: string;
  overall_risk_score: number;
  risk_level: RiskLevel;
  recommended_action: RecommendedAction;
  ml_inference_latency_ms?: number;
  steady_state_latency_ms: number | null;
  score_semantics?: "uncalibrated";
  speaker_similarity?: number | null;
  speaker_state?: string;
  fusion_state?: string;
}

export interface AcceptanceSnapshot {
  call_id: string;
  microphone_state?: MicrophoneState;
  audio_context_sample_rate?: number;
  channels?: number;
  browser_frames_produced?: number;
  browser_frames_sent?: number;
  browser_frames_dropped?: number;
  transport_sequence_gap_count?: number;
  warmup_latency_ms?: number | null;
  cleanup?: { tracks_active: boolean; audio_context_active: boolean; audio_socket_active: boolean };
  inferences: AcceptanceInference[];
}

const MAX_INFERENCES = 50;

/** Creates a bounded metadata-only snapshot for local acceptance tooling. */
export function createAcceptanceSnapshot(input: Omit<AcceptanceSnapshot, "inferences"> & { inferences: AcceptanceInference[] }): AcceptanceSnapshot {
  return { ...input, inferences: input.inferences.slice(-MAX_INFERENCES).map((inference) => ({ ...inference, score_semantics: "uncalibrated" })) };
}

/** Publishes ephemeral developer telemetry without persistence or audio data. */
export function publishAcceptanceSnapshot(snapshot: AcceptanceSnapshot): void {
  if (!import.meta.env.DEV && import.meta.env.VITE_ACCEPTANCE_TELEMETRY !== "1") return;
  (window as Window & { __VOXSENTINEL_ACCEPTANCE__?: AcceptanceSnapshot }).__VOXSENTINEL_ACCEPTANCE__ = snapshot;
}
