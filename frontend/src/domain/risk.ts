export const RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"] as const;
export type RiskLevel = (typeof RISK_LEVELS)[number];

export const RECOMMENDED_ACTIONS = [
  "NONE",
  "MONITOR",
  "REQUIRE_OTP",
  "REQUIRE_CALLBACK",
  "REQUIRE_VOICE_CHALLENGE",
  "REQUIRE_SUPERVISOR",
  "BLOCK_ACTION",
] as const;
export type RecommendedAction = (typeof RECOMMENDED_ACTIONS)[number];
export const EVIDENCE_AVAILABILITY = ["MEASURED", "NOT_EVALUATED"] as const;
export type EvidenceAvailability = (typeof EVIDENCE_AVAILABILITY)[number];

export interface LiveRiskEvent {
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
  risk_level: RiskLevel;
  reasons: string[];
  recommended_action: RecommendedAction;
  inference_latency_ms?: number;
  provider_round_trip_ms?: number;
  synthetic_score_semantics?: "uncalibrated";
  evidence_availability?: Record<string, EvidenceAvailability>;
}

const MIN_RISK_SCORE = 0;
const MAX_RISK_SCORE = 100;
const MIN_PROBABILITY = 0;
const MAX_PROBABILITY = 1;

/** Maps an overall score to the product's inclusive severity bands. */
export function getRiskLevel(score: number): RiskLevel {
  if (score < 30) return "LOW";
  if (score < 60) return "MEDIUM";
  if (score < 80) return "HIGH";
  return "CRITICAL";
}

/** Validates and narrows an unknown stream payload to a safe risk event. */
export function validateLiveRiskEvent(value: unknown): LiveRiskEvent {
  if (!isRecord(value)) throw new Error("Risk event must be an object");

  const event = value as Partial<LiveRiskEvent>;
  if (typeof event.call_id !== "string" || event.call_id.trim().length === 0) throw new Error("Risk event call_id is required");
  if (!isPositiveInteger(event.sequence)) throw new Error("Risk event sequence must be a positive integer");
  if (!isNonNegativeNumber(event.timestamp_ms)) throw new Error("Risk event timestamp_ms must be non-negative");

  const syntheticProbability = readProbability(event, "synthetic_probability");
  const speakerMatchScore = readProbability(event, "speaker_match_score");
  const speakerMismatchScore = readProbability(event, "speaker_mismatch_score");
  const prosodyAnomalyScore = readProbability(event, "prosody_anomaly_score");
  const replayRiskScore = readProbability(event, "replay_risk_score");
  const contextRiskScore = readProbability(event, "context_risk_score");

  if (!isNonNegativeNumber(event.overall_risk_score) || event.overall_risk_score < MIN_RISK_SCORE || event.overall_risk_score > MAX_RISK_SCORE) {
    throw new Error("Risk event overall_risk_score must be between 0 and 100");
  }
  if (!isRiskLevel(event.risk_level)) throw new Error("Risk event risk_level is invalid");
  if (event.risk_level !== getRiskLevel(event.overall_risk_score)) throw new Error("Risk event risk_level does not match overall_risk_score");
  if (!isRecommendedAction(event.recommended_action)) throw new Error("Risk event recommended_action is invalid");
  if (!Array.isArray(event.reasons) || event.reasons.some((reason) => typeof reason !== "string" || reason.trim().length === 0)) {
    throw new Error("Risk event reasons must contain non-empty strings");
  }
  if (event.evidence_availability !== undefined) {
    if (!isRecord(event.evidence_availability) || Object.values(event.evidence_availability).some((availability) => !isEvidenceAvailability(availability))) {
      throw new Error("Risk event evidence_availability is invalid");
    }
  }
  if (event.synthetic_score_semantics !== undefined && event.synthetic_score_semantics !== "uncalibrated") {
    throw new Error("Risk event synthetic_score_semantics is invalid");
  }
  for (const field of ["inference_latency_ms", "provider_round_trip_ms"] as const) {
    if (event[field] !== undefined && (typeof event[field] !== "number" || !Number.isFinite(event[field]) || event[field] < 0)) {
      throw new Error(`Risk event ${field} is invalid`);
    }
  }

  return {
    call_id: event.call_id,
    sequence: event.sequence,
    timestamp_ms: event.timestamp_ms,
    synthetic_probability: syntheticProbability,
    speaker_match_score: speakerMatchScore,
    speaker_mismatch_score: speakerMismatchScore,
    prosody_anomaly_score: prosodyAnomalyScore,
    replay_risk_score: replayRiskScore,
    context_risk_score: contextRiskScore,
    overall_risk_score: event.overall_risk_score,
    risk_level: event.risk_level,
    reasons: [...event.reasons],
    recommended_action: event.recommended_action,
    inference_latency_ms: event.inference_latency_ms,
    provider_round_trip_ms: event.provider_round_trip_ms,
    synthetic_score_semantics: event.synthetic_score_semantics,
    evidence_availability: event.evidence_availability ? { ...event.evidence_availability } : undefined,
  };
}

/** Checks whether a value is a plain object suitable for event parsing. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Checks whether a value is a finite non-negative number. */
function isNonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= MIN_RISK_SCORE;
}

/** Checks whether a value is a probability in the inclusive 0..1 range. */
function isProbability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= MIN_PROBABILITY && value <= MAX_PROBABILITY;
}

/** Reads and validates one probability field from an untrusted event. */
function readProbability(event: Partial<LiveRiskEvent>, field: keyof Pick<LiveRiskEvent, "synthetic_probability" | "speaker_match_score" | "speaker_mismatch_score" | "prosody_anomaly_score" | "replay_risk_score" | "context_risk_score">): number {
  const value = event[field];
  if (!isProbability(value)) throw new Error(`Risk event ${field} must be between 0 and 1`);
  return value;
}

/** Checks whether a value is a positive integer. */
function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0;
}

/** Checks whether a value is a known risk level. */
function isRiskLevel(value: unknown): value is RiskLevel {
  return typeof value === "string" && RISK_LEVELS.includes(value as RiskLevel);
}

/** Checks whether a value is a known recommended action. */
function isRecommendedAction(value: unknown): value is RecommendedAction {
  return typeof value === "string" && RECOMMENDED_ACTIONS.includes(value as RecommendedAction);
}

/** Checks whether a detector availability value is part of the shared contract. */
function isEvidenceAvailability(value: unknown): value is EvidenceAvailability {
  return typeof value === "string" && EVIDENCE_AVAILABILITY.includes(value as EvidenceAvailability);
}
