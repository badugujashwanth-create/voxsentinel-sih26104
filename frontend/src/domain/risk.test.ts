import { describe, expect, it } from "vitest";
import { getRiskLevel, validateLiveRiskEvent } from "./risk";

describe("risk domain", () => {
  it.each([
    [29, "LOW"],
    [30, "MEDIUM"],
    [59, "MEDIUM"],
    [60, "HIGH"],
    [79, "HIGH"],
    [80, "CRITICAL"],
    [100, "CRITICAL"],
  ])("maps %i to %s", (score, level) => {
    expect(getRiskLevel(score)).toBe(level);
  });

  it("rejects an event with a probability outside 0..1", () => {
    expect(() => validateLiveRiskEvent({ synthetic_probability: 2 })).toThrow();
  });

  it("rejects a severity that contradicts the overall score", () => {
    const event = {
      call_id: "call-1", sequence: 1, timestamp_ms: 100, synthetic_probability: 0.1,
      speaker_match_score: 0.9, speaker_mismatch_score: 0.1, prosody_anomaly_score: 0.1,
      replay_risk_score: 0.1, context_risk_score: 0.1, overall_risk_score: 92,
      risk_level: "LOW", reasons: ["baseline"], recommended_action: "MONITOR",
    };
    expect(() => validateLiveRiskEvent(event)).toThrow(/does not match/);
  });

  it("preserves explicit unavailable evidence metadata", () => {
    const event = {
      call_id: "call-1", sequence: 1, timestamp_ms: 100, synthetic_probability: 0.8,
      speaker_match_score: 0, speaker_mismatch_score: 0, prosody_anomaly_score: 0,
      replay_risk_score: 0, context_risk_score: 0, overall_risk_score: 70,
      risk_level: "HIGH", reasons: ["review"], recommended_action: "REQUIRE_CALLBACK",
      evidence_availability: { speaker_match_score: "NOT_EVALUATED", context_risk_score: "NOT_EVALUATED" },
    };
    expect(validateLiveRiskEvent(event).evidence_availability?.speaker_match_score).toBe("NOT_EVALUATED");
  });

  it("rejects unknown evidence availability values", () => {
    const event = {
      call_id: "call-1", sequence: 1, timestamp_ms: 100, synthetic_probability: 0.8,
      speaker_match_score: 0, speaker_mismatch_score: 0, prosody_anomaly_score: 0,
      replay_risk_score: 0, context_risk_score: 0, overall_risk_score: 70,
      risk_level: "HIGH", reasons: ["review"], recommended_action: "REQUIRE_CALLBACK",
      evidence_availability: { speaker_match_score: "SAFE" },
    };
    expect(() => validateLiveRiskEvent(event)).toThrow();
  });
});
