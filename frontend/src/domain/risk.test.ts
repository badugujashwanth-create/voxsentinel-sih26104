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
      synthetic_score_semantics: "uncalibrated",
      evidence_availability: { speaker_match_score: "NOT_EVALUATED", context_risk_score: "NOT_EVALUATED" },
    };
    expect(validateLiveRiskEvent(event).evidence_availability?.speaker_match_score).toBe("NOT_EVALUATED");
    expect(validateLiveRiskEvent(event).synthetic_score_semantics).toBe("uncalibrated");
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

  it("preserves optional audio source correlation metadata", () => {
    const event = {
      call_id: "call-1", sequence: 1, timestamp_ms: 100, synthetic_probability: 0.1,
      speaker_match_score: 0, speaker_mismatch_score: 0, prosody_anomaly_score: 0,
      replay_risk_score: 0, context_risk_score: 0, overall_risk_score: 20,
      risk_level: "LOW", reasons: ["baseline"], recommended_action: "MONITOR",
      audio_source_frame_start: 100, audio_source_frame_end: 200,
      audio_source_transport_sequence_start: 1, audio_source_transport_sequence_end: 2,
      audio_source_gap_count: 0, audio_window_sequence: 1,
    };
    expect(validateLiveRiskEvent(event).audio_source_frame_end).toBe(200);
  });

  it("accepts explicit uncalibrated speaker similarity and fusion action", () => {
    const event = {
      call_id: "call-1", sequence: 1, timestamp_ms: 100, synthetic_probability: 0.1,
      speaker_match_score: 0.82, speaker_mismatch_score: 0.18, prosody_anomaly_score: 0,
      replay_risk_score: 0, context_risk_score: 0, overall_risk_score: 50,
      risk_level: "MEDIUM", reasons: ["Speaker identity mismatch requires verification"], recommended_action: "VERIFY_IDENTITY",
      speaker_score_semantics: "uncalibrated_similarity", speaker_similarity: 0.82, speaker_threshold: 0.55,
      speaker_state: "INCONSISTENT", fusion_state: "IDENTITY_REVIEW",
      evidence_availability: { speaker_match_score: "EVALUATED" },
    } as const;
    expect(validateLiveRiskEvent(event).speaker_score_semantics).toBe("uncalibrated_similarity");
    expect(validateLiveRiskEvent(event).recommended_action).toBe("VERIFY_IDENTITY");
  });
});
