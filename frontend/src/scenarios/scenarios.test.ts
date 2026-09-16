import { describe, expect, it } from "vitest";
import { SCENARIOS } from "./scenarios";

describe("scenario fixtures", () => {
  it("contains the required deterministic risk progressions", () => {
    expect(SCENARIOS.GENUINE.events.map((event) => event.overall_risk_score)).toEqual([12, 10, 14, 11, 13]);
    expect(SCENARIOS.HUMAN_IMPOSTOR.events.map((event) => event.overall_risk_score)).toEqual([18, 26, 39, 54, 67, 76]);
    expect(SCENARIOS.AI_CLONE.events.map((event) => event.overall_risk_score)).toEqual([20, 31, 46, 63, 78, 88]);
  });

  it("ends the transfer attack at critical block action", () => {
    const finalEvent = SCENARIOS.HIGH_VALUE_TRANSFER_ATTACK.events.at(-1);
    expect(finalEvent).toMatchObject({ overall_risk_score: 92, risk_level: "CRITICAL", recommended_action: "BLOCK_ACTION" });
  });

  it("distinguishes a human impostor from synthetic speech", () => {
    const finalEvent = SCENARIOS.HUMAN_IMPOSTOR.events.at(-1);
    expect(finalEvent?.synthetic_probability).toBeLessThan(0.2);
    expect(finalEvent?.speaker_mismatch_score).toBeGreaterThan(0.5);
  });

  it("ends the AI clone scenario at critical risk", () => {
    expect(SCENARIOS.AI_CLONE.events.at(-1)).toMatchObject({ risk_level: "CRITICAL" });
  });
});
