import { describe, expect, it } from "vitest";
import { initialVerificationState, verificationReducer } from "./verification";

describe("verification state", () => {
  it("moves callback through pending to verified without changing voice risk", () => {
    const pending = verificationReducer(initialVerificationState, { type: "REQUEST", method: "VERIFIED_CALLBACK" });
    const verified = verificationReducer(pending, { type: "VERIFY" });

    expect(pending.status).toBe("PENDING");
    expect(verified).toMatchObject({ method: "VERIFIED_CALLBACK", status: "VERIFIED" });
    expect(verified.voiceRiskPreserved).toBe(true);
  });

  it("resets verification to idle", () => {
    const pending = verificationReducer(initialVerificationState, { type: "REQUEST", method: "OTP" });
    expect(verificationReducer(pending, { type: "RESET" })).toEqual(initialVerificationState);
  });

  it("supports a failed verification without changing preserved voice risk", () => {
    const pending = verificationReducer(initialVerificationState, { type: "REQUEST", method: "VOICE_CHALLENGE" });
    const failed = verificationReducer(pending, { type: "FAIL" });
    expect(failed).toMatchObject({ status: "FAILED", voiceRiskPreserved: true });
  });
});
