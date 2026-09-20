import { describe, expect, it } from "vitest";
import { createAcceptanceSnapshot, publishAcceptanceSnapshot, type AcceptanceInference } from "./acceptance-telemetry";

describe("acceptance telemetry", () => {
  it("keeps bounded inference metadata and no audio payload", () => {
    const inferences: AcceptanceInference[] = Array.from({ length: 3 }, (_, index) => ({
      audio_window_sequence: index + 1,
      raw_spoof_score: 0.1,
      aggregate_score: 0.1,
      policy_state: "NORMAL",
      overall_risk_score: 20,
      risk_level: "LOW",
      recommended_action: "MONITOR",
      ml_inference_latency_ms: 3,
      steady_state_latency_ms: null,
    }));
    const snapshot = createAcceptanceSnapshot({ call_id: "call-1", inferences });
    expect(snapshot.inferences).toHaveLength(3);
    expect(snapshot.inferences[0].score_semantics).toBe("uncalibrated");
    expect(Object.keys(snapshot)).not.toContain("samples");
    expect(Object.keys(snapshot)).not.toContain("pcm");
  });

  it("publishes only an ephemeral developer snapshot", () => {
    const snapshot = createAcceptanceSnapshot({ call_id: "call-2", inferences: [] });
    publishAcceptanceSnapshot(snapshot);
    expect((window as Window & { __VOXSENTINEL_ACCEPTANCE__?: unknown }).__VOXSENTINEL_ACCEPTANCE__).toEqual(snapshot);
  });
});
