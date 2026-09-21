import { describe, expect, it } from "vitest";
import { captureEndPerformanceMs, createAudioTimingAnchor, steadyStateLatencyMs } from "./timing";

describe("browser audio timing", () => {
  it("maps the final source frame to browser-observed latency", () => {
    const anchor = createAudioTimingAnchor(2, 1_000, 48_000);
    const captureEnd = captureEndPerformanceMs(48_000n + 2_399n, anchor);
    expect(captureEnd).toBeCloseTo(49.979, 2);
    expect(steadyStateLatencyMs(2_100, captureEnd)).toBeCloseTo(2_050.021, 2);
  });
});
