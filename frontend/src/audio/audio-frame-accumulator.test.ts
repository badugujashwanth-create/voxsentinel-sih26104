import { describe, expect, it } from "vitest";
import { AudioFrameAccumulator } from "./audio-frame-accumulator";

describe("AudioFrameAccumulator", () => {
  it("emits one 1024-sample frame and retains bounded residual", () => {
    const accumulator = new AudioFrameAccumulator(1_024, 1);
    const emitted = accumulator.append([new Float32Array(700)], 40n);
    expect(emitted).toHaveLength(0);
    expect(accumulator.append([new Float32Array(500)], 740n)).toHaveLength(1);
    expect(accumulator.residualSamples).toBe(176);
  });

  it("flushes a final non-empty residual without zero padding", () => {
    const accumulator = new AudioFrameAccumulator(1_024, 1);
    accumulator.append([new Float32Array(7)], 0n);
    const residual = accumulator.flush();
    expect(residual?.samples).toHaveLength(7);
    expect(residual?.firstSampleFrame).toBe(0n);
    expect(accumulator.flush()).toBeNull();
  });
});
