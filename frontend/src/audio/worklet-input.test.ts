import { describe, expect, it } from "vitest";
import { selectTransportChannels } from "./worklet-input";

describe("selectTransportChannels", () => {
  it("keeps mono transport when Web Audio exposes extra source channels", () => {
    const left = new Float32Array([0.1, 0.2]);
    const right = new Float32Array([0.8, 0.9]);

    expect(selectTransportChannels([left, right], 1)).toEqual([left]);
  });
});
