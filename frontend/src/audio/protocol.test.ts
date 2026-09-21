import { describe, expect, it } from "vitest";
import { AUDIO_HEADER_LENGTH, encodeAudioFrame, type TransportFrame } from "./protocol";

describe("VXAF audio protocol", () => {
  it("writes the exact 32-byte little-endian header and PCM payload", () => {
    const frame: TransportFrame = { sequence: 7n, firstSampleFrame: 19n, channels: 1, samples: new Float32Array([0.25, -0.5]) };
    const encoded = new Uint8Array(encodeAudioFrame(frame));
    const view = new DataView(encoded.buffer);
    expect(new TextDecoder().decode(encoded.slice(0, 4))).toBe("VXAF");
    expect(encoded[4]).toBe(1);
    expect(encoded[5]).toBe(32);
    expect(view.getUint16(6, true)).toBe(0);
    expect(view.getBigUint64(8, true)).toBe(7n);
    expect(view.getBigUint64(16, true)).toBe(19n);
    expect(view.getUint32(24, true)).toBe(2);
    expect(view.getUint32(28, true)).toBe(8);
    expect(encoded.byteLength).toBe(AUDIO_HEADER_LENGTH + 8);
  });
});
