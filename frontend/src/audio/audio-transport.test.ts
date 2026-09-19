import { describe, expect, it } from "vitest";
import { AudioTransport, AUDIO_WS_HIGH_WATERMARK_BYTES, AUDIO_WS_LOW_WATERMARK_BYTES } from "./audio-transport";

function frame() {
  return { firstSampleFrame: 0n, channels: 1, samples: new Float32Array([0, 1]) };
}

describe("AudioTransport", () => {
  it("drops new frames while congested until the low watermark", () => {
    const socket = { bufferedAmount: 0, readyState: 1, send: vi.fn() };
    const transport = new AudioTransport(socket);
    socket.bufferedAmount = AUDIO_WS_HIGH_WATERMARK_BYTES + 1;
    expect(transport.sendFrame(frame())).toBe(false);
    socket.bufferedAmount = AUDIO_WS_HIGH_WATERMARK_BYTES - 1;
    expect(transport.sendFrame(frame())).toBe(false);
    socket.bufferedAmount = AUDIO_WS_LOW_WATERMARK_BYTES - 1;
    expect(transport.sendFrame(frame())).toBe(true);
    expect(socket.send).toHaveBeenCalledTimes(1);
    expect(transport.telemetry.browser_frames_produced).toBe(3);
    expect(transport.telemetry.browser_frames_dropped).toBe(2);
  });
});
