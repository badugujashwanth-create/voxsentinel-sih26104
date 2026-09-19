import { describe, expect, it, vi } from "vitest";
import { AudioWorkletBridge } from "./audio-worklet-bridge";

describe("AudioWorkletBridge", () => {
  it("loads, connects, and forwards worklet frames", async () => {
    const node = { connect: vi.fn(), disconnect: vi.fn(), port: { postMessage: vi.fn(), onmessage: null as ((event: MessageEvent) => void) | null } };
    const context = { audioWorklet: { addModule: vi.fn().mockResolvedValue(undefined) }, destination: {} } as unknown as AudioContext;
    vi.stubGlobal("AudioWorkletNode", vi.fn(() => node));
    const frames: bigint[] = [];
    const bridge = new AudioWorkletBridge(context, 1, (frame) => frames.push(frame.firstSampleFrame));
    await bridge.start();
    node.port.onmessage?.({ data: { type: "frame", firstSampleFrame: 1024n, channels: 1, samples: new Float32Array([0]) } } as MessageEvent);
    expect(context.audioWorklet.addModule).toHaveBeenCalledOnce();
    expect(node.connect).toHaveBeenCalledWith(context.destination);
    expect(frames).toEqual([1024n]);
    bridge.dispose();
    expect(node.disconnect).toHaveBeenCalledOnce();
  });
});
