import { describe, expect, it, vi } from "vitest";
import { MicrophoneSession, type MicrophoneSessionDependencies } from "./microphone-session";

function dependencies() {
  const tracks = [{ stop: vi.fn() }];
  const stream = { getTracks: () => tracks } as unknown as MediaStream;
  const socket = {
    readyState: 1,
    bufferedAmount: 0,
    send: vi.fn((message: string) => {
      if (message.includes("audio_start")) queueMicrotask(() => socket.onmessage?.({ data: JSON.stringify({ type: "audio_ready" }) } as MessageEvent));
    }),
    close: vi.fn(),
    onopen: null as (() => void) | null,
    onmessage: null as ((event: MessageEvent) => void) | null,
    onerror: null as (() => void) | null,
    onclose: null as (() => void) | null,
  };
  const context = { sampleRate: 48_000, state: "running", close: vi.fn().mockResolvedValue(undefined), resume: vi.fn().mockResolvedValue(undefined) } as unknown as AudioContext;
  const bridge = { start: vi.fn().mockResolvedValue(undefined), flush: vi.fn(), dispose: vi.fn() };
  const deps: MicrophoneSessionDependencies = {
    getUserMedia: vi.fn().mockResolvedValue(stream),
    createAudioContext: vi.fn().mockReturnValue(context),
    createSocket: vi.fn().mockImplementation(() => {
      queueMicrotask(() => socket.onopen?.());
      return socket;
    }),
    createBridge: vi.fn().mockReturnValue(bridge),
  };
  return { deps, tracks, socket, context, bridge };
}

describe("MicrophoneSession", () => {
  it("does not expose STREAMING until backend audio_ready", async () => {
    const setup = dependencies();
    const states: string[] = [];
    const session = new MicrophoneSession(setup.deps, (state) => states.push(state));
    const start = session.start("call-1");
    await Promise.resolve();
    expect(states).toContain("CONNECTING");
    await start;
    expect(states.at(-1)).toBe("STREAMING");
    expect(setup.bridge.start).toHaveBeenCalledOnce();
  });

  it("rolls back tracks, socket, bridge, and context after startup failure", async () => {
    const setup = dependencies();
    setup.deps.createBridge = vi.fn(() => { throw new Error("worklet failed"); });
    const session = new MicrophoneSession(setup.deps);
    await expect(session.start("call-1")).rejects.toThrow("worklet failed");
    expect(setup.tracks[0].stop).toHaveBeenCalledOnce();
    expect(setup.socket.close).toHaveBeenCalledOnce();
    expect(setup.context.close).toHaveBeenCalledOnce();
    expect(session.state).toBe("ERROR");
  });

  it("stop is idempotent and releases the microphone resources", async () => {
    const setup = dependencies();
    const session = new MicrophoneSession(setup.deps);
    await session.start("call-1");
    await session.stop();
    await session.stop();
    expect(setup.tracks[0].stop).toHaveBeenCalledOnce();
    expect(setup.bridge.dispose).toHaveBeenCalledOnce();
    expect(setup.context.close).toHaveBeenCalledOnce();
  });

  it("fails and cleans up when the active audio socket disconnects", async () => {
    const setup = dependencies();
    const session = new MicrophoneSession(setup.deps);
    await session.start("call-1");
    setup.socket.onclose?.();
    await Promise.resolve();
    expect(session.state).toBe("ERROR");
    expect(setup.tracks[0].stop).toHaveBeenCalledOnce();
  });
});
