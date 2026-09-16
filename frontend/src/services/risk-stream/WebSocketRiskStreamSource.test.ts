import { beforeEach, describe, expect, it, vi } from "vitest";
import { WebSocketRiskStreamSource } from "./WebSocketRiskStreamSource";

describe("WebSocketRiskStreamSource", () => {
  beforeEach(() => {
    vi.stubGlobal("WebSocket", class {
      public static OPEN = 1;
      public readyState = 0;
      public url: string;
      public onopen: (() => void) | null = null;
      public onmessage: ((event: MessageEvent) => void) | null = null;
      public onerror: (() => void) | null = null;
      public onclose: (() => void) | null = null;
      public constructor(url: string) { this.url = url; }
      public send(): void { }
      public close(): void { this.onclose?.(); }
    });
  });

  it("delivers validated messages and reports malformed payloads", async () => {
    const source = new WebSocketRiskStreamSource("call-17", "https://api.example.test/");
    const onEvent = vi.fn();
    const onStatusChange = vi.fn();
    const onError = vi.fn();
    await source.start({ onEvent, onStatusChange, onError });
    const socket = (source as unknown as { socket: { url: string; onopen: () => void; onmessage: (event: MessageEvent) => void; onclose: () => void } }).socket;

    expect(socket.url).toBe("wss://api.example.test/api/v1/calls/call-17/risk-stream");
    socket.onopen();
    socket.onmessage(new MessageEvent("message", { data: JSON.stringify({ call_id: "call-17", sequence: 1, timestamp_ms: 100, synthetic_probability: 0.1, speaker_match_score: 0.9, speaker_mismatch_score: 0.1, prosody_anomaly_score: 0.1, replay_risk_score: 0.1, context_risk_score: 0.1, overall_risk_score: 10, risk_level: "LOW", reasons: ["baseline"], recommended_action: "MONITOR" }) }));
    socket.onmessage(new MessageEvent("message", { data: "malformed" }));

    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(1);
    socket.onclose();
    expect(onStatusChange).toHaveBeenCalledWith("DISCONNECTED");
    source.dispose();
  });
});
