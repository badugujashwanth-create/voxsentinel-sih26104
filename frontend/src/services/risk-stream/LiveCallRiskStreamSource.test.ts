import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RiskStreamHandlers } from "./RiskStreamSource";
import { LiveCallRiskStreamSource } from "./LiveCallRiskStreamSource";

const responseSession = { call_id: "backend-call-42", status: "CREATED" as const, claimed_identity: "Arjun Mehta", scenario: "HIGH_VALUE_TRANSFER_ATTACK" as const };

class FakeWebSocket {
  public static OPEN = 1;
  public url: string;
  public onopen: (() => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onerror: (() => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;
  public constructor(url: string) { this.url = url; }
  public close(): void { this.onclose?.({ code: 1000 } as CloseEvent); }
}

function handlers(): RiskStreamHandlers {
  return { onEvent: vi.fn(), onStatusChange: vi.fn(), onError: vi.fn() };
}

describe("LiveCallRiskStreamSource", () => {
  beforeEach(() => vi.stubGlobal("WebSocket", FakeWebSocket));

  it("creates and starts before using the returned call id for WebSocket", async () => {
    const client = { createCall: vi.fn().mockResolvedValue(responseSession), startCall: vi.fn().mockResolvedValue({ ...responseSession, status: "LIVE" }), stopCall: vi.fn().mockResolvedValue({ ...responseSession, status: "COMPLETED" }) };
    const source = new LiveCallRiskStreamSource({ baseUrl: "https://backend.test", request: { claimed_identity: "Arjun Mehta", scenario: "HIGH_VALUE_TRANSFER_ATTACK", transaction_value: 2500000, currency: "INR" } }, client);
    const received = handlers();

    await source.start(received);
    const socket = (source as unknown as { stream: { socket: { url: string } } }).stream;

    expect(client.createCall).toHaveBeenCalledWith({ claimed_identity: "Arjun Mehta", scenario: "HIGH_VALUE_TRANSFER_ATTACK", transaction_value: 2500000, currency: "INR" });
    expect(client.startCall).toHaveBeenCalledWith("backend-call-42");
    expect(socket.socket.url).toBe("wss://backend.test/api/v1/calls/backend-call-42/risk-stream");
    expect(client.createCall.mock.invocationCallOrder[0]).toBeLessThan(client.startCall.mock.invocationCallOrder[0]);
  });

  it("stops the backend session when the source is stopped or disposed", async () => {
    const client = { createCall: vi.fn().mockResolvedValue(responseSession), startCall: vi.fn().mockResolvedValue({ ...responseSession, status: "LIVE" }), stopCall: vi.fn().mockResolvedValue({ ...responseSession, status: "COMPLETED" }) };
    const source = new LiveCallRiskStreamSource({ baseUrl: "https://backend.test", request: { claimed_identity: "Arjun Mehta", scenario: "GENUINE" } }, client);
    await source.start(handlers());

    await source.stop();

    expect(client.stopCall).toHaveBeenCalledWith("backend-call-42");
  });

  it("stops the live session when reset disposes the active source", async () => {
    const client = { createCall: vi.fn().mockResolvedValue(responseSession), startCall: vi.fn().mockResolvedValue({ ...responseSession, status: "LIVE" }), stopCall: vi.fn().mockResolvedValue({ ...responseSession, status: "COMPLETED" }) };
    const source = new LiveCallRiskStreamSource({ request: { claimed_identity: "Arjun Mehta", scenario: "GENUINE" } }, client);
    await source.start(handlers());

    source.reset();
    await vi.waitFor(() => expect(client.stopCall).toHaveBeenCalledWith("backend-call-42"));
  });

  it("stops a call when reset races with backend start", async () => {
    const liveResponse = { ...responseSession, status: "LIVE" as const };
    let resolveStart: ((value: typeof liveResponse) => void) | undefined;
    const startResult = new Promise<typeof liveResponse>((resolve) => { resolveStart = resolve; });
    const client = { createCall: vi.fn().mockResolvedValue(responseSession), startCall: vi.fn().mockReturnValue(startResult), stopCall: vi.fn().mockResolvedValue({ ...responseSession, status: "COMPLETED" }) };
    const source = new LiveCallRiskStreamSource({ request: { claimed_identity: "Arjun Mehta", scenario: "GENUINE" } }, client);

    const startPromise = source.start(handlers());
    await vi.waitFor(() => expect(client.startCall).toHaveBeenCalledWith("backend-call-42"));
    source.reset();
    resolveStart?.(liveResponse);
    await startPromise;
    await vi.waitFor(() => expect(client.stopCall).toHaveBeenCalledWith("backend-call-42"));
  });
  it("reports create failures and never opens a socket", async () => {
    const client = { createCall: vi.fn().mockRejectedValue(new Error("backend unavailable")), startCall: vi.fn(), stopCall: vi.fn() };
    const received = handlers();
    const source = new LiveCallRiskStreamSource({ request: { claimed_identity: "Arjun Mehta", scenario: "GENUINE" } }, client);

    await source.start(received);

    expect(received.onError).toHaveBeenCalledWith(expect.objectContaining({ message: "backend unavailable" }));
    expect(client.startCall).not.toHaveBeenCalled();
  });

  it("starts microphone capture only after the real backend call is live", async () => {
    const client = { createCall: vi.fn().mockResolvedValue(responseSession), startCall: vi.fn().mockResolvedValue({ ...responseSession, status: "LIVE" }), stopCall: vi.fn().mockResolvedValue({ ...responseSession, status: "COMPLETED" }) };
    const microphone = { start: vi.fn().mockResolvedValue(undefined), stop: vi.fn().mockResolvedValue(undefined), reset: vi.fn(), dispose: vi.fn() };
    const source = new LiveCallRiskStreamSource({ request: { claimed_identity: "Arjun Mehta", scenario: "GENUINE" }, microphone }, client);
    await source.start(handlers());
    expect(microphone.start).toHaveBeenCalledWith("backend-call-42");
    await source.stop();
    expect(microphone.stop).toHaveBeenCalledOnce();
  });

  it("rolls back the backend call when microphone startup fails", async () => {
    const client = { createCall: vi.fn().mockResolvedValue(responseSession), startCall: vi.fn().mockResolvedValue({ ...responseSession, status: "LIVE" }), stopCall: vi.fn().mockResolvedValue({ ...responseSession, status: "COMPLETED" }) };
    const microphone = { start: vi.fn().mockRejectedValue(new Error("permission denied")), stop: vi.fn().mockResolvedValue(undefined), reset: vi.fn(), dispose: vi.fn() };
    const source = new LiveCallRiskStreamSource({ request: { claimed_identity: "Arjun Mehta", scenario: "GENUINE" }, microphone }, client);

    await source.start(handlers());

    expect(microphone.stop).toHaveBeenCalledOnce();
    expect(client.stopCall).toHaveBeenCalledWith("backend-call-42");
  });

  it("tears down microphone capture when risk transport disconnects", async () => {
    const client = { createCall: vi.fn().mockResolvedValue(responseSession), startCall: vi.fn().mockResolvedValue({ ...responseSession, status: "LIVE" }), stopCall: vi.fn().mockResolvedValue({ ...responseSession, status: "COMPLETED" }) };
    const microphone = { start: vi.fn().mockResolvedValue(undefined), stop: vi.fn().mockResolvedValue(undefined), reset: vi.fn(), dispose: vi.fn() };
    const source = new LiveCallRiskStreamSource({ request: { claimed_identity: "Arjun Mehta", scenario: "GENUINE" }, microphone }, client);
    await source.start(handlers());
    const socket = (source as unknown as { stream: { socket: FakeWebSocket } }).stream.socket;

    socket.close();
    await vi.waitFor(() => expect(microphone.stop).toHaveBeenCalledOnce());
    expect(client.stopCall).toHaveBeenCalledWith("backend-call-42");
  });
});
