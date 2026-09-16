import { describe, expect, it, vi } from "vitest";
import { CallSessionClient, CallSessionError, type CallSessionResponse } from "./CallSessionClient";

const session: CallSessionResponse = {
  call_id: "real-call-42",
  status: "LIVE",
  claimed_identity: "Arjun Mehta",
  scenario: "HIGH_VALUE_TRANSFER_ATTACK",
  transaction_value: 2500000,
  currency: "INR",
  created_at: "2026-09-16T00:00:00Z",
  started_at: "2026-09-16T00:00:01Z",
  completed_at: null,
};

function response(body: unknown, ok = true, status = 200): Response {
  return { ok, status, json: async () => body } as Response;
}

describe("CallSessionClient", () => {
  it("creates, starts, and stops a call with the exact backend contract", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response({ call_id: "real-call-42", status: "CREATED", claimed_identity: "Arjun Mehta", scenario: "HIGH_VALUE_TRANSFER_ATTACK" }, true, 201))
      .mockResolvedValueOnce(response(session))
      .mockResolvedValueOnce(response({ ...session, status: "COMPLETED", completed_at: "2026-09-16T00:00:10Z" }));
    const client = new CallSessionClient({ baseUrl: "http://backend.test", fetcher });

    await expect(client.createCall({ claimed_identity: "Arjun Mehta", scenario: "HIGH_VALUE_TRANSFER_ATTACK", transaction_value: 2500000, currency: "INR" })).resolves.toMatchObject({ call_id: "real-call-42" });
    await client.startCall("real-call-42");
    await client.stopCall("real-call-42");

    expect(fetcher.mock.calls.map(([url, init]) => [url, (init as RequestInit).method, (init as RequestInit).body])).toEqual([
      ["http://backend.test/api/v1/calls", "POST", JSON.stringify({ claimed_identity: "Arjun Mehta", scenario: "HIGH_VALUE_TRANSFER_ATTACK", transaction_value: 2500000, currency: "INR" })],
      ["http://backend.test/api/v1/calls/real-call-42/start", "POST", undefined],
      ["http://backend.test/api/v1/calls/real-call-42/stop", "POST", undefined],
    ]);
  });

  it("normalizes backend failures without hiding their status", async () => {
    const client = new CallSessionClient({ baseUrl: "http://backend.test", fetcher: vi.fn().mockResolvedValue(response({ detail: "Call unavailable" }, false, 503)) });

    await expect(client.startCall("real-call-42")).rejects.toMatchObject({ name: "CallSessionError", status: 503, message: "Call unavailable" });
  });

  it("reports network failures as CallSessionError", async () => {
    const client = new CallSessionClient({ baseUrl: "http://backend.test", fetcher: vi.fn().mockRejectedValue(new Error("network down")) });

    await expect(client.createCall({ claimed_identity: "Arjun Mehta", scenario: "GENUINE" })).rejects.toBeInstanceOf(CallSessionError);
  });
});
