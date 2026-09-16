import { describe, expect, it, vi } from "vitest";
import { MockRiskStreamSource } from "./MockRiskStreamSource";

describe("MockRiskStreamSource", () => {
  it("emits ordered events and pauses/resumes deterministically", async () => {
    vi.useFakeTimers();
    const onEvent = vi.fn();
    const onStatusChange = vi.fn();
    const source = new MockRiskStreamSource("HIGH_VALUE_TRANSFER_ATTACK", 100);

    await source.start({ onEvent, onStatusChange, onError: vi.fn() });
    await vi.advanceTimersByTimeAsync(100);
    expect(onEvent.mock.calls[0][0].sequence).toBe(1);

    source.pause();
    await vi.advanceTimersByTimeAsync(400);
    expect(onEvent).toHaveBeenCalledTimes(1);
    source.resume();
    await vi.advanceTimersByTimeAsync(100);
    expect(onEvent.mock.calls[1][0].sequence).toBe(2);

    source.dispose();
    vi.useRealTimers();
  });
});
