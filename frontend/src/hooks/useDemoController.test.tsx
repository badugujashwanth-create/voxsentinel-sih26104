import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useDemoController } from "./useDemoController";

describe("useDemoController", () => {
  afterEach(() => vi.useRealTimers());

  it("streams the selected transfer scenario into a critical blocked state", async () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useDemoController());

    act(() => result.current.selectScenario("HIGH_VALUE_TRANSFER_ATTACK"));
    await act(async () => result.current.start());
    await act(async () => vi.advanceTimersByTimeAsync(4200));

    expect(result.current.presentationState).toBe("CRITICAL_INTERVENTION");
    expect(result.current.voiceRisk.score).toBe(92);
    expect(result.current.voiceRisk.level).toBe("CRITICAL");
    expect(result.current.protectedAction.status).toBe("BLOCKED");
    expect(result.current.timeline).toHaveLength(6);
  });

  it("preserves critical voice risk when callback verification succeeds", async () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useDemoController());

    act(() => result.current.selectScenario("HIGH_VALUE_TRANSFER_ATTACK"));
    await act(async () => result.current.start());
    await act(async () => vi.advanceTimersByTimeAsync(4200));
    act(() => result.current.beginVerification("VERIFIED_CALLBACK"));
    act(() => result.current.completeVerification());

    expect(result.current.voiceRisk.level).toBe("CRITICAL");
    expect(result.current.voiceRisk.score).toBe(92);
    expect(result.current.identityVerification.status).toBe("VERIFIED");
    expect(result.current.protectedAction.status).toBe("ELIGIBLE");
    expect(result.current.currentEvent?.reasons).toContain("Synthetic speech characteristics detected");
  });
});
