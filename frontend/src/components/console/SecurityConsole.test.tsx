import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { initialVerificationState } from "../../domain/verification";
import type { DemoController } from "../../hooks/useDemoController";
import { SecurityConsole } from "./SecurityConsole";

const liveEvent = {
  call_id: "live-call-1",
  sequence: 1,
  timestamp_ms: 1000,
  synthetic_probability: 0.1,
  speaker_match_score: 0,
  speaker_mismatch_score: 0,
  prosody_anomaly_score: 0,
  replay_risk_score: 0,
  context_risk_score: 0,
  overall_risk_score: 20,
  risk_level: "LOW" as const,
  reasons: ["Voice signal remains within baseline"],
  recommended_action: "MONITOR" as const,
  synthetic_score_semantics: "uncalibrated" as const,
  evidence_availability: {
    speaker_match_score: "NOT_EVALUATED" as const,
    speaker_mismatch_score: "NOT_EVALUATED" as const,
    prosody_anomaly_score: "NOT_EVALUATED" as const,
    replay_risk_score: "NOT_EVALUATED" as const,
    context_risk_score: "NOT_EVALUATED" as const,
  },
};

function buildDisconnectedMicrophoneController(stop: () => Promise<void>): DemoController {
  return { ...buildController(stop), streamStatus: 'DISCONNECTED' };
}

function buildController(stop: () => Promise<void>): DemoController {
  return {
    mode: "LIVE",
    selectedScenario: "GENUINE",
    context: { caller: "Arjun Mehta", role: "Chief Financial Officer", request: "Vendor Transfer" },
    presentationState: "CALM",
    streamStatus: "LIVE",
    streamError: null,
    currentEvent: liveEvent,
    timeline: [liveEvent],
    voiceRisk: { score: 20, level: "LOW", reasons: liveEvent.reasons },
    identityVerification: initialVerificationState,
    protectedAction: { status: "MONITORING", label: "Transaction monitoring" },
    selectScenario: vi.fn(),
    selectMode: vi.fn(),
    start: vi.fn(),
    stop,
    pause: vi.fn(),
    resume: vi.fn(),
    reset: vi.fn(),
    beginVerification: vi.fn(),
    completeVerification: vi.fn(),
    failVerification: vi.fn(),
    callId: liveEvent.call_id,
    liveMode: true,
    microphoneState: "STREAMING",
  };
}

describe("SecurityConsole live controls", () => {
  it("provides an explicit stop action for live microphone sessions", async () => {
    const user = userEvent.setup();
    const stop = vi.fn().mockResolvedValue(undefined);
    render(<SecurityConsole controller={buildController(stop)} verificationOpen={false} onOpenVerification={vi.fn()} onCloseVerification={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Stop analysis" }));

    expect(stop).toHaveBeenCalledOnce();
  });

  it('keeps stop available when the live risk stream is terminal but the microphone is active', async () => {
    const stop = vi.fn().mockResolvedValue(undefined);
    render(<SecurityConsole controller={buildDisconnectedMicrophoneController(stop)} verificationOpen={false} onOpenVerification={vi.fn()} onCloseVerification={vi.fn()} />);

    expect(screen.getByRole('button', { name: 'Stop analysis' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Resume' })).not.toBeInTheDocument();
  });
});

describe("SecurityConsole speaker evidence", () => {
  it("does not describe live signal annotations as confidence", () => {
    render(<SecurityConsole controller={buildController(vi.fn().mockResolvedValue(undefined))} verificationOpen={false} onOpenVerification={vi.fn()} onCloseVerification={vi.fn()} />);

    expect(screen.queryByText("CONFIDENCE BAND ACTIVE", { exact: true })).not.toBeInTheDocument();
  });
  it("displays the actual negative cosine similarity", () => {
    const controller = {
      ...buildController(vi.fn().mockResolvedValue(undefined)),
      presentationState: "ACTIVE" as DemoController["presentationState"],
      currentEvent: {
        ...liveEvent,
        speaker_match_score: 0,
        speaker_similarity: -0.1132,
        speaker_score_semantics: "uncalibrated_similarity" as const,
        speaker_state: "INCONSISTENT" as const,
        evidence_availability: {
          ...liveEvent.evidence_availability,
          speaker_match_score: "EVALUATED" as const,
        },
      },
    };
    render(<SecurityConsole controller={controller} verificationOpen={false} onOpenVerification={vi.fn()} onCloseVerification={vi.fn()} />);
    expect(screen.getByText("-0.11")).toBeInTheDocument();
  });
});

describe("SecurityConsole elapsed timer", () => {
  it("renders live epoch timestamps as elapsed call duration", () => {
    const firstEvent = { ...liveEvent, timestamp_ms: 1790268300178 };
    const latestEvent = { ...firstEvent, timestamp_ms: 1790268304750, sequence: 2 };
    const controller = {
      ...buildController(vi.fn().mockResolvedValue(undefined)),
      currentEvent: latestEvent,
      timeline: [firstEvent, latestEvent],
    };

    render(<SecurityConsole controller={controller} verificationOpen={false} onOpenVerification={vi.fn()} onCloseVerification={vi.fn()} />);

    expect(screen.getByText("00:04.57", { exact: true })).toBeInTheDocument();
  });
});
