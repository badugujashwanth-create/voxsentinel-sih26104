import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { ScenarioLauncher } from "../components/demo/ScenarioLauncher";

describe("App", () => {
  it("renders the VoxSentinel scenario launcher", () => {
    render(<App />);
    expect(screen.getByText("VoxSentinel")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /choose an incident/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /High-Value Transfer Attack/i })).toBeInTheDocument();
  });

  it("allows idle mode selection without starting microphone capture", async () => {
    const user = userEvent.setup();
    const onModeChange = vi.fn();
    let mode: "DEMO" | "LIVE" = "DEMO";
    const rerenderLauncher = () => render(<ScenarioLauncher mode={mode} onModeChange={(nextMode) => { mode = nextMode; onModeChange(nextMode); }} onSelect={vi.fn()} />);
    const { rerender } = rerenderLauncher();

    const liveMode = screen.getByRole("button", { name: "Live mode" });
    expect(liveMode).toBeEnabled();
    await user.click(liveMode);

    expect(onModeChange).toHaveBeenCalledWith("LIVE");
    rerender(<ScenarioLauncher mode={mode} onModeChange={(nextMode) => { mode = nextMode; onModeChange(nextMode); }} onSelect={vi.fn()} />);
    expect(screen.getByText(/LIVE MODE/)).toBeInTheDocument();
    const demoMode = screen.getByRole("button", { name: "Demo mode" });
    expect(demoMode).toBeEnabled();
    await user.click(demoMode);
    expect(onModeChange).toHaveBeenCalledWith("DEMO");
    rerender(<ScenarioLauncher mode={mode} onModeChange={(nextMode) => { mode = nextMode; onModeChange(nextMode); }} onSelect={vi.fn()} />);
    expect(screen.getByText(/DEMO MODE/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Live mode" })).toBeEnabled();
  });
});
