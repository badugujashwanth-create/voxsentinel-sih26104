import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("App", () => {
  it("renders the VoxSentinel scenario launcher", () => {
    render(<App />);
    expect(screen.getByText("VoxSentinel")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /choose an incident/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /High-Value Transfer Attack/i })).toBeInTheDocument();
  });
});
