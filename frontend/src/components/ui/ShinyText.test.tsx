import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ShinyText } from "./ShinyText";

describe("ShinyText", () => {
  it("renders its accessible brand text without creating a second visual label", () => {
    render(<ShinyText text="VoxSentinel" />);

    expect(screen.getByText("VoxSentinel")).toHaveClass("shiny-text");
    expect(screen.getByText("VoxSentinel")).toHaveAttribute("aria-label", "VoxSentinel");
  });
});
