import { expect, test } from "@playwright/test";

test.skip(process.env.VITE_DEMO_MODE === "false", "Canonical scenario is offline demo-only");

test("runs the canonical high-value transfer protection flow", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /High-Value Transfer Attack/i }).click();
  await expect(page.getByText("CALL READY", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: /Start analysis/i }).click();

  await expect(page.getByText("LIVE CALL", { exact: true })).toBeVisible();
  await expect(page.getByText("18").first()).toBeVisible();
  await expect(page.getByText("79").first()).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText("92").first()).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText("CRITICAL", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("TRANSFER TEMPORARILY BLOCKED")).toBeVisible();
  await expect(page.getByText("Synthetic speech characteristics detected", { exact: true })).toBeVisible();
  await expect(page.getByText("Speaker identity mismatch", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /VERIFY IDENTITY/i }).click();
  await page.getByRole("button", { name: /Verified Callback/i }).click();
  await expect(page.getByRole("heading", { name: /Callback initiated/i })).toBeVisible();
  await expect(page.getByText("SECURE CHANNEL ESTABLISHED").last()).toBeVisible();
  await expect(page.getByText("IDENTITY CONFIRMED").last()).toBeVisible();
  await page.getByRole("button", { name: /Simulate success/i }).click();

  await expect(page.getByText("IDENTITY VERIFIED THROUGH SECONDARY CHANNEL", { exact: true })).toBeVisible();
  await expect(page.getByText("Voice authenticity remains")).toBeVisible();
  await expect(page.getByText("Protected action is now eligible to proceed according to policy.", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("VOICE AUTHENTICITY", { exact: true })).toBeVisible();
  await expect(page.getByText("IDENTITY", { exact: true })).toBeVisible();
  await expect(page.getByText("ACTION", { exact: true })).toBeVisible();
  await expect(page.getByText("92", { exact: true }).last()).toBeVisible();
});
