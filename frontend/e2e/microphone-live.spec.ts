import { expect, test } from "@playwright/test";

test.skip(process.env.JASH005_LIVE_MIC_ACCEPTANCE !== "1", "Set JASH005_LIVE_MIC_ACCEPTANCE=1 with backend ML and a browser microphone");

test("streams browser microphone audio through the live ML path", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => { if (message.type() === "error") consoleErrors.push(message.text()); });
  page.on("pageerror", (error) => consoleErrors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: /High-Value Transfer Attack/i }).click();
  await page.getByRole("button", { name: /Start analysis/i }).click();
  await expect(page.getByText("MIC STREAMING", { exact: true })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText("LIVE CALL", { exact: true })).toBeVisible();
  await expect(page.getByText("N/A", { exact: true }).first()).toBeVisible({ timeout: 15_000 });
  expect(consoleErrors).toEqual([]);
});
