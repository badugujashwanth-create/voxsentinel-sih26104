import { expect, test } from "@playwright/test";

test.skip(process.env.VITE_DEMO_MODE !== "false", "Live backend acceptance runs with VITE_DEMO_MODE=false");

test("streams real ML evidence from a backend call session", async ({ page }) => {
  const viewport = process.env.JASH004_VIEWPORT?.split("x").map(Number);
  if (viewport?.length === 2 && viewport.every(Number.isFinite)) await page.setViewportSize({ width: viewport[0], height: viewport[1] });
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: /High-Value Transfer Attack/i }).click();
  const createResponsePromise = page.waitForResponse((response) => response.request().method() === "POST" && new URL(response.url()).pathname.replace(/\/$/, "") === "/api/v1/calls");
  await page.getByRole("button", { name: /Start analysis/i }).click();
  await createResponsePromise;
  await expect(page.getByText("MIC STREAMING", { exact: true })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("region", { name: "Call context" }).getByText("LIVE CALL", { exact: true })).toBeVisible();

  const observedScores: number[] = [];
  const scoreLocator = page.locator(".context-risk strong");
  for (const expectedScore of [20, 70]) {
    await expect.poll(async () => Number(await scoreLocator.innerText()), { timeout: 5_000 }).toBe(expectedScore);
    observedScores.push(expectedScore);
  }

  expect(observedScores).toEqual([20, 70]);
  await expect(page.getByText("HIGH", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("N/A", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Not evaluated", { exact: true }).first()).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
