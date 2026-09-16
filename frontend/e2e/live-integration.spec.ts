import { expect, test } from "@playwright/test";

test.skip(process.env.VITE_DEMO_MODE !== "false", "Live backend acceptance runs with VITE_DEMO_MODE=false");

test("streams the high-value transfer scenario from a real backend call session", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: /High-Value Transfer Attack/i }).click();
  await page.getByRole("button", { name: /Start analysis/i }).click();

  const observedScores: number[] = [];
  const scoreLocator = page.locator(".context-risk strong");
  for (const expectedScore of [18, 27, 43, 61, 79, 92]) {
    await expect.poll(async () => Number(await scoreLocator.innerText()), { timeout: 5_000 }).toBe(expectedScore);
    observedScores.push(expectedScore);
  }

  expect(observedScores).toEqual([18, 27, 43, 61, 79, 92]);
  await expect(page.getByText("LIVE CALL", { exact: true })).toBeVisible();
  await expect(page.getByText("TRANSFER TEMPORARILY BLOCKED", { exact: true })).toBeVisible();
  await expect(page.getByText("CRITICAL", { exact: true }).first()).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
