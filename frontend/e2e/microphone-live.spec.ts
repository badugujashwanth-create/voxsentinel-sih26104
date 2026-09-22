import { expect, test } from "@playwright/test";

const LIVE_MIC_ENABLED = process.env.JASH005_LIVE_MIC_ACCEPTANCE === "1";
const LIVE_MIC_AUDIO = process.env.JASH005_LIVE_MIC_AUDIO;
test.skip(!LIVE_MIC_ENABLED || !LIVE_MIC_AUDIO, "Set JASH005_LIVE_MIC_ACCEPTANCE=1 and JASH005_LIVE_MIC_AUDIO to a real controlled WAV");

test("streams browser microphone audio through the live ML path", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => { if (message.type() === "error") consoleErrors.push(message.text()); });
  page.on("pageerror", (error) => consoleErrors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: /High-Value Transfer Attack/i }).click();
  await page.getByRole("button", { name: /Start analysis/i }).click();
  await expect(page.getByText("MIC STREAMING", { exact: true })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText("LIVE CALL", { exact: true })).toBeVisible();
  await expect(page.getByText("LIVE AUDIO ANALYSIS", { exact: true })).toBeVisible();
  await expect(page.getByText("RAW AUDIO NOT RETAINED", { exact: true })).toBeVisible();
  await expect.poll(async () => await page.evaluate(() => window.__VOXSENTINEL_ACCEPTANCE__?.inferences.length ?? 0), { timeout: 30_000 }).toBeGreaterThan(0);
  if (process.env.JASH006_LIVE_FUSION_ACCEPTANCE === "1") {
    await expect.poll(async () => await page.evaluate(() => window.__VOXSENTINEL_ACCEPTANCE__?.inferences.some((inference) => inference.speaker_similarity !== null && inference.speaker_similarity !== undefined) ?? false), { timeout: 30_000 }).toBe(true);
    const latestSimilarity = await page.evaluate(() => window.__VOXSENTINEL_ACCEPTANCE__?.inferences.at(-1)?.speaker_similarity ?? null);
    expect(latestSimilarity).not.toBeNull();
    const displayedSimilarity = await page.locator(".evidence-row").filter({ hasText: "Speaker identity" }).locator("strong").innerText();
    expect(Number(displayedSimilarity)).toBeCloseTo(latestSimilarity as number, 2);
    await expect(page.getByText("Uncalibrated speaker similarity", { exact: true })).toBeVisible();
    await expect(page.getByText(/probability|confidence|% fake/i)).toHaveCount(0);
  }
  await page.getByRole("button", { name: /Stop/i }).first().click();
  await expect.poll(async () => await page.evaluate(() => window.__VOXSENTINEL_ACCEPTANCE__?.microphone_state ?? "UNKNOWN"), { timeout: 10_000 }).toBe("IDLE");
  expect(consoleErrors).toEqual([]);
});
