import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

test.skip(process.env.VITE_DEMO_MODE !== "false", "Live backend acceptance runs with VITE_DEMO_MODE=false");

test("streams real ML evidence from a backend call session", async ({ page, request }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: /High-Value Transfer Attack/i }).click();
  const createResponsePromise = page.waitForResponse((response) => response.request().method() === "POST" && new URL(response.url()).pathname.replace(/\/$/, "") === "/api/v1/calls");
  await page.getByRole("button", { name: /Start analysis/i }).click();
  const callId = (await (await createResponsePromise).json()).call_id as string;

  const audioPath = process.env.JASH004_ACCEPTANCE_AUDIO;
  if (!audioPath) throw new Error("JASH004_ACCEPTANCE_AUDIO is required for live ML E2E");
  const audioResponse = await request.post(`http://127.0.0.1:8000/api/v1/calls/${callId}/audio`, { data: readFileSync(audioPath), headers: { "Content-Type": "audio/wav", "X-Audio-Chunk-Sequence": "1" } });
  expect(audioResponse.ok()).toBeTruthy();
  const observedScores: number[] = [];
  const scoreLocator = page.locator(".context-risk strong");
  for (const expectedScore of [20, 70]) {
    await expect.poll(async () => Number(await scoreLocator.innerText()), { timeout: 5_000 }).toBe(expectedScore);
    observedScores.push(expectedScore);
  }

  expect(observedScores).toEqual([20, 70]);
  await expect(page.getByText("LIVE CALL", { exact: true })).toBeVisible();
  await expect(page.getByText("HIGH", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("N/A", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Not evaluated", { exact: true }).first()).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
