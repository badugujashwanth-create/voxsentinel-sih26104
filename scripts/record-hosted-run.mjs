import { chromium } from "file:///C:/Users/JASHWANTH/voxsentinel-sih26104/frontend/node_modules/@playwright/test/index.mjs";
import fs from "node:fs";

const audioPath = process.argv[2];
const outputVideo = process.argv[3];
const outputScreenshot = process.argv[4];
const durationMs = Number(process.argv[5] ?? "25000");
const hostedUrl = "https://voxsentinel.vercel.app";

if (!audioPath || !outputVideo || !outputScreenshot) throw new Error("audio, video, and screenshot paths are required");
fs.mkdirSync(outputVideo.slice(0, outputVideo.lastIndexOf("\\")), { recursive: true });

const browser = await chromium.launch({
  headless: true,
  args: ["--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${audioPath}`],
});
const context = await browser.newContext({
  permissions: ["microphone"],
  viewport: { width: 1920, height: 1080 },
  recordVideo: { dir: outputVideo.slice(0, outputVideo.lastIndexOf("\\")), size: { width: 1920, height: 1080 } },
});
const page = await context.newPage();
const errors = [];
page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
page.on("pageerror", (error) => errors.push(error.message));

await page.goto(hostedUrl, { waitUntil: "networkidle" });
await page.screenshot({ path: outputScreenshot.replace(/\.png$/i, "-idle.png"), fullPage: true });
await page.getByRole("button", { name: /High-Value Transfer Attack/i }).click();
await page.getByRole("button", { name: /Start analysis/i }).click();
await page.getByText("MIC STREAMING", { exact: true }).waitFor({ state: "visible", timeout: 45000 });
await page.waitForTimeout(durationMs);
await page.screenshot({ path: outputScreenshot, fullPage: true });
const inferences = await page.evaluate(() => window.__VOXSENTINEL_ACCEPTANCE__?.inferences.length ?? 0);
const bodyText = await page.locator("body").innerText();
const telemetry = await page.evaluate(async () => {
  const callId = window.__VOXSENTINEL_ACCEPTANCE__?.callId;
  if (!callId) return null;
  const response = await fetch(`https://voxsentinel-backend.onrender.com/api/v1/calls/${callId}/acceptance-telemetry`);
  return response.ok ? response.json() : { status: response.status };
});
const stop = page.getByRole("button", { name: /Stop/i }).first();
if (await stop.isVisible()) await stop.click();
await page.waitForTimeout(1000);
await context.close();
await browser.close();

const recorded = fs.readdirSync(outputVideo.slice(0, outputVideo.lastIndexOf("\\")))
  .map((name) => outputVideo.slice(0, outputVideo.lastIndexOf("\\")) + "\\" + name)
  .filter((name) => name.toLowerCase().endsWith(".webm"))
  .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
fs.renameSync(recorded, outputVideo);
console.log(JSON.stringify({ hostedUrl, outputVideo, outputScreenshot, inferences, errors, telemetry, hasAasist: bodyText.includes("Uncalibrated spoof evidence"), hasEcapa: bodyText.includes("Uncalibrated speaker similarity"), bodyExcerpt: bodyText.slice(-3000) }));
