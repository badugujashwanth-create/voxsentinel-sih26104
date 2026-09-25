import { defineConfig, devices } from "@playwright/test";

const PLAYWRIGHT_PORT = process.env.PLAYWRIGHT_PORT ?? "4173";
const PLAYWRIGHT_BASE_URL = process.env.PLAYWRIGHT_BASE_URL ?? `http://127.0.0.1:${PLAYWRIGHT_PORT}`;
const LIVE_MIC_AUDIO = process.env.JASH005_LIVE_MIC_AUDIO ?? process.env.JASH004_ACCEPTANCE_AUDIO;
const LIVE_MIC_ENABLED = process.env.JASH005_LIVE_MIC_ACCEPTANCE === "1" || Boolean(process.env.JASH004_ACCEPTANCE_AUDIO);
const LIVE_MIC_ARGS = LIVE_MIC_AUDIO
  ? [`--use-file-for-fake-audio-capture=${LIVE_MIC_AUDIO}`]
  : ["--use-fake-device-for-media-stream"];

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: "html",
  use: {
    baseURL: PLAYWRIGHT_BASE_URL,
    trace: "on-first-retry",
  },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${PLAYWRIGHT_PORT}`,
    url: PLAYWRIGHT_BASE_URL,
    reuseExistingServer: !process.env.CI,
  },
  projects: [{
    name: "chromium",
    use: {
      ...devices["Desktop Chrome"],
      permissions: LIVE_MIC_ENABLED ? ["microphone"] : undefined,
      launchOptions: LIVE_MIC_ENABLED
        ? { args: ["--use-fake-ui-for-media-stream", ...LIVE_MIC_ARGS] }
        : undefined,
    },
  }],
});
