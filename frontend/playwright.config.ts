import { defineConfig, devices } from "@playwright/test";

const PLAYWRIGHT_PORT = process.env.PLAYWRIGHT_PORT ?? "4173";
const PLAYWRIGHT_BASE_URL = process.env.PLAYWRIGHT_BASE_URL ?? `http://127.0.0.1:${PLAYWRIGHT_PORT}`;

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
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
