import { defineConfig, devices } from "@playwright/test";

// SPEC.md 10.1 / 12.2: smoke test hits the deployed URL, not local dev.
// PLAYWRIGHT_BASE_URL is set in CI to the live Vercel deployment; falls
// back to localhost for running the suite locally against `pnpm dev`.
const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000";

export default defineConfig({
  testDir: "./tests",
  timeout: 45_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "list" : "html",
  use: {
    baseURL,
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
});
