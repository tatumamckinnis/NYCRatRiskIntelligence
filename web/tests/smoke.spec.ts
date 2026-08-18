import { test, expect } from "@playwright/test";

// SPEC.md 10.1: Playwright smoke test hitting the deployed URL that
// (1) loads the map, (2) opens /chat, (3) streams a response,
// (4) asserts at least one citation rendered.

test("loads the map on the landing page", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "NYC Rat Risk Map" })).toBeVisible();
  // MapLibre draws a canvas into the map container once tiles/data are ready.
  await expect(page.locator(".maplibregl-canvas")).toBeVisible({ timeout: 30_000 });
});

test("opens /chat, streams a response, and renders a citation", async ({ page }) => {
  // Render's free tier can cold-start (~30s) before the model responds;
  // give this test more headroom than the suite default.
  test.setTimeout(90_000);
  await page.goto("/chat");
  await expect(page.getByRole("heading", { name: "NYC Regulation Assistant" })).toBeVisible();

  const input = page.getByRole("textbox", { name: "Question input" });
  await input.fill("What does 'active rat signs' mean under the NYC Health Code?");
  await page.getByRole("button", { name: "Send" }).click();

  // Streaming indicator appears, then the Send button re-enables once the
  // stream completes. Render on Render's free tier can cold-start (~30s),
  // so allow generous headroom before the response finishes.
  await expect(page.getByRole("button", { name: "Send" })).toBeEnabled({ timeout: 60_000 });

  // At least one §-citation pill should be rendered in the final answer.
  await expect(page.getByText(/§\d/).first()).toBeVisible();
});
