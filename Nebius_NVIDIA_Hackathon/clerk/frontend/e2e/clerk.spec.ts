import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DEMO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../backend/clerk/demo");

async function loadDemo(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("welcome")).toBeVisible();
  // Against a deployment (CLERK_E2E_URL) the badge may say LIVE; locally it must say OFFLINE.
  await expect(page.getByTestId("mode-badge")).toContainText(process.env.CLERK_E2E_URL ? /LIVE|OFFLINE/ : /OFFLINE SIMULATOR/);
  await page.getByRole("button", { name: "Try it with Maria's paperwork" }).click();
  await expect(page.getByTestId("doc-list-bill").getByRole("button")).toHaveCount(3);
  await expect(page.getByTestId("doc-list-contract").getByRole("button")).toHaveCount(3);
}

async function trial(page: Page, provider: string) {
  await page.getByTestId("doc-list-bill").getByRole("button", { name: new RegExp(provider) }).click();
  await page.getByTestId("run-tribunal").click();
  await expect(page.getByTestId("verdict-banner")).toBeVisible();
}

test("Northwind: overcharge and add-on upheld, included router dismissed", async ({ page }) => {
  await loadDemo(page);
  await trial(page, "Northwind Fiber");
  await expect(page.getByTestId("recoverable")).toHaveText("$39.99");
  await expect(page.locator('[data-testid="finding"][data-verdict="upheld"]')).toHaveCount(2);
  await expect(page.locator('[data-testid="finding"][data-verdict="dismissed"]')).toHaveCount(1);
  for (const stage of ["reader", "vault", "auditors", "verifier", "skeptic", "researcher", "advocate"]) {
    await expect(page.getByTestId(`stage-${stage}`)).toHaveAttribute("data-status", /done|skipped/);
  }
  await expect(page.getByTestId("score-total")).toHaveText("$39.99");
  await expect(page.getByTestId("live-trace")).toContainText("model calls");
});

test("Brightside and Lakeview: three errors found, then a clean bill", async ({ page }) => {
  await loadDemo(page);
  await trial(page, "Brightside Energy");
  await expect(page.getByTestId("recoverable")).toHaveText("$34.50");
  await expect(page.locator('[data-testid="finding"][data-verdict="upheld"]')).toHaveCount(3);
  await trial(page, "Lakeview Apartments");
  await expect(page.getByTestId("recoverable")).toHaveText("This bill checks out.");
  await expect(page.locator('[data-testid="finding"][data-verdict="dismissed"]')).toHaveCount(4);
  await expect(page.getByTestId("review-actions")).toHaveCount(0);
  await expect(page.getByTestId("score-total")).toHaveText("$34.50");
});

test("saved verdicts reappear when you come back to a bill", async ({ page }) => {
  await loadDemo(page);
  await trial(page, "Northwind Fiber");
  await page.getByTestId("doc-list-bill").getByRole("button", { name: /Brightside/ }).click();
  await expect(page.getByTestId("verdict-banner")).toHaveCount(0);
  await page.getByTestId("doc-list-bill").getByRole("button", { name: /Northwind/ }).click();
  await expect(page.getByTestId("recoverable")).toHaveText("$39.99");
});

test("actions must be approved before they can be downloaded", async ({ page }) => {
  await loadDemo(page);
  await trial(page, "Northwind Fiber");
  await page.getByTestId("review-actions").click();
  const letter = page.locator('[data-testid="action"][data-type="letter"]');
  await expect(letter).toHaveAttribute("data-status", "pending");
  await expect(letter.getByTestId("download")).toHaveCount(0);
  await expect(letter.getByTestId("action-content")).toContainText("credit $39.99");
  await letter.getByTestId("approve").click();
  await expect(letter).toHaveAttribute("data-status", "approved");
  const download = page.waitForEvent("download");
  await letter.getByTestId("download").click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("dispute-northwind-fiber-2026-03-05.txt");
  expect(readFileSync(await file.path(), "utf-8")).toContain("Premium Tech Support add-on: $9.99");
  const reminder = page.locator('[data-testid="action"][data-type="reminder"]');
  await reminder.getByTestId("reject").click();
  await expect(reminder).toHaveAttribute("data-status", "rejected");
  await expect(page.getByTestId("tab-actions")).toHaveText("Actions (1)");
});

test("editing the Vault changes the next verdict", async ({ page }) => {
  await loadDemo(page);
  await page.getByTestId("tab-vault").click();
  const rate = page.getByTestId("vault").locator("div.card", { hasText: "Northwind Fiber" }).getByTestId("fact-monthly_rate");
  await expect(rate).toContainText("39.99");
  await rate.getByRole("button", { name: "Edit" }).click();
  await page.getByTestId("fact-edit-input").fill("69.99");
  await page.getByTestId("fact-save").click();
  await expect(rate).toContainText("69.99");
  await page.getByTestId("ask-input").fill("What is the Northwind monthly rate?");
  await page.getByTestId("ask-button").click();
  await expect(page.getByTestId("ask-answer")).toContainText("Northwind Fiber");
  await page.getByTestId("tab-tribunal").click();
  await trial(page, "Northwind Fiber");
  await expect(page.getByTestId("recoverable")).toHaveText("$9.99");
});

test("uploading a bill puts it on trial automatically", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles([
    path.join(DEMO, "02_brightside_agreement.txt"),
    path.join(DEMO, "12_brightside_april_statement.txt"),
  ]);
  await expect(page.getByTestId("recoverable")).toHaveText("$34.50");
});

test("a bill with no agreement on file still gets the checks it can", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles(path.join(DEMO, "12_brightside_april_statement.txt"));
  await expect(page.getByTestId("stage-vault")).toContainText("No agreement on file");
  // Duplicate and arithmetic errors need no agreement; the full $35 fee is disputed without a cap on file.
  await expect(page.getByTestId("recoverable")).toHaveText("$49.50");
});

test("bad uploads show a clear error", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles({ name: "virus.exe", mimeType: "application/octet-stream", buffer: Buffer.from("MZ") });
  await expect(page.getByTestId("error")).toContainText("Clerk reads .txt, .pdf");
  await page.getByTestId("file-input").setInputFiles({ name: "bill.png", mimeType: "image/png", buffer: Buffer.from("\x89PNG") });
  await expect(page.getByTestId("error")).toContainText("live mode");
});

test("turning off skills is respected", async ({ page }) => {
  await loadDemo(page);
  await page.getByTestId("tab-permissions").click();
  await page.getByTestId("perm-draft_letters").uncheck();
  await page.getByTestId("perm-reminders").uncheck();
  await page.getByTestId("tab-tribunal").click();
  await trial(page, "Northwind Fiber");
  await expect(page.getByTestId("stage-advocate")).toContainText("0 actions");
  await expect(page.getByTestId("review-actions")).toHaveCount(0);
  await page.getByTestId("tab-permissions").click();
  await expect(page.getByTestId("perm-draft_letters")).not.toBeChecked();
});

test("trace panel shows every model call", async ({ page }) => {
  await loadDemo(page);
  await trial(page, "Brightside Energy");
  await page.getByTestId("tab-trace").click();
  await expect(page.getByTestId("trace")).toContainText("Skeptic");
  await expect(page.getByTestId("trace")).toContainText("offline-sim/nemotron-super-standin");
});

test("forget everything wipes the workspace", async ({ page }) => {
  await loadDemo(page);
  await page.getByTestId("tab-vault").click();
  page.once("dialog", (d) => d.accept());
  await page.getByTestId("wipe").click();
  await page.getByTestId("tab-tribunal").click();
  await expect(page.getByTestId("welcome")).toBeVisible();
  await expect(page.getByTestId("doc-list-bill")).toHaveCount(0);
});

test("workspaces are private to each browser", async ({ browser }) => {
  const alice = await browser.newPage();
  await loadDemo(alice);
  const bob = await (await browser.newContext()).newPage();
  await bob.goto("/");
  await expect(bob.getByTestId("welcome")).toBeVisible();
});

test("phone-sized screens have no horizontal scroll", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });
  await loadDemo(page);
  await trial(page, "Northwind Fiber");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
