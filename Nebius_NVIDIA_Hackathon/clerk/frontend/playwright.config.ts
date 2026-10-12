import { defineConfig, devices } from "@playwright/test";

const PORT = 8123;
const python = process.env.CLERK_PYTHON ?? "../backend/.venv/bin/python";
// Set CLERK_E2E_URL to test an already-running Clerk (e.g. the Docker image or your Nebius deployment).
const external = process.env.CLERK_E2E_URL;

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: true,
  reporter: [["list"]],
  use: { baseURL: external ?? `http://localhost:${PORT}`, trace: "retain-on-failure", acceptDownloads: true },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: external
    ? undefined
    : {
    command: `${python} -m uvicorn clerk.api:create_app --factory --port ${PORT}`,
    cwd: "../backend",
    url: `http://localhost:${PORT}/api/health`,
    reuseExistingServer: false,
    env: { CLERK_MODE: "offline", CLERK_DB_PATH: ":memory:", CLERK_STATIC_DIR: "../frontend/dist" },
  },
});
