# Deploying Clerk on Nebius

Clerk is one container: FastAPI serves the API and the built React UI on port 8000.

> The `nebius` CLI commands below follow the Nebius Serverless quickstart. Flags change between CLI versions, so **check `nebius ai endpoint create --help` before running them.**

## 0. Before you start
- The Nebius CLI is installed and logged in (`nebius profile create`), and you know your project ID.
- `.env` is filled in and `make live-smoke` passes on your laptop.
- The image builds: `make docker-build`.

## 1. Push the image to a registry
Either use Nebius Container Registry (create a registry in the console, which gives you a path like `cr.<region>.nebius.cloud/<registry-id>`):
```bash
docker tag clerk:latest cr.<region>.nebius.cloud/<registry-id>/clerk:v1
docker push cr.<region>.nebius.cloud/<registry-id>/clerk:v1
```
or use any **public** registry, such as GitHub Container Registry. The image contains no secrets: keys are passed as environment variables at deploy time.

## 2. Create the endpoint
Judges must open the app in a browser, so the endpoint has **no token auth** (`--auth none`). Clerk protects your credits itself, with `CLERK_AUDITS_PER_HOUR` per visitor and `CLERK_DAILY_LLM_CALL_LIMIT` overall.

```bash
nebius ai endpoint create \
  --name clerk \
  --image cr.<region>.nebius.cloud/<registry-id>/clerk:v1 \
  --platform cpu-d3 \
  --preset 4vcpu-16gb \
  --container-port 8000/http \
  --auth none \
  --env CLERK_MODE=live \
  --env NEBIUS_API_KEY=<key> \
  --env CLERK_MODEL_SUPER=<id> \
  --env CLERK_MODEL_NANO=<id> \
  --env CLERK_MODEL_VISION=<id> \
  --env TAVILY_API_KEY=<key>
```
- Use the **smallest CPU preset** your region offers (`nebius compute platform list --parent-id <project-id>`). Clerk needs no GPU, because inference runs on Token Factory.
- If your organisation uses custom networking, add `--subnet-id`.
- Prefer secrets over plain `--env` for the keys if your CLI supports it (MysteryBox), because env values can be visible to anyone with access to the project.

## 3. Get the URL and test it
```bash
ENDPOINT_ID=$(nebius ai endpoint get-by-name --name clerk --format jsonpath='{.metadata.id}')
nebius ai endpoint get "$ENDPOINT_ID" --format json | jq -r '.status.public_endpoints[0]'
nebius ai endpoint logs "$ENDPOINT_ID" --follow
```
Then run the whole browser test suite against the deployment:
```bash
cd frontend && CLERK_E2E_URL=https://<your-endpoint> npx playwright test
```
The tests assert exact dollar amounts for Maria's bills. Offline, the simulator guarantees them. Live, they depend on Nemotron reaching the same verdicts. If one differs, treat it as a model-quality signal to look at (read the Skeptic's reason in the UI), not automatically a bug.

## 4. Keep it alive until December 15
- Endpoints bill from creation to deletion, including idle time. Check the hourly price of your preset against your credits **before** you rely on it for two months.
- There is no in-place update: to ship a new image, create a new endpoint, test it, switch the link in your Devpost and README, then delete the old one.
- SQLite lives inside the container, so a restart clears visitors' workspaces. That's fine for a demo: judges click "Try it with Maria's paperwork" again.
- Add a free uptime monitor (for example UptimeRobot) on `https://<endpoint>/api/health`.

## Fallback if the endpoint is too expensive
The hackathon rules accept "a runtime call to the Token Factory inference API" as qualifying Nebius use. You can host the same Docker image on any cheap container host and keep every model call on Token Factory. Say so in the Devpost write-up.
