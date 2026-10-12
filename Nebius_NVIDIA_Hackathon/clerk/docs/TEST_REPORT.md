# Test report

Build and verification run of 12 October 2026, in a Linux build environment that **could not reach Nebius or Tavily** and had **no API keys**. Everything below was therefore run in **offline mode**, where a deterministic simulator stands in for Nemotron. See "Not tested" at the end.

## Results

| Check | Command | Result |
|---|---|---|
| Clean install from `pyproject.toml` in a fresh virtualenv, then the full suite | `pip install -e 'backend[dev]'` · `pytest` | ✅ 83 passed |
| Backend unit + API tests | `make test` | ✅ **83 passed**, **95% line coverage** (clerk + evals) |
| Lint / format / types | `make lint` | ✅ ruff clean · ruff format clean · mypy `--strict` clean (30 files) · `tsc --strict` clean |
| Offline eval, 50 synthetic bills | `make eval` | ✅ 0 failures · 80 planted errors, 43 decoys · after the Skeptic: precision 100%, recall 100%, $0 wrongly accused (see the caveat below) |
| Browser end-to-end (Chromium, Playwright) against the dev server | `make e2e` | ✅ **13 passed** |
| Same suite, 3 repeats, to catch flaky tests | `npx playwright test --repeat-each=3` | ✅ 39/39 passed |
| Docker image builds and runs | `docker build` · `docker run` | ✅ built (246 MB), runs as non-root user `clerk`, HEALTHCHECK reports `healthy` |
| Browser end-to-end against the **Docker container** | `CLERK_E2E_URL=http://localhost:8200 npx playwright test` | ✅ **13 passed** |
| Manual visual check (desktop 1400px, phone 390px), browser console | screenshots | ✅ no console errors, no horizontal scroll on phone |
| No secrets, and no mention of AI assistants, in the repository | grep scan | ✅ clean |

### What the tests cover
- **Parsing:** statements and agreements, thousands separators, missing fields, documents that are neither.
- **Auditors:** duplicates, line/subtotal/tax/total arithmetic, contract-rate overcharges, late-fee caps, services not in the agreement, behaviour with no agreement on file.
- **Verifier:** rejects lines that don't exist, caps model over-claims, recomputes every type of finding, dismisses fees within the cap.
- **Model output handling** (scripted "live" responses): `<think>` blocks, markdown fences, trailing commas, prose around JSON, a single corrective retry then a clear error, `$1,200.50`-style strings, a 5 written as a tax rate of 5%, invalid Skeptic verdicts treated as *dismissed*, a Skeptic model failure never turning into an accusation, the Advocate letter guardrail falling back when the amounts don't match.
- **Token Factory client contract:** exact URL, auth header and JSON body, retries on 429/5xx/timeouts, no retry on 4xx, unexpected response shapes, embeddings order, vision role routing, missing keys.
- **Tavily client contract:** request body and header, result trimming, error handling, a search failure becoming a visible notice.
- **Budgets:** daily model-call cap in live mode, per-workspace audits per hour (HTTP 429), cost accounting when prices are configured.
- **API:** every endpoint; upload limits (size, empty, too long, wrong type, broken PDF, scanned PDF, photo offline); PDF bills end to end; approval required before download; permissions; editing the Vault changing the next verdict; export, delete and wipe; SSE event order; audit failures reported instead of hanging, with internal errors hidden from users.
- **Security:** workspaces isolated between browsers; invalid cookies replaced; path traversal through the static file server and the samples endpoint blocked; uploaded filenames sanitised.
- **Browser journeys:** demo → trial for each of Maria's three bills; verdicts restored after navigating away; approve → download → file contents; reject; edit Vault → re-audit; ask the Vault; upload auto-runs the Tribunal; a bill without an agreement; bad uploads; permissions; trace panel; forget everything; two browsers see separate data; phone layout.

### Bugs found by testing and fixed before delivery
1. **Permission toggles felt broken**: the checkbox didn't move until the server responded. Caught by Playwright, fixed with an optimistic update that rolls back on error.
2. **The offline Reader can't read photos**: an image upload in offline mode returned HTTP 503. It now returns a clear HTTP 400 explaining that photos need live mode.
3. **Dispute letters quoted the Skeptic's internal reasoning to the company.** Removed from the letter (it still shows in the UI).
4. Build hygiene: `egg-info` build output excluded from git and Docker.

## ⚠️ Caveat on the offline eval numbers
The offline simulator's Skeptic was written with knowledge of the decoy types, so **100% precision offline is expected by construction**. The offline eval proves the pipeline, scoring and reporting work. It says **nothing** about how Nemotron performs. Only `make eval-live` produces numbers you may publish.

## Not tested here (needs your keys or your accounts)
| Not tested | Why | How to test it |
|---|---|---|
| Real Nemotron calls on Token Factory | network blocked, no key | `make models`, then `make live-smoke` |
| Exact Token Factory model IDs | unknown until listed with your key | `make models` |
| Real model quality (precision/recall/$) | needs live models | `make eval-live`, `make eval-ablation` |
| Reading **photos** of bills (vision model) | needs live mode + a vision-capable Nemotron model | upload a phone photo in live mode |
| Tavily search results and citation quality | network blocked, no key | run an audit in live mode with `TAVILY_API_KEY` set |
| Nemotron's Advocate letters | live only | read 3–4 letters it writes |
| Deployment on a Nebius Serverless Endpoint | no Nebius account access | `docs/DEPLOY_NEBIUS.md`, then the e2e suite against the URL |
| The **unmodified** Dockerfile | the build sandbox needed its own TLS certificate added to reach PyPI and npm; the tested build used a copy of the Dockerfile with only that certificate added | `make docker-build` on your machine |
| GitHub Actions CI workflow | it only runs once the project is in its own GitHub repository | push to the new repo and watch the Actions tab |
| Real-world bill layouts (yours) | only synthetic data was used | upload 2–3 of your own (redacted) bills in live mode |
