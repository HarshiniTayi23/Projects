# Project Proposal: **Clerk**
### *The personal AI that audits your bills and fights back.*
Nebius x NVIDIA Global AI Hackathon 2026 · Primary track: **Personal AI** · Also targets: **Grand Prize**, **Best Use of Tavily**

---

## 1. One-line pitch
**Drop in a bill. Clerk finds the money you're owed, proves it, and writes the letter to get it back.**

## 2. The problem
People sign plans, leases and contracts, then get billed by systems that make mistakes, almost always in the company's favor: duplicate charges, wrong rates, fees the contract doesn't allow, and math that doesn't add up. Checking a bill means:
1. finding the original contract or plan,
2. remembering what it said,
3. checking every line,
4. knowing which rule protects you, and
5. writing a formal dispute before the deadline.

Almost nobody does all five steps, so the money stays lost.

**Audience:** anyone who pays bills. That is especially true for students, immigrants, elderly people and caregivers, who face the most paperwork with the least support.

## 3. What makes it "crazy" (and still real)

### 3.1 The Tribunal: agents that argue before they accuse
Most AI apps show the first thing the model says. Clerk puts every claim through a tribunal:

| Stage | Agent | Model (Nebius Token Factory) | Job |
|---|---|---|---|
| 1 | **Reader** | Nemotron Nano Omni (multimodal) | Turns a photo, PDF or text bill into structured line items |
| 2 | **Auditors** (5 specialists) | Nemotron Nano (fast/cheap) | Duplicate hunter · Rate checker (vs. remembered contract) · Fee lawyer · Math checker · Subscription-trap finder |
| 3 | **Verifier** | Plain Python, not AI | Recomputes every number. **The AI proposes and the code verifies**, so the math is never made up. |
| 4 | **Skeptic** | Nemotron 3 Super (120B reasoning) | Cross-examines each finding: "Show me the evidence line. Could this be legitimate?" Findings that don't survive are thrown out. |
| 5 | **Researcher** | Tavily search | Pulls the **current** rule or policy behind each surviving finding, with a source link |
| 6 | **Advocate** | Nemotron 3 Super | Drafts the dispute letter, a phone-call script with rebuttals, and the deadline |

The UI streams this live, like watching a courtroom: claims appear, get challenged, and are either upheld (green) or dismissed (struck through). It is the demo's wow moment, and it is honest engineering, because it measurably cuts false accusations.

### 3.2 The Vault: memory that connects documents
Every document becomes **facts**: "Rent: $1,450/mo · Lease clause 7: late fee max $50 · Internet promo rate $39.99 until 2027-03." When a new bill arrives, the auditors check it against everything Clerk remembers.
**The killer moment:** the March internet bill charges $69.99, and Clerk says: *"Your contract from January locks $39.99 until March 2027. You've been overcharged $30/month for 3 months: $90."*
The user can see, edit, export or delete every fact, and a single button forgets everything.

### 3.3 Hands, with a leash
Clerk prepares actions (a dispute letter, a calendar reminder as an `.ics` file, a call script), but **every action waits in an approval queue**. A permissions panel shows which tools Clerk may use. This matches the track brief's requirement for "user-controlled tools."

### 3.4 Glass box
A trace panel shows every step: which agent ran, which Nemotron model, on Nebius, tokens, latency and cost in dollars. A judge sees the sponsor stack in under 10 seconds.

### 3.5 The scoreboard
"$412.60 found across 6 documents · 9 findings upheld · 4 dismissed by the Skeptic." It is personal and visible, and it shows impact directly.

### 3.6 Receipts: a built-in evaluation lab
A **synthetic bill generator** creates 50+ realistic documents (utilities, internet, medical, rent, phone, subscriptions) with **planted errors whose answers are known**. An eval runner measures:
- precision and recall of findings,
- dollars recovered vs. dollars actually wrong,
- cost and latency per document,
- **ablations**: Skeptic on vs. off, and routed (Nano+Super) vs. Super-only.

The results table is generated straight into the README. That is the "evidence, not adjectives" point from the playbook.

## 4. Requirement fit (Stage 1 pass/fail)
| Requirement | How Clerk meets it |
|---|---|
| Runtime call to Nebius Token Factory | Every agent calls `api.tokenfactory.nebius.com/v1` (OpenAI-compatible) |
| ≥1 NVIDIA open-source model | Nemotron 3 Super, Nemotron Nano, Nemotron Nano Omni |
| Runs on Nebius | One Docker container deployed to a **Nebius Serverless Endpoint** |
| Personal AI: persistent memory | The Vault (SQLite + embeddings) |
| Personal AI: reusable skills | Each auditor and the Advocate is a registered, reusable skill module |
| Personal AI: user-controlled tools | Approval queue + permissions panel |
| Tavily | The Researcher agent, with citations on every upheld finding |
| Open source | Apache-2.0 LICENSE, public repo, README with setup steps |

## 5. Architecture
```
 Browser (React + Vite + Tailwind, built to static files)
    │  REST + Server-Sent Events (live tribunal stream)
    ▼
 FastAPI app  ── one Docker image ── Nebius Serverless Endpoint
    ├── /api/documents     upload → Reader → facts → Vault
    ├── /api/audit/{id}    runs the Tribunal and streams events (SSE)
    ├── /api/vault         list / edit / delete / export facts
    ├── /api/actions       approval queue (approve / reject / download .ics/.txt)
    ├── /api/trace         per-step model, tokens, latency, cost
    └── /api/demo/reset    loads the "Maria" demo persona for judges
    │
    ├── llm/      Provider interface
    │     ├── NebiusProvider   (live: OpenAI-compatible client)
    │     └── ReplayProvider   (offline: deterministic, used in tests and the offline demo)
    ├── search/   TavilyClient + ReplaySearch
    ├── agents/   reader, auditors/*, verifier, skeptic, researcher, advocate
    ├── vault/    SQLite store + embeddings (Nebius embedding model, hashed fallback offline)
    └── evals/    synthetic generator + runner + report writer
```
**Deliberately boring stack:** Python 3.12, FastAPI, SQLite, React. A single container means a single deploy and nothing extra to break in December.

**Model IDs live in config, not code.** A `scripts/list_models.py` script asks Token Factory which Nemotron models it actually serves, so you plug in the exact names on day one.

## 6. How I will test it, fully
| Layer | What runs | Tool |
|---|---|---|
| Unit | Every auditor, the verifier math, the Vault, the router, cost accounting, the .ics generator, the synthetic generator | pytest |
| Agent logic | The full tribunal against replayed model responses: findings upheld/dismissed exactly as expected | pytest + ReplayProvider |
| API | Every endpoint, error cases (bad file, huge file, empty bill, unknown id), SSE stream order | pytest + httpx TestClient |
| Live-provider contract | NebiusProvider sends exactly the right OpenAI-format request, and retries/timeouts behave | pytest with a fake HTTP server |
| End-to-end UI | A real browser: load demo → upload bill → watch the tribunal → approve the letter → download it → delete memory | Playwright + Chromium |
| Evals | 50 synthetic docs through the full pipeline (offline mode), with a metrics report | eval runner |
| Container | `docker build` + run + health check + one full audit through the container | Docker |
| Quality | Lint + format + type checks | ruff, mypy, eslint/tsc |
| Security | No secrets in the repo, upload size/type limits, no path traversal, no shell calls | review + tests |

### ⚠️ The honest limit
This build environment **cannot reach Nebius or Tavily** (the network blocks them) and has **no API keys**. So:
- Everything above is tested with **replayed, deterministic model outputs**. That proves the code, the pipeline and the UI are correct.
- It does **not** prove how well the real Nemotron models perform. That takes one command on your machine with your keys (`make live-smoke`, then `make eval-live`), and it is item #1 on your action list.
- The offline eval numbers will be clearly labeled **"offline/replay: not model quality."** Real numbers go into the README only after you run the live eval.

## 7. What you get at the end
1. `clerk/`: the complete app (backend, frontend, Dockerfile, tests, evals, demo data)
2. A README written for judges: pitch, GIF placeholder, architecture, "How we use Nebius & NVIDIA," eval table, setup, judge credentials section, license
3. `VIDEO_SCRIPT.md`: a timed 3-minute script matching the demo flow
4. `DEVPOST_SUBMISSION.md`: every Devpost field pre-written, including the technology feedback section
5. `ACTION_ITEMS.md`: **your checklist**, every item you must do or look through, in order, with how to verify each one
6. A test report: what ran, what passed, and what could not be tested here and why

## 8. Build plan (my execution order)
1. Scaffold, config, provider interface, replay provider, and tests
2. Reader + Vault + synthetic document generator
3. The five auditors + Python verifier
4. Skeptic + Researcher + Advocate, then the full tribunal with SSE streaming
5. Approval queue, .ics/letter export, trace and cost accounting
6. Frontend: upload, live tribunal, Vault, actions, trace, scoreboard, demo persona
7. Eval lab + report generator
8. Docker image + container test
9. Playwright end-to-end tests
10. README, video script, Devpost draft, action list, and the final full test run

## 9. Decisions I need from you
1. **Go with Clerk?** (Or tell me a problem you've personally lived through, since domain knowledge scores points.)
2. **Where should the code live?** Devpost needs a *public* repo containing only this project. Options: (a) I build it in `Nebius_NVIDIA_Hackathon/clerk/` here, and you copy it into a new public repo later, or (b) you create a new empty public repo, and I add it to this session and push there.
