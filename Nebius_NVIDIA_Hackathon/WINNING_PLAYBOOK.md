# How I Would Beat You: The Rival's Playbook
### Nebius x NVIDIA Global AI Hackathon 2026

> Written from the point of view of the competitor most likely to take first place from you.
> Copy all of it, then do it better.

---

## 0. The facts (verified 8 Oct 2026, so re-check the Devpost rules page before you submit)

| Item | Detail |
|---|---|
| Deadline | **Oct 30, 2026, 10:00 AM PDT**. That is 22 days away. |
| Judging | Dec 1–15, 2026. **Your demo must stay live, working and free for the whole judging window.** |
| Winners | About Jan 11, 2027 |
| Hard requirement | Must make a **runtime call** to **Nebius Token Factory** inference, or deploy on Nebius Serverless Jobs, Serverless Endpoints or DevPods. Must use **at least one NVIDIA open-source model** (for example Nemotron). |
| Tracks | 1. Coding & Agentic Engineering · 2. Best Apps & Agents · 3. Personal AI · 4. Physical AI |
| Prizes (reported) | Grand $20k · 2nd $10k · 3rd $6k · **Best Use of Tavily $3k** · Jetson Orin Nano for each track winner · $500 City Winner awards |
| Submission | Hosted demo · public repo with a visible OSS LICENSE · README with setup steps · **public YouTube video under 3 min** · English · technology feedback form |
| Competition | About 18,000 registrants. Typically only 5–10% submit anything, and fewer submit something polished. |

### How judging works (this matters most)
- **Stage 1, pass/fail:** Does the project *genuinely* fit the track? Does it make *meaningful* use of Nebius and NVIDIA? A thin wrapper or rebrand gets cut here. Thousands of entries will die at this stage.
- **Stage 2, four criteria with equal weight:**
  1. **Technological implementation**
  2. **Design**: a complete, coherent product
  3. **Potential impact**: a real problem for a real audience
  4. **Quality of idea**: creative, non-obvious, shows domain understanding

---

## 1. The win: what first place looks like

> *January 11, 2027. The grand prize goes to **"Clerk: the personal AI that fights your paperwork for you."** The same project also wins the Personal AI track and Best Use of Tavily, for $23,000 and a Jetson.*
>
> The judge opened the demo link, logged in with the test account and dropped in a photo of a hospital bill. Within 20 seconds Clerk:
> 1. read the bill (an NVIDIA vision/doc model),
> 2. remembered the user's insurance plan from a letter uploaded three weeks earlier (persistent memory),
> 3. found a duplicate charge and an out-of-network charge that should not apply,
> 4. used Tavily to pull the **current** regulation that applies, with a citation link,
> 5. drafted a dispute letter, set a 30-day follow-up reminder, and asked permission before doing anything external,
> 6. showed a live trace panel: *"Planner: Nemotron 3 Super on Nebius Token Factory · Executor: Nemotron Nano · 3 tool calls · 0.004 USD."*
>
> The judge had a bill like that last year. **That is why it won.**

The idea was not the only reason it won. Six things decided it:

## 2. The six causes of the win

### Cause 1. A problem the judge has *personally* suffered
Judges score "impact" on gut feeling. Paperwork, bills, leases, visas, insurance denials and parking fines hurt everyone, and every judge has a story. Abstract infra tools score as "nice"; a pain the judge feels scores as "I need this."
**Test:** Can you explain the problem to your grandmother in one sentence? If not, pick another problem.

### Cause 2. NVIDIA and Nebius are load-bearing, not decorative
Stage 1 kills wrappers. The winner used the sponsor stack *in ways that matter* and **made that use visible**:
- **Nemotron 3 Super** (120B hybrid MoE, built for agents) as the planner/reasoner.
- **Nemotron Nano** as the fast, cheap executor. Routing between the two is a real engineering decision you can show with numbers (cost and latency per task).
- **An NVIDIA embedding and reranker model** for memory retrieval, if Token Factory lists one. Check the catalog.
- **An NVIDIA vision or document model** for reading uploads. Again, check what the catalog actually serves.
- **Nebius Serverless Endpoint** hosting the backend, so the "runs on Nebius" claim is literal.
- **NVIDIA's own agent tooling named in the track brief** (NemoClaw / OpenShell / Hermes Agent for Personal AI). Judges come from these companies. Using their newest toys tells them you did the homework.
- A **trace panel in the UI** showing which model ran where. The judge must never have to wonder whether it really uses Nebius.

### Cause 3. Stacking prizes
Every entry can be judged for the **grand prize + its track prize + Best Use of Tavily**. Most people compete for one prize. The winner competed for three with one project.
Tavily is meaningful when the agent **needs fresh, real-world facts it could not know**, such as current laws, prices, deadlines or policies. Give every claim a citation link.

### Cause 4. A 3-minute video that sells the project
Many judges watch the video *before* touching the demo, and some only watch the video. The winning video:
- **0:00–0:15:** The hook. A real person's pain ("Maria was charged $1,840 twice").
- **0:15–1:45:** The live demo with the wow moment in the first minute. No slides yet.
- **1:45–2:30:** The architecture diagram: Nemotron Super + Nano, Token Factory, Serverless, Tavily, memory. Then one numbers slide ("routing cut cost 78% with the same accuracy on 50 test documents").
- **2:30–2:55:** Impact and what comes next. End on the product name.
- Clear voice, captions, no dead air, 1080p screen recording, cursor zoomed. Re-record until it is clean.

### Cause 5. Evidence, not adjectives
Most entries say their project is "accurate" or "fast." The winner **proved it**: a small eval set (30–50 synthetic documents with known errors), plus a table in the README of detection rate, false positives, latency and cost per document for Super-only, Nano-only and routed. Engineers on the judging panel trust numbers.

### Cause 6. The demo did not break
Many good entries lose because:
- the endpoint died in December when credits ran out,
- login needed an email confirmation the judge could not get,
- the first request cold-started for 90 seconds,
- the repo had no LICENSE file, or the video was "unlisted/private."

The winner had **a judge test account with pre-loaded synthetic data**, a **warm endpoint**, **budget reserved through Dec 15**, a **"Try sample document" button** so the judge never needs their own file, and checked all of it **in a logged-out incognito window**.

---

## 3. The idea, chosen on purpose

### My pick: **Personal AI track → "Clerk"**
A private, always-on assistant that **reads, remembers and acts on your life admin**: bills, insurance, leases, letters, subscriptions and deadlines.

| Track requirement | How Clerk meets it |
|---|---|
| Persistent memory | Remembers your plan, landlord, account numbers and past disputes, with a vector memory you can view and edit |
| Reusable skills | "Dispute a charge," "Find the deadline," "Cancel a subscription," "Explain this letter simply." Each skill is a reusable, versioned module |
| User-controlled tools | Every external action (email, calendar, letter) needs explicit approval, and there is a permissions panel |
| Private | User data stays in the user's store, there is a visible "forget everything" button, and no training on user data |

**Why this beats the crowd:** most Personal AI entries will be "a chatbot with memory." Clerk has **a specific job, a measurable outcome (money recovered, deadlines met), and one demo moment anyone understands.**

### Backup picks, if you have a better domain story
- **Physical AI** (fewest competitors, because hardware scares people). Simulation counts: the rules accept "key application modules in action" when there is no hardware. A Nemotron agent driving an Isaac Sim robot, or a Jetson/webcam safety monitor. High ceiling, but the hardest path in 3 weeks.
- **Coding & Agentic Engineering:** a coding agent that **proves** its fix by running tests in Token Factory Sandboxes before proposing it. Sandbox access may be gated or in beta, so **request access today**.
- **Best Apps & Agents:** any narrow, painful, vertical workflow (farmers, small clinics, visa applicants). This track is crowded, so you need a *very* specific audience.

**Rule:** pick the idea where you have the **most unfair domain insight**. Criterion 4 rewards domain understanding. A problem you have lived through beats a clever problem you have only read about.

---

## 4. The 22-day roadmap

**Ship a full version by Day 19 (Oct 27). Days 20–21 are buffer. Submit by Oct 29. Never submit on the last day.**

### Week 1: Foundations (Oct 8 – Oct 14)
| Day | Do this |
|---|---|
| **D1 (Oct 8)** | Register on Devpost. Claim the **$25 Token Factory credits** and join the **Nebius Builders Program** (more credits, Tavily credits, office hours). Get a Tavily API key. Request **Sandbox access** if relevant. Read the full rules page twice. |
| **D2** | Lock the idea. Write the **one-sentence pitch** and the **wow moment**. Write the video script *now*: the script is your spec. |
| **D3** | Hello world: Python/FastAPI backend calls **Nemotron 3 Super on Token Factory** (OpenAI-compatible API). Deploy that to a **Nebius Serverless Endpoint** today, not in week 3. |
| **D4** | Document ingestion: upload → vision/doc model → structured JSON (amounts, dates, parties, account numbers). |
| **D5** | Memory: embeddings + vector store (pgvector, Chroma or LanceDB) + a "what do you know about me" view. |
| **D6** | Agent loop: the planner (Super) picks skills; the executor (Nano) runs extraction and drafting. Add Tavily as a tool with citations. |
| **D7** | **End-to-end ugly demo works** on one sample bill. Record a rough screen capture. If this is not working, cut scope now. |

### Week 2: Make it good (Oct 15 – Oct 21)
| Day | Do this |
|---|---|
| **D8** | Build 3–4 reusable **skills** as clean modules. Add a human-approval gate before external actions. |
| **D9–10** | Frontend (React + Tailwind or Next.js): an upload area, a timeline of documents, a memory panel, a **trace panel showing models and cost**, and a "Try sample" button. |
| **D11** | Build the **eval set**: 30–50 synthetic documents with planted errors. Measure Super-only vs Nano-only vs routed. |
| **D12** | Use the eval results to tune prompts and routing. Put the numbers table in the README. |
| **D13** | Guardrails: refuse to give legal advice, cite every claim, add a confidence score, and add "I'm not sure, here's what to ask." |
| **D14** | Feature freeze. **Show it to 3 real people** who match the target user, watch them use it without helping, and note where they get confused. |

### Week 3: Polish and ship (Oct 22 – Oct 29)
| Day | Do this |
|---|---|
| **D15–16** | Fix everything those 3 people hit. Fix the empty, loading and error states. Pre-warm the endpoint. |
| **D17** | README: problem, demo GIF, architecture diagram, **"How we use Nebius & NVIDIA" section**, eval table, setup steps, the judge test credentials, and an **MIT/Apache-2.0 LICENSE**. |
| **D18** | Record the video, 5+ takes. Add captions. Upload to YouTube as **Public**. |
| **D19 (Oct 27)** | Fill in the Devpost form completely: every field, screenshots, the tech feedback section (write real, thoughtful feedback, because sponsors read it). |
| **D20 (Oct 28)** | **Logged-out incognito test** of everything: the demo URL, the test login, the video link, the repo, the LICENSE. Have a friend do it too. |
| **D21 (Oct 29)** | **SUBMIT.** Keep a day in reserve. |
| **Oct 30 → Dec 15** | Do not touch production. Watch credits and uptime weekly. Set an uptime monitor (UptimeRobot is free). |

---

## 5. Architecture blueprint

```
 User (web app)
   │  upload / chat
   ▼
 FastAPI backend ── deployed on Nebius Serverless Endpoint
   │
   ├─► Planner: Nemotron 3 Super (Token Factory)   decides which skill(s) to run
   ├─► Executor: Nemotron Nano (Token Factory)     extraction, drafting, cheap steps
   ├─► Doc/vision model (Token Factory catalog)    reads images and PDFs
   ├─► Embeddings + rerank (NVIDIA, if available)  memory retrieval
   ├─► Tavily                                      fresh facts and regulations, with citations
   ├─► Memory store (Postgres + pgvector)          user-editable, deletable
   └─► Action tools (calendar/email drafts)        behind a human-approval gate
   │
   └─► Trace log → UI panel: model, latency, tokens, cost per step
```

Keep it **boring and reliable**: Python, FastAPI, Postgres, React. Novelty belongs in the *idea and agent design*, not in an exotic stack that breaks at 2 AM.

---

## 6. What kills other teams (avoid all of it)

1. **Building a generic chatbot.** It fails Stage 1 or scores low on idea quality.
2. **Scope creep.** Ten half-features lose to three polished ones. Judges remember **one** moment.
3. **No deployed demo until the last day.** Deploy on Day 3.
4. **Video made in the last 3 hours.** It shows, so script it on Day 2.
5. **Credits run out in December** and the demo is dead when judges arrive.
6. **Repo hygiene:** missing LICENSE, secrets committed (use `.env` + `.gitignore`), and no setup steps.
7. **Hiding the sponsor tech.** If the judge cannot *see* Nemotron + Nebius in 10 seconds, assume they will not find it.
8. **Overclaiming.** "Replaces lawyers" makes judges skeptical. "Finds billing errors and drafts the dispute, and you decide" earns trust.
9. **Building alone when a teammate would help.** A designer or video editor is worth more than a second backend dev.

---

## 7. Daily scoreboard (score yourself every night)

| Criterion | Question | 1–5 |
|---|---|---|
| Tech implementation | Does it really work, with numbers that prove it? | |
| Design | Could a stranger use it without me explaining anything? | |
| Impact | Would a real person pay for or beg for this? | |
| Idea quality | Has the judge seen this 50 times already? | |
| Sponsor fit | Is Nemotron + Nebius visible in 10 seconds? | |
| Reliability | Will it work, untouched, on Dec 15? | |

Anything below 4 is your next task.

---

## 8. Day-1 checklist (do these today)

- [ ] Register on Devpost and join the hackathon
- [ ] Claim the $25 Token Factory credits plus the Builders Program credits
- [ ] Get a Tavily API key
- [ ] Make one successful API call to Nemotron on Token Factory
- [ ] Write the one-sentence pitch and the wow moment
- [ ] Draft the 3-minute video script
- [ ] Create the public repo with a LICENSE, `.gitignore` and README skeleton

---

### Sources
- [Devpost: Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) · [Rules](https://nebiusglobalaihackathon.devpost.com/rules) · [Resources](https://nebiusglobalaihackathon.devpost.com/resources)
- [Nemotron 3 Super on Nebius Token Factory (Nebius blog)](https://nebius.com/blog/posts/nemotron3-super-now-available)
- [Nebius Token Factory launch](https://nebius.com/newsroom/nebius-launches-nebius-token-factory-to-deliver-production-ai-inference-at-scale)
- [Participant notes summarizing rules and judging stages](https://github.com/RenatoMignone/nvidia-x-nebius-personal-ai/blob/main/docs/hackathon.md)
- [Prize summary (tierones.io)](https://tierones.io/opportunities/nebius-nvidia-ai-2026) · [startupgrantsindia](https://www.startupgrantsindia.com/competitions/nebius-x-nvidia-global-ai-hackathon) · [Internshala listing](https://internshala.com/competitions/nebius-x-nvidia-global-ai-hackathon-2026/)
- [Token Factory Sandbox demos](https://github.com/kreuzhofer/nebius-token-factory-sandbox-demos)

*Prize amounts and some details come from secondary sources. The official Devpost rules page decides.*
