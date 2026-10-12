# Your action list: Clerk → submission

Deadline: **Oct 30, 2026, 10:00 AM PDT**. Plan to **submit Oct 29**.
Work top to bottom. Each item says how to check it worked. 🔴 = blocks the submission.

---

## A. Accounts and keys (today, about 30 min)
- [ ] 🔴 **Register on Devpost** and join the Nebius x NVIDIA Global AI Hackathon. Read the rules page in full.
- [ ] 🔴 **Get a Nebius Token Factory API key.** Claim the **$25 credits** (Devpost Resources page) and join the **Nebius Builders Program** for more credits plus Tavily credits.
- [ ] **Get a Tavily API key** (tavily.com). Needed for the $3k Best Use of Tavily prize.
- [ ] **Delete the old branch** `claude/nebius-nvidia-hackathon-strategy-ffbmzk` on GitHub (Branches page → trash icon). The sandbox couldn't delete it.

## B. Give Clerk its own public repo (about 20 min)
Devpost needs a public repo containing only this project.
- [ ] 🔴 Create a new **public** GitHub repo, e.g. `clerk`.
- [ ] Copy everything inside `Nebius_NVIDIA_Hackathon/clerk/` to the **root** of that repo, then push.
  - ✔ Check: GitHub shows the **Apache-2.0** license badge on the repo page.
  - ✔ Check: the **Actions** tab runs CI and it goes **green**. This workflow has never run, so if it fails, read the log; it's usually a path or version issue.
- [ ] Make sure `.env` is **not** in the repo (it's in `.gitignore`; double-check after your first push).

## C. Go live: the part I could NOT test (most important)
The build sandbox couldn't reach Nebius or Tavily, so **nothing has run against the real models yet.**
- [ ] 🔴 `make setup`, then `cp .env.example .env` and paste in `NEBIUS_API_KEY`.
- [ ] 🔴 `make models` lists the Nemotron models your key can use. Fill in `.env`:
  - `CLERK_MODEL_SUPER` = Nemotron 3 Super
  - `CLERK_MODEL_NANO` = a Nemotron Nano model
  - `CLERK_MODEL_VISION` = a Nemotron model that **accepts images** (check the model card)
  - `CLERK_MODEL_EMBED` = an NVIDIA embedding model, if one is listed (optional)
- [ ] 🔴 `make live-smoke` must print PASS on every line.
  - ✔ Check: it upholds the Northwind **rate ($30.00)** and **add-on ($9.99)** findings.
  - If the Reader fails: the model may be wrapping its JSON oddly. Look at the raw output, then adjust `backend/clerk/agents/prompts.py` (`READER`).
- [ ] Add prices from the Token Factory pricing page to `.env` (`CLERK_PRICE_*`) so the cost meter shows dollars.
- [ ] `make dev`, open http://localhost:5173, check the green **LIVE** badge, and run all three of Maria's bills.
  - ✔ Expected: Northwind **$39.99**, Brightside **$34.50**, Lakeview **"This bill checks out"**. If live differs, read the Skeptic's reason. It may be a prompt to tune or a genuinely different (defensible) judgement.
- [ ] **Upload a phone photo of a bill** in live mode. Photo reading has never been tested.
- [ ] Try **2–3 of your own real bills** (cover account numbers). Note anything Clerk gets wrong. Real layouts are the biggest unknown.
- [ ] 🔴 `make eval-live` (about 300 model calls; check your credits first) and `make eval-ablation`.
  - ✔ Paste `backend/evals/results/live.md` into the README **Results** table and the Devpost write-up.
  - If precision or recall is weak, read `live.json` → `per_case` to see which error types are missed, and tune the prompts. **Never publish the offline numbers as model quality.**

## D. Look through what I built (about 1 hour)
- [ ] Read `clerk/README.md` top to bottom. It's what judges read first.
- [ ] Read the agent prompts in `backend/clerk/agents/prompts.py`. Are the tone and rules what you want?
- [ ] Read a dispute letter and a call script (`backend/clerk/paperwork.py`). Are you comfortable with the wording? Clerk must never sound like legal advice.
- [ ] Check the Tavily queries in `backend/clerk/agents/researcher.py`. In live mode, are the sources relevant and trustworthy? Consider adding the user's country/state to the queries.
- [ ] Check the default deadline: with no agreement on file, Clerk *suggests* 30 days (`paperwork.py`). Fine?
- [ ] Skim `docs/TEST_REPORT.md` so you can answer judges' questions about testing.

## E. Deploy on Nebius (about 1–2 hours)
- [ ] 🔴 Follow `clerk/docs/DEPLOY_NEBIUS.md`. Check the flags with `nebius ai endpoint create --help` first.
- [ ] 🔴 **Price check:** endpoints bill every hour until deleted. Make sure your credits cover the **smallest CPU preset** until **Dec 15**. If not, use the fallback in the guide (any cheap host, with model calls still on Token Factory).
- [ ] Set `CLERK_DAILY_LLM_CALL_LIMIT` so a day of judge traffic can't drain your credits (rough guide: about 8 model calls per audit).
- [ ] Run the browser tests against your deployment: `cd frontend && CLERK_E2E_URL=https://<url> npx playwright test`.
- [ ] Add a free uptime monitor on `https://<url>/api/health`.

## F. Submission assets (about half a day)
- [ ] 🔴 Record the video following `clerk/docs/VIDEO_SCRIPT.md`, **in live mode**. Keep it under 3:00 and upload to YouTube as **Public**.
- [ ] Add a demo GIF to the README, and replace every `TODO` / `<...>` (demo link, video link, results).
- [ ] 🔴 Fill in the Devpost form from `clerk/docs/DEVPOST_SUBMISSION.md`. Write the **Technology feedback** section in your own words, from your real experience.
- [ ] 🔴 Open every link in a **logged-out incognito window**: demo, video, repo, license.
- [ ] 🔴 **Submit on Oct 29.**

## G. After submitting (Oct 30 → Dec 15)
- [ ] Don't change production. If you must, deploy a new endpoint, test it, then switch the link.
- [ ] Check credits and the uptime monitor **weekly** until Dec 15.

---

## Known limitations (decide whether any matter to you)
| Limitation | Impact | Option |
|---|---|---|
| SQLite lives inside the container | a restart clears demo workspaces | fine for judging; judges re-click the demo |
| Per-browser audit limit can be dodged by clearing cookies | someone could burn more calls | the global daily cap is the real protection; keep it set |
| Upload size is checked after the file arrives | very large uploads waste bandwidth | acceptable for a demo |
| Scanned PDFs (no text layer) are refused | users must upload a photo instead | live mode handles photos |
| Exact-amount browser tests assume the simulator's verdicts | against live, a mismatch may be a model judgement, not a bug | read the Skeptic's reason |
| Fonts load from Google Fonts | a third-party request on page load | optional: self-host the fonts |
