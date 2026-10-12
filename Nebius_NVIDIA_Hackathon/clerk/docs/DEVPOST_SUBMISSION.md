# Devpost submission draft

Fill every `<...>`, then paste each section into the matching Devpost field. Re-read the official rules page first, because field names may differ.

**Project name:** Clerk
**Tagline (≤ 60 chars):** The personal AI that audits your bills and fights back
**Track:** Personal AI
**Also eligible for:** Best Use of Tavily

**Links**
- Demo: `https://<endpoint>` (no login: click "Try it with Maria's paperwork")
- Video: `https://youtu.be/<id>` (public, under 3 minutes)
- Code: `https://github.com/<you>/clerk` (Apache-2.0)

---

## Inspiration
Every month, machines send us bills, and machines make mistakes: double charges, prices above the rate we signed for, fees our contracts don't allow. Checking means finding the contract, reading every line, knowing your rights and writing a formal dispute before a deadline. Almost nobody does that, so the money stays lost. We wanted a personal AI that does the tedious part, and that is careful enough to trust.

## What it does
Clerk is a private, always-on assistant for life admin.
- **Remembers:** upload an agreement once, and Clerk's Vault stores its terms as facts you can see, edit, export and delete.
- **Audits:** upload a bill (text, PDF or photo). Clerk matches it to the right agreement and checks every line.
- **Puts every accusation on trial:** rule-based auditors and a Nemotron AI Auditor raise possible problems, plain Python recomputes every number, and a Nemotron 3 Super "Skeptic" cross-examines each one, looking for innocent explanations. False alarms are thrown out before the user sees them.
- **Cites:** Tavily finds current, citable guidance for each upheld finding.
- **Acts, on a leash:** it drafts a dispute letter, a phone script with rebuttals and a calendar reminder for the dispute deadline. All of them wait in an approval queue.
- **Shows its work:** a trace panel lists every model call with its model, tokens, latency and cost.

## How we built it
- **NVIDIA Nemotron on Nebius Token Factory** for every model call, through the OpenAI-compatible API: Nemotron Nano for reading, auditing and Q&A; Nemotron 3 Super for cross-examination and letter writing; <vision model> for photos of bills.
- **Routing:** cheap steps on Nano, judgement on Super. <live ablation result: cost/latency vs Super-only>.
- **"The AI proposes, the code verifies":** every dollar amount is recomputed by deterministic code from the bill's own lines. Model findings that cite non-existent lines or over-claim are rejected or corrected. The Advocate's letter is rejected if its amounts don't match the verified total.
- **Tavily** for live research with citations.
- **FastAPI** + Server-Sent Events, so the browser watches the Tribunal live. **SQLite** for the Vault, with embeddings for semantic recall. **React + Tailwind** front end.
- Deployed as **one container on a Nebius Serverless Endpoint**.
- **Evaluation lab:** a generator for synthetic bills with planted errors *and decoys* (legitimate charges that look wrong). <live numbers: precision/recall, $ wrongly accused, before vs after the Skeptic>.
- **Testing:** 83 backend tests (95% coverage), 13 Playwright browser tests, strict type-checking, CI.

## Challenges we ran into
- **Reasoning models wrap JSON in thinking.** We built a tolerant extractor (strips `<think>` blocks and code fences, repairs trailing commas) plus one corrective retry.
- **High recall creates false accusations.** Our first auditors flagged installments and renamed services. Instead of loosening them, we added the Skeptic, so recall stays high and precision comes from cross-examination.
- **Never trusting the model's arithmetic.** Moving every number into the Verifier took a redesign, but it means users never see an amount the code didn't confirm.
- <add one real thing that happened while going live>

## Accomplishments that we're proud of
- On the demo set, the Skeptic throws out every decoy while keeping every real error. <replace with live numbers>
- A personal AI that says "this bill checks out" when it does. No accusation without evidence.
- Every action is user-approved, every fact is user-editable, and everything can be erased with one button.

## What we learned
- Two cheap models arguing beats one expensive model guessing.
- Deterministic code and LLMs are better together: the model finds and explains, the code proves.
- <your own lesson>

## What's next for Clerk
- Email-inbox and bank-statement connectors, so bills arrive automatically
- Tracking dispute outcomes ("Northwind credited $39.99")
- More document types: medical EOBs, insurance denials, leases
- Running fully on-device on an NVIDIA Jetson for maximum privacy

## Built with
nvidia-nemotron · nebius-token-factory · nebius-serverless · tavily · python · fastapi · sqlite · react · typescript · tailwindcss · vite · playwright · docker

---

## Technology feedback (Devpost field: write this in your own words, sponsors read it)
Write 3–5 honest, specific points from your own experience. Prompts:
- How easy was it to get a Token Factory key and make the first Nemotron call? Anything confusing?
- Did the OpenAI-compatible API work as a drop-in? Any differences (response_format, `<think>` output, usage fields)?
- Nemotron Super vs Nano: where did each shine or struggle?
- Serverless Endpoint deployment: what was smooth, what was missing (e.g. in-place updates, secrets, logs)?
- Tavily: result quality for consumer-rights queries?
