# Step-by-Step Build Plan

Every step ends with a **test you can run** and a clear **done when**. Do not move on until the step's check passes. Commit after each step.

Target time: ~6 hours of build + ~2 hours of docs/video.

---

## Target architecture

```
┌─────────────────────┐
│  Streamlit UI       │  app.py
│  (chat + sidebar)   │
└─────────┬───────────┘
          │ user text / reply
┌─────────▼───────────┐       OpenAI-compatible HTTP       ┌──────────────────┐
│  Agent loop         │ ─────────────────────────────────▶ │ Llama 3.x 8B     │
│  agent/agent.py     │ ◀───────── text or tool_calls ──── │ (Groq/Together)  │
└─────────┬───────────┘                                    └──────────────────┘
          │ MCP client: list_tools / call_tool (stdio)
┌─────────▼───────────┐
│  MCP server         │  mcp_server/server.py
│  reservation tools  │
└─────────┬───────────┘
          │ SQL
┌─────────▼───────────┐
│  SQLite             │  data/goodfoods.db
└─────────────────────┘
```

## Target folder layout

```
goodfoods-reservation-agent/
├── app.py                     # Streamlit frontend
├── agent/
│   ├── llm_client.py          # thin wrapper over chat/completions
│   ├── agent.py               # the tool-calling loop
│   ├── mcp_bridge.py          # MCP client + schema conversion
│   └── prompts.py             # system prompt
├── mcp_server/
│   ├── server.py              # FastMCP server exposing tools
│   ├── tools.py               # pure business logic (testable without MCP)
│   └── db.py                  # SQLite access
├── data/
│   └── seed.py                # generates 50–100 restaurants
├── tests/
│   ├── test_db.py
│   ├── test_tools.py
│   ├── test_agent_loop.py     # with mocked LLM
│   └── evals/intent_eval.py   # real-LLM intent accuracy
├── docs/
│   ├── USE_CASE.md
│   ├── LEARNING_PATH.md
│   └── BUILD_PLAN.md
├── .env.example
├── requirements.txt
└── README.md
```

---

## Phase 0 — Setup (20 min)

1. Create a **private** GitHub repo `goodfoods-reservation-agent`.
2. `python -m venv .venv && source .venv/bin/activate`
3. `requirements.txt`: `openai`, `mcp`, `streamlit`, `pydantic`, `python-dotenv`, `faker`, `pytest`.
4. Get an API key from a provider that serves a Llama 8B model with tool calling (e.g. Groq `llama-3.1-8b-instant`). Put it in `.env`; commit only `.env.example`.

**Test:** `python -c "import openai, mcp, streamlit; print('ok')"`
**Done when:** prints `ok` and `.env` is in `.gitignore`.

---

## Phase 1 — LLM smoke test (20 min)

Write `agent/llm_client.py` with one function `chat(messages, tools=None)`.

**Test:** a script that sends "Say hello in 3 words" and prints the reply.
**Done when:** you get a reply and you know your latency and the model name.

---

## Phase 2 — Toy tool-calling loop (45 min) ⭐ most important

Before touching restaurants, prove the loop with one fake tool, e.g. `get_time(city)`.

Implement in `agent/agent.py`:
- send messages + tool schema
- if response has `tool_calls`, run each, append `{"role": "tool", "tool_call_id": ..., "content": json}`
- loop until plain text, max 5 iterations
- catch malformed JSON arguments → return an error message to the model instead of crashing

**Test:**
- "What time is it in Mumbai?" → tool is called.
- "Tell me a joke" → tool is **not** called.
**Done when:** both behave correctly 3 runs in a row. This proves intent is decided by the LLM, not by `if "time" in text`.

---

## Phase 3 — Database + seed data (60 min)

`mcp_server/db.py` – tables:

| table | key columns |
|---|---|
| `restaurants` | id, name, area, cuisine, price_tier (1–4), rating, open_time, close_time, total_seats, features (JSON: outdoor, vegan, bar, parking, private_dining, kid_friendly) |
| `reservations` | id (short code like `GF-7K2P`), restaurant_id, customer_name, phone, party_size, date, start_time, end_time, status (confirmed/cancelled), special_requests, created_at |

`data/seed.py` – generate 75 restaurants across ~10 areas and ~12 cuisines with seat counts 20–200. Use a fixed random seed so data is reproducible.

**Test (`tests/test_db.py`):**
- seed runs twice without duplicates
- count is between 50 and 100
- every restaurant has a valid open < close time

**Done when:** `pytest tests/test_db.py` is green.

---

## Phase 4 — Business logic tools, no LLM (75 min)

In `mcp_server/tools.py`, plain Python functions with pydantic-validated inputs:

| tool | purpose |
|---|---|
| `search_restaurants(cuisine?, area?, party_size?, price_tier?, features?)` | filtered list |
| `recommend_restaurants(preferences, party_size, date, time)` | scored top 3 with a "why" for each |
| `check_availability(restaurant_id, date, time, party_size)` | yes/no + alternative times if no |
| `make_reservation(restaurant_id, name, phone, date, time, party_size, special_requests?)` | returns confirmation code |
| `get_reservation(code)` | look up |
| `modify_reservation(code, new_date?, new_time?, new_party_size?)` | re-checks capacity |
| `cancel_reservation(code)` | soft cancel |
| `get_restaurant_details(restaurant_id)` | hours, features, address |

Rules to enforce in code (not in the prompt):
- no bookings in the past or > 60 days ahead
- time must be within opening hours and ≥ 90 min before close
- party size 1–20 (larger → suggest private dining contact)
- capacity check with overlapping 90-min slots, inside a DB transaction
- phone format validation
- return structured errors: `{"ok": false, "error": "FULLY_BOOKED", "alternatives": [...]}`

**Test (`tests/test_tools.py`):**
- book → availability decreases
- overbooking the last seats fails
- booking outside hours fails with a clear error
- cancel frees the seats
- modify to a full slot fails and keeps the original booking
- recommend never returns a restaurant too small for the party

**Done when:** all tool tests pass. You now have a correct system with zero AI.

---

## Phase 5 — MCP server (45 min)

`mcp_server/server.py` – wrap each function from `tools.py` with `@mcp.tool()` using FastMCP. Docstrings become tool descriptions, so write them carefully ("Use this when the user…").

**Test:**
- run the MCP Inspector against the server, list tools, call `search_restaurants` manually
- a tiny script that starts the server over stdio, calls `list_tools()` and prints 8 tools

**Done when:** all 8 tools are listed and callable from the Inspector.

---

## Phase 6 — Agent ↔ MCP bridge (60 min)

`agent/mcp_bridge.py`:
- start MCP server as a subprocess (stdio), `initialize`, `list_tools()`
- convert each MCP tool's `inputSchema` into the `{"type": "function", "function": {...}}` format
- `call(name, args)` → `call_tool()` → returns JSON string for the `tool` message

Plug it into the loop from Phase 2, replacing the toy tool.

**Test (`tests/test_agent_loop.py`, mocked LLM):**
- fake LLM returns a `search_restaurants` call → bridge is invoked → final answer returned
- fake LLM returns invalid JSON args → agent sends an error back and does not crash
- fake LLM loops forever on tools → agent stops at max iterations with a polite fallback

**Done when:** tests pass, and in a terminal REPL you can complete one full booking by chatting.

---

## Phase 7 — System prompt & conversation design (60 min)

`agent/prompts.py` – include:
- role and scope (GoodFoods only; politely decline unrelated requests)
- today's date, weekday and timezone injected at runtime
- slot-filling policy: required fields = restaurant, date, time, party size, name, phone; ask for missing ones, at most two at a time
- **always** call `check_availability` before `make_reservation`
- **always** read back a summary and wait for explicit confirmation before booking
- never invent restaurants, prices or availability – only use tool results
- on tool error: explain in plain language and offer the alternatives returned
- style: short, warm, max ~3 sentences, use lists for options

**Test (`tests/evals/intent_eval.py`, real LLM):** ~25 utterances with expected first tool, e.g.

| utterance | expected |
|---|---|
| "Any good Italian places in Indiranagar?" | `search_restaurants` |
| "Somewhere quiet for an anniversary, 2 people, Saturday 8pm" | `recommend_restaurants` |
| "Cancel booking GF-7K2P" | `cancel_reservation` |
| "Move my booking to 9pm" | asks for code (no tool) |
| "What's the capital of France?" | no tool, polite decline |
| "Book a table" | no tool, asks for details |

**Done when:** ≥ 85% correct. Record the score and every prompt change in a log — it goes in the README's prompt-engineering section.

---

## Phase 8 — Streamlit frontend (60 min)

`app.py`:
- chat window with history in `st.session_state`
- one agent/MCP session per browser session (cache it)
- sidebar: "Your bookings this session", a **"Show tool calls"** toggle that renders each tool call + result in an expander (great for the demo)
- quick-start buttons: "Recommend a place", "Check my booking", "Cancel a booking"
- friendly message if the API key is missing or the provider is down

**Test (manual script, write it down):**
1. Discover → recommend → book → receive code
2. Look up that code
3. Modify party size
4. Cancel
5. Try to overbook a small restaurant → get alternatives
6. Ask something off-topic
7. Give a past date → get corrected

**Done when:** all 7 journeys work in the browser without restarting.

---

## Phase 9 — Hardening (45 min)

- retries with backoff on 429/5xx from the LLM provider
- timeout on every LLM and tool call
- trim conversation history when it gets long (keep system prompt + last N turns)
- log every turn (user text, tool calls, latency) to a file
- handle the model replying with a tool call as plain text (known small-model quirk): detect and parse it, or re-ask

**Test:** run the full `pytest` suite + eval script; kill the network mid-chat and confirm the UI shows a friendly error.
**Done when:** no stack trace ever reaches the user.

---

## Phase 10 — Business document (2–3 h, can run in parallel)

`docs/USE_CASE.md`, following the provided template. Must cover:
- problem statement & non-obvious opportunities (no-show reduction via reminders, off-peak demand shaping, upsell of pre-orders/occasion packages, first-party guest data / CRM, multilingual & voice access, phone-load reduction for staff)
- users & stakeholders (diners, hosts, managers, marketing, ops, IT)
- success metrics with baselines and targets, plus a simple ROI calculation
- implementation timeline (pilot → 5 locations → full chain)
- vertical expansion (other chains, salons, clinics, hotels, coworking, events)
- 2–3 competitive advantages
- risks, assumptions, limitations, future enhancements

**Done when:** every template section is filled and each metric has a number.

---

## Phase 11 — README, demo video, submission (60–90 min)

README sections:
- one-paragraph overview + demo video link (top of page)
- architecture diagram
- setup (copy-paste commands, `.env.example`)
- prompt engineering approach (with the eval scores from Phase 7)
- 4–5 example conversations (real transcripts)
- business strategy summary (link to `docs/USE_CASE.md`)
- assumptions, limitations, future enhancements

Demo video (3–5 min): architecture in 30 s → 3 happy-path journeys → 1 error case → tool-call view → business summary.

Final checklist:
- [ ] repo is private, collaborators added
- [ ] fresh clone + setup instructions work on a clean machine
- [ ] no API keys in git history
- [ ] all tests green
- [ ] video link works without login
