# What You Need to Learn

This is everything required to build the GoodFoods reservation agent, ordered from most to least critical. Estimated times assume you already know basic Python.

---

## 1. Python foundations (must already be solid)

| Topic | Why it matters here | Time |
|---|---|---|
| Functions, classes, dataclasses | Tools, agent, and data models are all classes/functions | review only |
| Type hints | Tool schemas are generated from typed function signatures | 1 h |
| `json` module | Every tool call and tool result is JSON | 30 min |
| Exceptions (`try/except`, custom errors) | Error handling is 1 of 5 graded technical criteria | 1 h |
| `datetime` / `zoneinfo` | Parsing "tomorrow at 8pm", checking time-slot overlap | 1–2 h |
| Virtual envs + `requirements.txt` | Reviewers must be able to run your setup | 30 min |
| `.env` files (`python-dotenv`) | Keep API keys out of git | 15 min |

## 2. LLM basics (core concept, learn first)

- **Chat Completions format** – the `messages` list with `system`, `user`, `assistant`, `tool` roles.
- **Tokens, context window, temperature** – why you use `temperature=0`–`0.3` for tool calling.
- **OpenAI-compatible APIs** – Groq, Together, OpenRouter, Fireworks all expose the same `/chat/completions` endpoint, so one client works for any provider.
- **Small-model limits** – 8B models hallucinate arguments, forget earlier turns, and sometimes emit malformed JSON. Your code must defend against this.

Resources: OpenAI "Function calling" guide (the format is the de-facto standard), Groq docs on tool use, Meta's Llama 3.1/3.3 model card (prompt format section).

Time: 3–4 h

## 3. Tool / function calling (the heart of the challenge)

You must understand this loop end-to-end and be able to write it **without LangChain**:

```
user message
  → send messages + tool schemas to LLM
  → LLM returns either (a) text or (b) tool_calls [{name, arguments}]
  → if tool_calls: run each tool in Python, append a `tool` message with the result
  → send everything back to the LLM
  → repeat until LLM returns plain text (or you hit a max-steps limit)
```

Learn:
- **JSON Schema** (`type`, `properties`, `required`, `enum`, `description`) – how tools are described to the model.
- **Writing good tool descriptions** – the model chooses tools by reading them; this is where "LLM determines intent" happens.
- **Parallel tool calls** – the model may return more than one call at once.
- **Validating arguments** – use `pydantic` to reject bad args and return a helpful error to the model so it can self-correct.

Time: 4–6 h (build a toy weather/calculator agent first)

## 4. Model Context Protocol (MCP)

The challenge asks for "MCP, A2A or other latest protocols." MCP is the simplest fit.

- **Concepts**: server, client, tools, resources, prompts, transports (`stdio`, streamable HTTP).
- **Python SDK**: the official `mcp` package. `FastMCP` lets you expose a Python function as a tool with a decorator.
- **Architecture you will build**:
  - `mcp_server/` – exposes `search_restaurants`, `check_availability`, `make_reservation`, etc.
  - Your agent acts as the **MCP client**: it calls `list_tools()`, converts those schemas to the LLM's tool format, and forwards the LLM's tool calls via `call_tool()`.
- Bonus understanding: what A2A (Agent-to-Agent) is and why you did *not* need it (single agent) – good for the README.

Resources: modelcontextprotocol.io docs ("Build a server", "Build a client" Python quickstarts), MCP Inspector tool for debugging.

Time: 3–4 h

## 5. Prompt engineering for small models

- Structured system prompts: role → rules → tool-usage policy → output style.
- Injecting dynamic context (today's date, timezone) – small models can't know "tomorrow" otherwise.
- Slot filling: asking for missing info (party size, date, time, name, phone) one item at a time.
- Confirm-before-write: always summarize and get "yes" before `make_reservation`.
- Guardrails: off-topic requests, prompt injection, refusing to invent restaurants.
- Few-shot examples inside the system prompt and how they cost context.

Time: 2–3 h (mostly iteration while testing)

## 6. Data layer

- **SQLite** with Python's built-in `sqlite3` (or SQLModel/SQLAlchemy if you prefer) – tables for restaurants, tables/capacity, reservations.
- Basic SQL: `SELECT ... WHERE`, `JOIN`, `INSERT`, transactions (prevent double-booking).
- **Synthetic data generation** with `Faker` + `random` – 50–100 locations with cuisine, area, price, capacity, hours, features (outdoor, vegan, parking).
- Availability logic: a reservation occupies a slot (e.g. 90 min); capacity check = seats booked in overlapping slots + party size ≤ capacity.

Time: 3–4 h

## 7. Recommendation logic

Keep it explainable rather than fancy:
- Hard filters: cuisine, area, party size fits, open at requested time.
- Soft scoring: rating, price match, features, distance/area match.
- Optional: embeddings (`sentence-transformers`) for "somewhere romantic with a view" style queries.
- Fallback: if nothing fits, relax one constraint and say which one.

Time: 2 h

## 8. Streamlit (frontend)

- `st.chat_message`, `st.chat_input`, `st.session_state` (conversation memory across reruns).
- `st.sidebar` for showing current bookings / debug view of tool calls (reviewers love seeing tool calls).
- `st.spinner`, `st.expander`, `st.dataframe`.

Time: 2 h

## 9. Testing & quality

- `pytest` – unit tests for tools and availability logic (no LLM needed).
- Mocking the LLM client so agent-loop tests are deterministic.
- A small **eval script**: list of user utterances → expected tool name; report accuracy.
- `ruff` / `black` for formatting, `logging` instead of prints.

Time: 2–3 h

## 10. Business strategy (40% of the grade)

Not code, but you still need to learn:
- The use case template structure (problem, users, goals, metrics, scope, risks, timeline, stakeholders).
- **Restaurant economics**: no-show rates, table turnover, covers per day, average check size, peak vs off-peak utilization.
- **Metrics/ROI**: booking conversion rate, no-show reduction, staff hours saved on calls, off-peak fill rate, cost per conversation vs. cost of a host.
- **Vertical expansion**: salons/spas, clinics, hotels, coworking, event venues, cloud kitchens – anything with "capacity × time slot".
- **Competitive positioning**: vs. OpenTable/Zomato/Dineout/SevenRooms – think voice + Indic languages, WhatsApp channel, owned customer data, upsell during booking.

Time: 4–6 h of reading + writing

## 11. Delivery skills

- Git basics, private GitHub repo, adding collaborators.
- Screen recording a demo (OBS / Loom) – 3–5 minutes, show 3–4 user journeys + one error case.
- Writing a professional README.

---

## Suggested learning order (if you are starting cold)

1. Section 2 + 3 together: build a 50-line agent with one fake tool against Groq. **This is the single most important exercise.**
2. Section 6: build the database and seed script.
3. Section 4: move your tools behind an MCP server.
4. Section 8: wrap it in Streamlit.
5. Sections 5, 7, 9: iterate on quality.
6. Section 10: write the business doc while things are fresh.
