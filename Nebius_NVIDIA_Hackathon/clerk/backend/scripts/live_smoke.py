"""Live smoke test: one real end-to-end audit against Nebius Token Factory (and Tavily if configured).

    CLERK_MODE=live NEBIUS_API_KEY=... CLERK_MODEL_SUPER=... CLERK_MODEL_NANO=... python scripts/live_smoke.py

Checks, in order: every configured model answers; the Reader parses a real agreement and statement;
the Tribunal upholds the planted Northwind overcharges; the trace shows live Nebius calls.
Exit code 0 means Clerk is ready for live use.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clerk import ingest
from clerk.agents.runtime import AgentContext
from clerk.agents.tribunal import run_tribunal
from clerk.config import load_settings
from clerk.db import Database
from clerk.llm import LLMRequest, build_provider
from clerk.schemas import Bill
from clerk.search import build_search

OK = "\033[32mPASS\033[0m"
BAD = "\033[31mFAIL\033[0m"


def main() -> int:
    settings = load_settings()
    if not settings.live:
        print("Set CLERK_MODE=live (and your keys) to run the live smoke test.")
        return 1
    llm = build_provider(settings)
    search = build_search(settings)
    failures = 0

    roles = ["super", "nano"] + (["vision"] if settings.model_vision else [])
    for role in roles:
        try:
            result = llm.complete(
                LLMRequest(
                    task="ping",
                    role=role,
                    max_tokens=300,
                    messages=[{"role": "user", "content": "Reply with the word ready."}],
                )
            )
            print(f"{OK} {role:6} {result.model}: {result.latency_ms} ms, {result.completion_tokens} tokens")
        except Exception as exc:
            failures += 1
            print(f"{BAD} {role:6} {llm.model_for(role)}: {exc}")
    if failures:
        return 1

    if settings.model_embed:
        try:
            [vec] = llm.embed(["late fee cap"])
            print(f"{OK} embed  {settings.model_embed}: {len(vec)} dimensions")
        except Exception as exc:
            failures += 1
            print(f"{BAD} embed  {settings.model_embed}: {exc}")

    print(f"{'    ' if search.mode == 'live' else 'SKIP'} search mode: {search.mode}")

    db = Database(":memory:")
    ctx = AgentContext(settings=settings, llm=llm, search=search, db=db, workspace="smoke")
    demo = ingest.DEMO_DIR
    contract = ingest.ingest(ctx, "agreement.txt", (demo / "01_northwind_agreement.txt").read_bytes())
    print(
        f"{OK if contract['kind'] == 'contract' and contract['facts_remembered'] >= 4 else BAD} Reader on agreement: {contract}"
    )
    bill_doc = ingest.ingest(ctx, "statement.txt", (demo / "11_northwind_march_statement.txt").read_bytes())
    print(
        f"{OK if bill_doc['kind'] == 'bill' and bill_doc['line_items'] == 3 else BAD} Reader on statement: {bill_doc}"
    )
    doc = db.get_document("smoke", bill_doc["id"])
    if not doc or doc["kind"] != "bill":
        return 1

    ctx.audit_id = "smoke"
    ctx.emit = lambda e: (
        print(f"     {e['stage']:10} {e['status']:8} {e.get('detail', '')}") if e["type"] == "stage" else None
    )
    summary = run_tribunal(ctx, Bill.from_dict(doc["parsed"]), today=dt.date.today())
    upheld = {(f["type"], f["verified_amount"]) for f in summary["findings"] if f["verdict"] == "upheld"}
    expected = {("rate", 30.0), ("unauthorized", 9.99)}
    status = OK if expected <= upheld else BAD
    if status == BAD:
        failures += 1
    print(
        f"{status} Tribunal upheld {sorted(upheld)} (expected at least {sorted(expected)}); total ${summary['recoverable_total']}"
    )
    for f in summary["findings"]:
        print(f"     [{f['verdict']:9}] {f['title']}: {f['verdict_reason']}")

    traces = db.list_traces("smoke")
    live_calls = [t for t in traces if t["mode"] == "live"]
    print(
        f"{OK if live_calls else BAD} {len(live_calls)} live Nebius calls traced, "
        f"{sum(t['prompt_tokens'] + t['completion_tokens'] for t in live_calls):,} tokens"
    )
    letters = [a for a in db.list_actions("smoke") if a["type"] == "letter"]
    if letters:
        print("\n----- Advocate's letter -----\n" + letters[0]["content"][:1500])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
