"""Agents driven by a scripted 'live' model: parsing, guardrails and failure handling."""

import datetime as dt
import json
from dataclasses import replace

import pytest

from clerk import ingest, vault
from clerk.agents.llm_agents import ai_audit, cross_examine, draft_letter, read_document
from clerk.agents.runtime import AgentContext, BudgetExceeded
from clerk.agents.textformat import parse_bill_text, parse_contract_text
from clerk.agents.tribunal import run_tribunal
from clerk.llm.base import LLMError
from clerk.paperwork import template_letter
from clerk.schemas import Bill, ContractFacts, Finding, LineItem
from clerk.search import OfflineSearch, SearchResult
from tests.conftest import FakeProvider, demo_text


def live_ctx(settings, db, responses, search=None):
    return AgentContext(
        settings=replace(settings, mode="live"),
        llm=FakeProvider(responses),
        search=search or OfflineSearch(),
        db=db,
        workspace="w",
    )


MESSY_BILL = """Here is the extraction:
```json
{"kind": "bill", "provider": "Acme Power",
 "bill": {"account": "AP-1", "statement_date": "2026-03-05",
  "line_items": [
    {"description": "Energy", "amount": "$1,200.50", "quantity": "2", "unit_price": 600.25},
    {"description": "", "amount": 3},
    {"description": "Bad", "amount": "n/a"},
    {"description": "Fee", "amount": 4, "quantity": 0}
  ],
  "tax_rate": 5, "total": "1,204.50"}}
```"""


def test_reader_coerces_messy_model_output(settings, db):
    ctx = live_ctx(settings, db, {"reader": [MESSY_BILL]})
    result = read_document(ctx, text="ACME POWER statement")
    assert result.kind == "bill" and result.provider == "Acme Power"
    assert result.bill is not None
    assert [(li.index, li.description, li.amount, li.quantity) for li in result.bill.line_items] == [
        (0, "Energy", 1200.5, 2.0),
        (1, "Fee", 4.0, 1.0),
    ]
    assert result.bill.tax_rate == 0.05 and result.bill.total == 1204.5


def test_reader_retries_once_on_invalid_json(settings, db):
    ctx = live_ctx(settings, db, {"reader": ["I think it's a bill.", '{"kind": "unknown", "provider": "X"}']})
    assert read_document(ctx, text="x").kind == "unknown"
    assert ctx.llm.requests[1].messages[-1]["content"].startswith("That reply was not valid JSON")


def test_reader_gives_up_after_second_invalid_json(settings, db):
    ctx = live_ctx(settings, db, {"reader": ["nope", "still nope"]})
    with pytest.raises(LLMError, match="valid JSON"):
        read_document(ctx, text="x")


def test_reader_sends_images_to_the_vision_role(settings, db):
    ctx = live_ctx(settings, db, {"reader": ['{"kind": "unknown", "provider": "X", "transcript": "hello"}']})
    result = read_document(ctx, image=b"\x89PNG", mime="image/png")
    req = ctx.llm.requests[0]
    assert req.role == "vision"
    assert req.messages[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert result.text == "hello"


def test_bill_with_no_usable_lines_is_unknown(settings, db):
    ctx = live_ctx(settings, db, {"reader": ['{"kind": "bill", "provider": "X", "bill": {"line_items": []}}']})
    assert read_document(ctx, text="x").kind == "unknown"


BILL = Bill(
    "Acme",
    "A-1",
    "2026-03-05",
    None,
    [LineItem(0, "Mystery charge", 12.0), LineItem(1, "Plan", 40.0)],
    customer="Maria",
)


def test_ai_auditor_coerces_findings(settings, db):
    reply = json.dumps(
        {
            "findings": [
                {
                    "type": "weird",
                    "title": "Mystery",
                    "line_indexes": [0, "x", True],
                    "claimed_amount": "$12",
                    "rationale": "?",
                },
                {"type": "duplicate", "line_indexes": [1], "claimed_amount": 40},
                "garbage",
            ]
        }
    )
    ctx = live_ctx(settings, db, {"ai_auditor": [reply]})
    found = ai_audit(ctx, BILL, None, [])
    assert [(f.type, f.line_indexes, f.claimed_amount, f.source) for f in found] == [
        ("other", [0], 12.0, "llm"),
        ("other", [1], 40.0, "llm"),  # a one-line "duplicate" is not a duplicate
    ]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"verdict": "UPHELD", "reason": "clear", "confidence": 3}', ("upheld", "clear", 1.0)),
        ('{"verdict": "maybe", "reason": "unsure"}', ("dismissed", "unsure", 0.5)),
        ('<think>long</think>{"verdict": "dismissed", "confidence": -1}', ("dismissed", "No reason given.", 0.0)),
    ],
)
def test_skeptic_normalises_verdicts(settings, db, reply, expected):
    ctx = live_ctx(settings, db, {"skeptic": [reply]})
    f = Finding("f", "other", "t", [0], 12.0, "r", "llm")
    assert cross_examine(ctx, f, BILL, None) == expected
    assert ctx.llm.requests[0].role == "super"


def test_advocate_guardrail_falls_back_when_amount_missing(settings, db):
    f = Finding("f", "other", "Mystery", [0], 12.0, "r", "llm", verified_amount=12.0)
    fallback = template_letter(BILL, [f], dt.date(2026, 3, 10))
    ctx = live_ctx(settings, db, {"advocate": ["Dear Acme, please refund me some money. " * 10]})
    assert draft_letter(ctx, BILL, None, [f], fallback) == fallback
    good = "Dear Acme,\n" + "I was overcharged. " * 15 + "Please credit $12.00 to my account.\nMaria"
    ctx = live_ctx(settings, db, {"advocate": [f"<think>plan</think>{good}"]})
    assert draft_letter(ctx, BILL, None, [f], fallback) == good


def test_budget_is_enforced_in_live_mode(settings, db):
    ctx = live_ctx(replace(settings, daily_llm_call_limit=1), db, {"reader": ['{"kind":"unknown","provider":"X"}'] * 2})
    read_document(ctx, text="x")
    with pytest.raises(BudgetExceeded):
        read_document(ctx, text="x")


def test_traces_record_cost_when_prices_configured(settings, db):
    from clerk.config import ModelPrice

    s = replace(settings, prices={"fake/nano": ModelPrice(1.0, 2.0)})
    ctx = live_ctx(s, db, {"reader": ['{"kind":"unknown","provider":"X"}']})
    events = []
    ctx.emit = events.append
    read_document(ctx, text="x")
    [trace] = db.list_traces("w")
    assert trace["cost_usd"] == pytest.approx((100 * 1.0 + 20 * 2.0) / 1e6)
    assert events[0]["type"] == "trace" and events[0]["step"]["agent"] == "Reader"


class StubSearch:
    mode = "live"

    def search(self, query, max_results=3):
        return [SearchResult("FTC guidance", "https://example.org/guide", "snippet")]


def test_full_tribunal_with_scripted_live_models(settings, db):
    """Hallucinated AI findings are caught by the Verifier; a model error in the Skeptic never accuses."""
    responses = {
        "reader": [],
        "ai_auditor": [
            json.dumps(
                {
                    "findings": [
                        {
                            "type": "other",
                            "title": "Ghost line",
                            "line_indexes": [42],
                            "claimed_amount": 99,
                            "rationale": "x",
                        },
                        {
                            "type": "other",
                            "title": "Router price odd",
                            "line_indexes": [1],
                            "claimed_amount": 5,
                            "rationale": "y",
                        },
                    ]
                }
            )
        ],
        "skeptic": [
            '{"verdict": "upheld", "reason": "Rate is locked.", "confidence": 0.9}',
            '{"verdict": "dismissed", "reason": "Router is included.", "confidence": 0.8}',
            '{"verdict": "upheld", "reason": "Not in agreement.", "confidence": 0.9}',
            "not json",
            "still not json",
        ],
        "advocate": ["too short"],
    }
    ctx = live_ctx(settings, db, responses, search=StubSearch())
    contract = ContractFacts.from_dict(parse_contract_text(demo_text("01_northwind_agreement.txt")))
    vault.remember_contract(ctx, "doc", contract)
    statement = Bill.from_dict(parse_bill_text(demo_text("11_northwind_march_statement.txt")))
    events = []
    ctx.emit = events.append
    ctx.audit_id = "a1"
    summary = run_tribunal(ctx, statement, today=dt.date(2026, 3, 10))

    by_title = {f["title"]: f for f in summary["findings"]}
    assert by_title["Ghost line"]["verdict"] == "dismissed"
    assert "not on this bill" in by_title["Ghost line"]["verdict_reason"]
    assert by_title["Router price odd"]["verdict"] == "dismissed"
    assert "Could not be cross-examined" in by_title["Router price odd"]["verdict_reason"]
    assert summary["recoverable_total"] == 39.99 and summary["upheld"] == 2
    upheld = [f for f in summary["findings"] if f["verdict"] == "upheld"]
    assert all(f["sources"][0]["url"] == "https://example.org/guide" for f in upheld)
    letter = next(a for a in db.list_actions("w") if a["type"] == "letter")
    assert "$39.99" in letter["content"] and "https://example.org/guide" in letter["content"]
    stages = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
    assert stages[-1] == ("advocate", "done")
    assert [s for s, _ in stages].index("verifier") < [s for s, _ in stages].index("skeptic")


def test_ingest_rejects_photos_offline(ctx):
    with pytest.raises(ingest.IngestError, match="live mode"):
        ingest.ingest(ctx, "bill.png", b"\x89PNG....")


def test_research_failure_becomes_a_notice(ctx):
    from clerk.agents.researcher import research
    from clerk.search import SearchError

    class Broken:
        mode = "live"

        def search(self, query, max_results=3):
            raise SearchError("Tavily returned 432")

    ctx.search = Broken()
    events = []
    ctx.emit = events.append
    assert research(ctx, Finding("f", "fee", "t", [0], 1.0, "r", "detector")) == []
    assert events == [{"type": "notice", "message": "Web research unavailable: Tavily returned 432"}]
