import datetime as dt
import json

from clerk import paperwork
from clerk.agents.textformat import parse_bill_text, parse_contract_text
from clerk.schemas import Bill, ContractFacts, Finding
from clerk.synth import generate_case, generate_suite
from evals import run_eval


def test_generator_is_deterministic():
    assert generate_case(7).to_dict() == generate_case(7).to_dict()
    assert generate_case(7).bill_text != generate_case(8).bill_text


def test_generated_documents_parse_and_add_up():
    for case in generate_suite(40, start_seed=1):
        bill = parse_bill_text(case.bill_text)
        contract = parse_contract_text(case.contract_text)
        assert bill and contract, case.case_id
        line_sum = round(sum(li["amount"] for li in bill["line_items"]), 2)
        planted_math = {p["check"]: p["amount"] for p in case.planted if p["type"] == "math"}
        assert round(bill["subtotal"] - line_sum, 2) == planted_math.get("subtotal", 0.0), case.case_id
        for p in case.planted:
            for i in p["line_indexes"]:
                assert 0 <= i < len(bill["line_items"]), case.case_id


def test_eval_runner_offline(tmp_path):
    report = run_eval.main(["--n", "12", "--out", str(tmp_path)])
    assert report["mode"] == "offline" and report["failures"] == []
    assert report["scores"]["tribunal"]["precision"] == 1.0
    assert report["scores"]["tribunal"]["recall"] == 1.0
    assert report["scores"]["candidates"]["precision"] <= report["scores"]["tribunal"]["precision"]
    md = (tmp_path / "offline.md").read_text()
    assert "NOT a measure of Nemotron's quality" in md
    assert json.loads((tmp_path / "offline.json").read_text())["cases"] == 12


def test_eval_scoring_matches_by_type_and_lines():
    s = run_eval.Score()
    preds = [
        Finding("a", "rate", "t", [0], 10.0, "r", "detector", verified_amount=10.0),
        Finding("b", "math", "t", [], 5.0, "r", "detector", verified_amount=5.0, meta={"check": "total"}),
        Finding("c", "fee", "t", [3], 7.0, "r", "detector", verified_amount=7.0),
    ]
    planted = [
        {"type": "rate", "line_indexes": [0], "amount": 10.0},
        {"type": "math", "line_indexes": [], "check": "subtotal", "amount": 5.0},
    ]
    run_eval.score(preds, planted, s)
    assert (s.tp, s.fp, s.fn) == (1, 2, 1)
    assert s.found_dollars == 10.0 and s.false_dollars == 12.0


BILL = Bill("Acme, Inc.", "A-1", "2026-03-05", None, [], customer=None)
FINDING = Finding("f", "fee", "Late fee over cap", [0], 25.0, "Cap is $10.", "detector", verified_amount=25.0)


def test_deadline_uses_contract_window_or_default():
    facts = ContractFacts(provider="Acme", dispute_window_days=60)
    assert paperwork.dispute_deadline(BILL, facts) == dt.date(2026, 5, 4)
    assert paperwork.dispute_deadline(BILL, None) == dt.date(2026, 4, 4)
    undated = Bill("Acme", None, "not a date", None, [])
    assert paperwork.dispute_deadline(undated, None, today=dt.date(2026, 1, 1)) == dt.date(2026, 1, 31)


def test_letter_and_script_contents():
    letter = paperwork.template_letter(BILL, [FINDING], dt.date(2026, 3, 9))
    assert letter.startswith("March 9, 2026")
    assert "credit $25.00" in letter and "[Your name]" in letter and "Late fee over cap: $25.00" in letter
    script = paperwork.call_script(BILL, [FINDING])
    assert "$25.00" in script and "reference number" in script


def test_ics_is_valid_and_escaped():
    ics = paperwork.reminder_ics("Dispute; Acme, Inc.", "line1\nline2", dt.date(2026, 5, 4))
    lines = ics.split("\r\n")
    assert lines[0] == "BEGIN:VCALENDAR" and lines[-2] == "END:VCALENDAR"
    assert "SUMMARY:Dispute\\; Acme\\, Inc." in lines
    assert "DESCRIPTION:line1\\nline2" in lines
    assert "DTSTART;VALUE=DATE:20260504" in lines and "DTEND;VALUE=DATE:20260505" in lines


def test_slug():
    assert paperwork.slug("Northwind Fiber 2026-03-05") == "northwind-fiber-2026-03-05"
    assert paperwork.slug("!!!") == "document"
