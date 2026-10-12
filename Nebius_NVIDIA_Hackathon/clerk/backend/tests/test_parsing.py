import pytest

from clerk.agents.textformat import classify, parse_bill_text, parse_contract_text
from clerk.llm.jsonutil import JSONExtractionError, extract_json, strip_reasoning
from tests.conftest import demo_text


def test_classify_demo_documents():
    assert classify(demo_text("01_northwind_agreement.txt")) == "contract"
    assert classify(demo_text("03_lakeview_lease.txt")) == "contract"
    assert classify(demo_text("11_northwind_march_statement.txt")) == "bill"
    assert classify("Dear Maria, happy birthday!") == "unknown"


def test_parse_bill_reads_lines_and_totals():
    bill = parse_bill_text(demo_text("12_brightside_april_statement.txt"))
    assert bill is not None
    assert bill["provider"] == "Brightside Energy"
    assert bill["account"] == "BSE-20931"
    assert bill["customer"] == "Maria Alvarez"
    assert [li["amount"] for li in bill["line_items"]] == [85.0, 4.5, 4.5, 35.0]
    assert bill["subtotal"] == 129.0
    assert bill["tax_rate"] == 0.08
    assert bill["tax"] == 10.32
    assert bill["total"] == 149.32


def test_parse_bill_handles_thousands_separators():
    bill = parse_bill_text(demo_text("13_lakeview_may_statement.txt"))
    assert bill is not None
    assert bill["line_items"][0]["amount"] == 1450.0
    assert bill["subtotal"] == 1865.0
    assert bill["tax"] is None


def test_parse_bill_returns_none_without_rows():
    assert parse_bill_text("ACME\nMONTHLY STATEMENT\nnothing here") is None


def test_parse_contract_extracts_terms():
    facts = parse_contract_text(demo_text("01_northwind_agreement.txt"))
    assert facts == {
        "provider": "Northwind Fiber",
        "account": "NWF-448812",
        "plan_name": "Fiber 500 Internet",
        "monthly_rate": 39.99,
        "rate_locked_until": "2027-03-31",
        "included_services": ["Fiber 500 Internet", "Router rental"],
        "late_fee_cap": 10.0,
        "dispute_window_days": 60,
        "other": [],
    }
    lease = parse_contract_text(demo_text("03_lakeview_lease.txt"))
    assert lease is not None and lease["monthly_rate"] == 1450.0


def test_parse_contract_rejects_bills():
    assert parse_contract_text(demo_text("11_northwind_march_statement.txt")) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('<think>hmm {"no": 1}</think>\n{"a": 2}', {"a": 2}),
        ('Sure! Here it is:\n```json\n{"a": 3}\n```', {"a": 3}),
        ('prefix {"a": "brace } in string", "b": [1, 2,]} suffix', {"a": "brace } in string", "b": [1, 2]}),
        ("[1, 2, 3]", [1, 2, 3]),
        ('<think>truncated reasoning with no end {"x": 1}', None),
    ],
)
def test_extract_json(text, expected):
    if expected is None:
        with pytest.raises(JSONExtractionError):
            extract_json(text)
    else:
        assert extract_json(text) == expected


def test_strip_reasoning():
    assert strip_reasoning("<think>a</think>  hello ") == "hello"
    assert strip_reasoning("<THINK>a\nb</THINK>x") == "x"
