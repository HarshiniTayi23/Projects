from clerk.agents import detectors
from clerk.agents.textformat import parse_bill_text, parse_contract_text
from clerk.agents.verifier import verify
from clerk.schemas import Bill, ContractFacts, Finding, LineItem
from tests.conftest import demo_text


def bill(*items, subtotal=None, tax=None, tax_rate=None, total=None, date="2026-03-05"):
    lines = [
        LineItem(index=i, description=d, amount=a, date=dt, quantity=1, unit_price=a)
        for i, (d, a, dt) in enumerate(items)
    ]
    return Bill("Acme", "A-1", date, None, lines, subtotal, tax, tax_rate, total)


FACTS = ContractFacts(
    provider="Acme",
    account="A-1",
    plan_name="Fiber 500 Internet",
    monthly_rate=40.0,
    rate_locked_until="2027-01-01",
    included_services=["Fiber 500 Internet", "Router rental"],
    late_fee_cap=10.0,
    dispute_window_days=60,
)


def finding(type_, lines, amount, **meta):
    return Finding(
        id="x", type=type_, title="t", line_indexes=lines, claimed_amount=amount, rationale="r", source="llm", meta=meta
    )


# ------------------------------------------------------------------ detectors


def test_duplicates_group_identical_and_numbered_lines():
    b = bill(("Router rental", 5.0, "2026-03-01"), ("Router rental", 5.0, "2026-03-01"), ("Other", 1.0, "2026-03-01"))
    [dup] = detectors.detect_duplicates(b)
    assert dup.line_indexes == [0, 1] and dup.claimed_amount == 5.0
    inst = bill(("Installment 1 of 2", 50.0, "2026-02-01"), ("Installment 2 of 2", 50.0, "2026-03-01"))
    assert len(detectors.detect_duplicates(inst)) == 1  # flagged; the Skeptic decides


def test_math_detects_line_subtotal_tax_and_total_overcharges():
    b = bill(("A", 10.0, None), ("B", 20.0, None), subtotal=35.0, tax_rate=0.1, tax=4.0, total=45.0)
    checks = {f.meta["check"]: f.claimed_amount for f in detectors.detect_math(b)}
    assert checks == {"subtotal": 5.0, "tax": 0.5, "total": 6.0}
    b.line_items[0].unit_price = 8.0
    assert any(f.meta["check"] == "line" and f.claimed_amount == 2.0 for f in detectors.detect_math(b))


def test_math_ignores_undercharges():
    b = bill(("A", 10.0, None), subtotal=9.0, total=9.0)
    assert detectors.detect_math(b) == []


def test_rate_fee_and_unauthorized_detectors():
    b = bill(
        ("Fiber 500 Internet", 69.99, None),
        ("Late payment fee", 35.0, None),
        ("Premium add-on", 9.99, None),
        ("Router rental", 5.0, None),
        ("Sales tax", 3.0, None),
    )
    [rate] = detectors.detect_rate(b, FACTS)
    assert rate.claimed_amount == 29.99
    [fee] = detectors.detect_fees(b, FACTS)
    assert fee.claimed_amount == 25.0
    [unauth] = detectors.detect_unauthorized(b, FACTS)
    assert unauth.line_indexes == [2]


def test_detectors_need_an_agreement_for_contract_checks():
    b = bill(("Fiber 500 Internet", 69.99, None), ("Premium add-on", 9.99, None))
    assert detectors.detect_rate(b, None) == []
    assert detectors.detect_unauthorized(b, None) == []


# ------------------------------------------------------------------- verifier


def test_verifier_rejects_hallucinated_lines():
    b = bill(("A", 10.0, None))
    v = verify(finding("other", [7], 50.0), b, None)
    assert not v.ok and "not on this bill" in v.note


def test_verifier_caps_model_claims_at_the_cited_lines():
    b = bill(("Mystery fee", 12.0, None))
    v = verify(finding("other", [0], 99.0), b, None)
    assert v.ok and v.amount == 12.0 and "corrected" in v.note


def test_verifier_duplicate_requires_equal_amounts():
    b = bill(("A", 10.0, None), ("A", 11.0, None))
    assert not verify(finding("duplicate", [0, 1], 10.0), b, None).ok


def test_verifier_fee_within_cap_is_dismissed():
    b = bill(("Late payment fee", 10.0, None))
    v = verify(finding("fee", [0], 10.0), b, FACTS)
    assert not v.ok and "within your $10.00 cap" in v.note


def test_verifier_rate_recomputes_from_contract():
    b = bill(("Fiber 500 Internet", 45.0, None))
    assert verify(finding("rate", [0], 999.0), b, FACTS).amount == 5.0
    assert not verify(finding("rate", [0], 5.0), b, None).ok


def test_verifier_math_recomputes():
    b = bill(("A", 10.0, None), subtotal=10.0, tax=1.0, tax_rate=0.1, total=13.0)
    assert verify(finding("math", [], 99.0, check="total"), b, None).amount == 2.0
    assert not verify(finding("math", [], 1.0, check="tax"), b, None).ok


def test_demo_documents_produce_expected_candidates():
    contract = ContractFacts.from_dict(parse_contract_text(demo_text("01_northwind_agreement.txt")))
    statement = Bill.from_dict(parse_bill_text(demo_text("11_northwind_march_statement.txt")))
    found = {(f.type, tuple(f.line_indexes)) for f in detectors.run_detectors(statement, contract)}
    assert found == {("rate", (0,)), ("unauthorized", (1,)), ("unauthorized", (2,))}
