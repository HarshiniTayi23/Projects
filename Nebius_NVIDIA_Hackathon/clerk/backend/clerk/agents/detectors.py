"""Deterministic auditors: plain code that catches what code can catch.

These are tuned for *recall*: they flag anything that might be wrong and leave
it to the Verifier (arithmetic) and the Skeptic (judgement) to throw out the
false alarms. Every candidate names the exact bill lines it is about.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..schemas import Bill, ContractFacts, Finding
from .matching import is_late_fee, is_tax_line, mentions_plan, norm

EPS = 0.005


def _fid(type_: str, lines: list[int], suffix: str = "") -> str:
    return f"{type_}-{'-'.join(str(i) for i in lines) or 'bill'}{suffix}"


def _money(x: float) -> str:
    return f"${x:,.2f}"


def detect_duplicates(bill: Bill) -> list[Finding]:
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for item in bill.line_items:
        if item.amount <= 0 or is_tax_line(item.description):
            continue
        # Digits are stripped so "Installment 1 of 2" and "Installment 2 of 2" group together;
        # telling those apart is the Skeptic's job.
        key = (norm(re.sub(r"\d+", "", item.description)), round(item.amount, 2))
        groups[key].append(item.index)
    findings = []
    for (_, amount), idxs in groups.items():
        if len(idxs) > 1:
            first = bill.line_items[idxs[0]]
            findings.append(
                Finding(
                    id=_fid("duplicate", idxs),
                    type="duplicate",
                    title=f"Possible duplicate charge: {first.description}",
                    line_indexes=idxs,
                    claimed_amount=round(amount * (len(idxs) - 1), 2),
                    rationale=f"{len(idxs)} lines charge {_money(amount)} for what looks like the same item.",
                    source="detector",
                )
            )
    return findings


def detect_math(bill: Bill) -> list[Finding]:
    findings = []
    for item in bill.line_items:
        if item.unit_price is not None:
            expected = round(item.quantity * item.unit_price, 2)
            if item.amount - expected > EPS:
                findings.append(
                    Finding(
                        id=_fid("math", [item.index], "-line"),
                        type="math",
                        title=f"Line total doesn't match quantity x price: {item.description}",
                        line_indexes=[item.index],
                        claimed_amount=round(item.amount - expected, 2),
                        rationale=f"{item.quantity:g} x {_money(item.unit_price)} = {_money(expected)}, "
                        f"but the line charges {_money(item.amount)}.",
                        source="detector",
                        meta={"check": "line"},
                    )
                )
    line_sum = round(sum(i.amount for i in bill.line_items), 2)
    if bill.subtotal is not None and bill.subtotal - line_sum > EPS:
        findings.append(
            Finding(
                id=_fid("math", [], "-subtotal"),
                type="math",
                title="Subtotal is higher than the sum of the charges",
                line_indexes=[],
                claimed_amount=round(bill.subtotal - line_sum, 2),
                rationale=f"The lines add up to {_money(line_sum)} but the subtotal says {_money(bill.subtotal)}.",
                source="detector",
                meta={"check": "subtotal"},
            )
        )
    base = bill.subtotal if bill.subtotal is not None else line_sum
    if bill.tax is not None and bill.tax_rate is not None:
        expected_tax = round(base * bill.tax_rate, 2)
        if bill.tax - expected_tax > EPS:
            findings.append(
                Finding(
                    id=_fid("math", [], "-tax"),
                    type="math",
                    title="Tax is higher than the stated rate",
                    line_indexes=[],
                    claimed_amount=round(bill.tax - expected_tax, 2),
                    rationale=f"{bill.tax_rate * 100:g}% of {_money(base)} is {_money(expected_tax)}, "
                    f"but the bill charges {_money(bill.tax)} tax.",
                    source="detector",
                    meta={"check": "tax"},
                )
            )
    if bill.total is not None:
        expected_total = round(base + (bill.tax or 0.0), 2)
        if bill.total - expected_total > EPS:
            findings.append(
                Finding(
                    id=_fid("math", [], "-total"),
                    type="math",
                    title="Total due is higher than subtotal plus tax",
                    line_indexes=[],
                    claimed_amount=round(bill.total - expected_total, 2),
                    rationale=f"{_money(base)} + {_money(bill.tax or 0)} tax = {_money(expected_total)}, "
                    f"but the total due is {_money(bill.total)}.",
                    source="detector",
                    meta={"check": "total"},
                )
            )
    return findings


def _unit(item_amount: float, quantity: float, unit_price: float | None) -> float:
    if unit_price is not None:
        return unit_price
    return item_amount / quantity if quantity else item_amount


def detect_rate(bill: Bill, facts: ContractFacts | None) -> list[Finding]:
    if not facts or facts.monthly_rate is None or not facts.plan_name:
        return []
    findings = []
    for item in bill.line_items:
        if not mentions_plan(item.description, facts.plan_name):
            continue
        unit = _unit(item.amount, item.quantity, item.unit_price)
        if unit - facts.monthly_rate > EPS:
            over = round((unit - facts.monthly_rate) * item.quantity, 2)
            lock = f" (locked until {facts.rate_locked_until})" if facts.rate_locked_until else ""
            findings.append(
                Finding(
                    id=_fid("rate", [item.index]),
                    type="rate",
                    title=f"Charged above your contract rate: {item.description}",
                    line_indexes=[item.index],
                    claimed_amount=over,
                    rationale=f"Your agreement sets {facts.plan_name} at {_money(facts.monthly_rate)}/month{lock}; "
                    f"this bill charges {_money(unit)}.",
                    source="detector",
                )
            )
    return findings


def detect_fees(bill: Bill, facts: ContractFacts | None) -> list[Finding]:
    findings = []
    cap = facts.late_fee_cap if facts else None
    for item in bill.line_items:
        if not is_late_fee(item.description) or item.amount <= 0:
            continue
        claimed = round(item.amount - cap, 2) if cap is not None and item.amount > cap else item.amount
        cap_text = f"Your agreement caps late fees at {_money(cap)}." if cap is not None else "No fee cap on file."
        findings.append(
            Finding(
                id=_fid("fee", [item.index]),
                type="fee",
                title=f"Late fee to check: {item.description}",
                line_indexes=[item.index],
                claimed_amount=claimed,
                rationale=f"This bill charges a {_money(item.amount)} late fee. {cap_text}",
                source="detector",
            )
        )
    return findings


def detect_unauthorized(bill: Bill, facts: ContractFacts | None) -> list[Finding]:
    if not facts or not facts.included_services:
        return []
    allowed = {norm(s) for s in facts.included_services}
    findings = []
    for item in bill.line_items:
        desc = item.description
        if item.amount <= 0 or is_tax_line(desc) or is_late_fee(desc) or mentions_plan(desc, facts.plan_name):
            continue
        # Exact match only: "Router rental - monthly" vs "Router rental" is left for the Skeptic.
        if norm(desc) not in allowed:
            findings.append(
                Finding(
                    id=_fid("unauthorized", [item.index]),
                    type="unauthorized",
                    title=f"Charge not in your agreement: {desc}",
                    line_indexes=[item.index],
                    claimed_amount=round(item.amount, 2),
                    rationale=f"Your agreement lists these services: {', '.join(facts.included_services)}. "
                    f"'{desc}' isn't one of them.",
                    source="detector",
                )
            )
    return findings


def run_detectors(bill: Bill, facts: ContractFacts | None) -> list[Finding]:
    return (
        detect_duplicates(bill)
        + detect_math(bill)
        + detect_rate(bill, facts)
        + detect_fees(bill, facts)
        + detect_unauthorized(bill, facts)
    )
