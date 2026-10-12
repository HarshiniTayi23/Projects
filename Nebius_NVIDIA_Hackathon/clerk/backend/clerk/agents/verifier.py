"""The Verifier: plain Python that recomputes every number a finding claims.

The AI proposes, the code verifies. Model-generated findings that cite lines
that don't exist, or claim more money than the cited lines contain, are
rejected or corrected here, before any human sees them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import Bill, ContractFacts, Finding
from .detectors import EPS, _unit
from .matching import is_late_fee


@dataclass
class Verification:
    ok: bool
    amount: float | None
    note: str


def _money(x: float) -> str:
    return f"${x:,.2f}"


def verify(finding: Finding, bill: Bill, facts: ContractFacts | None) -> Verification:
    n = len(bill.line_items)
    bad = [i for i in finding.line_indexes if not isinstance(i, int) or i < 0 or i >= n]
    if bad:
        return Verification(False, None, f"Cites line(s) {bad} that are not on this bill.")
    lines = [bill.line_items[i] for i in finding.line_indexes]

    if finding.type == "duplicate":
        if len(lines) < 2:
            return Verification(False, None, "A duplicate needs at least two lines.")
        amounts = {round(li.amount, 2) for li in lines}
        if len(amounts) != 1:
            return Verification(False, None, "The cited lines have different amounts, so they aren't duplicates.")
        return Verification(True, round(lines[0].amount * (len(lines) - 1), 2), "Amounts match exactly.")

    if finding.type == "math":
        check = finding.meta.get("check")
        line_sum = round(sum(li.amount for li in bill.line_items), 2)
        base = bill.subtotal if bill.subtotal is not None else line_sum
        if check == "line" and lines:
            li = lines[0]
            if li.unit_price is None:
                return Verification(False, None, "No unit price on that line to check against.")
            diff = round(li.amount - li.quantity * li.unit_price, 2)
        elif check == "subtotal" and bill.subtotal is not None:
            diff = round(bill.subtotal - line_sum, 2)
        elif check == "tax" and bill.tax is not None and bill.tax_rate is not None:
            diff = round(bill.tax - round(base * bill.tax_rate, 2), 2)
        elif check == "total" and bill.total is not None:
            diff = round(bill.total - (base + (bill.tax or 0.0)), 2)
        else:
            return Verification(False, None, "Not enough numbers on the bill to recompute this.")
        if diff <= EPS:
            return Verification(False, None, "Recomputed: the numbers add up.")
        return Verification(True, diff, f"Recomputed difference: {_money(diff)}.")

    if finding.type == "rate":
        if not facts or facts.monthly_rate is None:
            return Verification(False, None, "No contract rate on file to compare against.")
        if not lines:
            return Verification(False, None, "No line cited.")
        over = round(
            sum((_unit(li.amount, li.quantity, li.unit_price) - facts.monthly_rate) * li.quantity for li in lines), 2
        )
        if over <= EPS:
            return Verification(False, None, f"Charged at or below the contract rate of {_money(facts.monthly_rate)}.")
        return Verification(True, over, f"{_money(over)} above the contract rate of {_money(facts.monthly_rate)}.")

    if finding.type == "fee":
        if not lines or not all(is_late_fee(li.description) for li in lines):
            return Verification(False, None, "The cited line isn't a late fee.")
        fee = round(sum(li.amount for li in lines), 2)
        cap = facts.late_fee_cap if facts else None
        if cap is None:
            return Verification(True, fee, "No fee cap on file; the full fee is in question.")
        over = round(fee - cap, 2)
        if over <= EPS:
            return Verification(False, None, f"The {_money(fee)} fee is within your {_money(cap)} cap.")
        return Verification(True, over, f"{_money(over)} over your {_money(cap)} late-fee cap.")

    # "unauthorized" and model-found "other" findings: never more than the cited lines contain.
    if not lines:
        return Verification(False, None, "No line cited.")
    ceiling = round(sum(li.amount for li in lines), 2)
    if ceiling <= EPS:
        return Verification(False, None, "The cited lines don't charge anything.")
    if finding.claimed_amount - ceiling > EPS:
        return Verification(
            True, ceiling, f"Claim corrected from {_money(finding.claimed_amount)} to {_money(ceiling)}."
        )
    amount = round(finding.claimed_amount, 2) if finding.claimed_amount > EPS else ceiling
    return Verification(True, amount, "Amount confirmed against the bill.")
