"""Parsers for Clerk's plain-text statement and agreement layout.

The demo documents and the synthetic eval set use this layout. The offline
simulator reads documents with these parsers; in live mode, Nemotron reads any
layout (including photos) and these are not used.
"""

from __future__ import annotations

import re
from typing import Any

_ROW = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.+?)\s*\|\s*([\d.]+)\s*\|\s*\$?(-?[\d,]*\.?\d+)\s*\|\s*\$?(-?[\d,]*\.?\d+)\s*$"
)
_MONEY = r"\$?(-?[\d,]*\.?\d+)"


def _money(value: str) -> float:
    return round(float(value.replace(",", "")), 2)


def _field(text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{label}\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def _provider(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip().title()
    return "Unknown"


def classify(text: str) -> str:
    head = "\n".join(text.strip().splitlines()[:4]).upper()
    if any(word in head for word in ("AGREEMENT", "CONTRACT", "LEASE", "TERMS OF SERVICE")):
        return "contract"
    if any(word in head for word in ("STATEMENT", "INVOICE", "BILL")):
        return "bill"
    return "unknown"


def parse_bill_text(text: str) -> dict[str, Any] | None:
    items: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _ROW.match(line)
        if match:
            date, desc, qty, unit, amount = match.groups()
            items.append(
                {
                    "index": len(items),
                    "date": date,
                    "description": desc.strip(),
                    "quantity": float(qty),
                    "unit_price": _money(unit),
                    "amount": _money(amount),
                }
            )
    if not items:
        return None

    tax_rate = None
    tax = None
    tax_match = re.search(rf"^\s*Tax\s*\(([\d.]+)%\)\s*:\s*{_MONEY}\s*$", text, re.IGNORECASE | re.MULTILINE)
    if tax_match:
        tax_rate = round(float(tax_match.group(1)) / 100, 6)
        tax = _money(tax_match.group(2))

    subtotal = _field(text, "Subtotal")
    total = _field(text, r"TOTAL DUE")
    return {
        "provider": _provider(text),
        "account": _field(text, "Account"),
        "statement_date": _field(text, "Statement date"),
        "due_date": _field(text, "Due date"),
        "line_items": items,
        "subtotal": _money(subtotal.lstrip("$")) if subtotal else None,
        "tax": tax,
        "tax_rate": tax_rate,
        "total": _money(total.lstrip("$")) if total else None,
        "currency": "USD",
        "customer": _field(text, "Customer"),
    }


def parse_contract_text(text: str) -> dict[str, Any] | None:
    if classify(text) != "contract":
        return None
    facts: dict[str, Any] = {
        "provider": _provider(text),
        "account": _field(text, "Account"),
        "plan_name": None,
        "monthly_rate": None,
        "rate_locked_until": None,
        "included_services": [],
        "late_fee_cap": None,
        "dispute_window_days": None,
        "other": [],
    }
    plan = re.search(r"Plan\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if plan:
        facts["plan_name"] = plan.group(1).strip().rstrip(".")
    rate = re.search(
        rf"(?:Monthly rate|Monthly rent)\s*:\s*{_MONEY}(?:[^\n]*?locked until\s*(\d{{4}}-\d{{2}}-\d{{2}}))?",
        text,
        re.IGNORECASE,
    )
    if rate:
        facts["monthly_rate"] = _money(rate.group(1))
        facts["rate_locked_until"] = rate.group(2)
    services = re.search(r"Included services\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if services:
        facts["included_services"] = [s.strip().rstrip(".") for s in services.group(1).split(";") if s.strip()]
    fee = re.search(rf"Late fee\s*:\s*maximum\s*{_MONEY}", text, re.IGNORECASE)
    if fee:
        facts["late_fee_cap"] = _money(fee.group(1))
    window = re.search(r"within\s+(\d+)\s+days", text, re.IGNORECASE)
    if window:
        facts["dispute_window_days"] = int(window.group(1))
    return facts
