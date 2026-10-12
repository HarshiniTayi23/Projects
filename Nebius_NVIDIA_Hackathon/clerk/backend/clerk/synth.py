"""Synthetic bills with planted errors whose answers are known (the eval set).

Each case is an agreement plus a statement. Errors are planted on purpose, and
so are *decoys*: things that look wrong but are legitimate (installments, a fee
within the cap, a price rise after a promotion ended, a renamed included service,
one-cent tax rounding). A good auditor catches the errors and lets the decoys go.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Any

PROVIDERS: list[dict[str, Any]] = [
    {
        "name": "NORTHWIND FIBER",
        "acct": "NWF",
        "plan": "Fiber 500 Internet",
        "rate": (39.99, 79.99),
        "extras": [("Router rental", 5.00)],
        "addons": ["Premium Tech Support add-on", "Static IP add-on"],
        "tax": 0.05,
    },
    {
        "name": "BRIGHTSIDE ENERGY",
        "acct": "BSE",
        "plan": "Fixed Bill Plan",
        "rate": (60.00, 140.00),
        "extras": [("Smart thermostat rental", 4.50)],
        "addons": ["Home Protection Plus", "Paperless processing fee"],
        "tax": 0.08,
    },
    {
        "name": "PINECREST MOBILE",
        "acct": "PCM",
        "plan": "Unlimited Talk and Text",
        "rate": (25.00, 65.00),
        "extras": [("Device payment", 33.33)],
        "addons": ["International Roaming Pass", "Visual Voicemail Pro"],
        "tax": 0.07,
    },
    {
        "name": "SUMMIT FITNESS CLUB",
        "acct": "SFC",
        "plan": "Standard Membership",
        "rate": (29.00, 59.00),
        "extras": [("Locker rental", 8.00)],
        "addons": ["Personal Training Starter Pack", "Towel service"],
        "tax": 0.0,
    },
    {
        "name": "CLEARWATER UTILITIES",
        "acct": "CWU",
        "plan": "Residential Water Service",
        "rate": (35.00, 70.00),
        "extras": [("Meter maintenance", 3.25)],
        "addons": ["Leak Protection Program", "Water Quality Report fee"],
        "tax": 0.06,
    },
]

ERROR_TYPES = ("duplicate", "rate", "fee", "unauthorized", "math")
DECOY_TYPES = ("installment", "fee_within_cap", "rate_after_lock", "service_alias", "tax_rounding")


@dataclass
class Case:
    case_id: str
    contract_text: str
    bill_text: str
    planted: list[dict[str, Any]] = field(default_factory=list)
    decoys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "contract_text": self.contract_text,
            "bill_text": self.bill_text,
            "planted": self.planted,
            "decoys": self.decoys,
        }


def _m(x: float) -> str:
    return f"{x:,.2f}"


def render_contract(
    provider: dict[str, Any],
    account: str,
    customer: str,
    agreed: dt.date,
    rate: float,
    lock: dt.date,
    services: list[str],
    fee_cap: float,
    window: int,
) -> str:
    return "\n".join(
        [
            provider["name"],
            "SERVICE AGREEMENT",
            f"Account: {account}",
            f"Customer: {customer}",
            f"Agreement date: {agreed.isoformat()}",
            "",
            "TERMS",
            f"1. Plan: {provider['plan']}",
            f"2. Monthly rate: ${_m(rate)} per month, locked until {lock.isoformat()}.",
            f"3. Included services: {'; '.join(services)}",
            f"4. Late fee: maximum ${_m(fee_cap)} per late payment.",
            f"5. Billing disputes must be raised within {window} days of the statement date.",
            "",
        ]
    )


def render_bill(
    provider_name: str,
    account: str,
    customer: str,
    statement: dt.date,
    lines: list[dict[str, Any]],
    subtotal: float,
    tax_rate: float,
    tax: float,
    total: float,
) -> str:
    out = [
        provider_name,
        "MONTHLY STATEMENT",
        f"Account: {account}",
        f"Customer: {customer}",
        f"Statement date: {statement.isoformat()}",
        f"Due date: {(statement + dt.timedelta(days=20)).isoformat()}",
        "",
        "CHARGES",
        "Date       | Description | Qty | Unit price | Amount",
    ]
    for li in lines:
        out.append(
            f"{li['date']} | {li['description']} | {li['quantity']:g} | {_m(li['unit_price'])} | {_m(li['amount'])}"
        )
    out += ["", f"Subtotal: {_m(subtotal)}"]
    if tax_rate:
        out.append(f"Tax ({tax_rate * 100:.2f}%): {_m(tax)}")
    out += [f"TOTAL DUE: {_m(total)}", ""]
    return "\n".join(out)


def _line(date: dt.date, desc: str, price: float, qty: float = 1.0) -> dict[str, Any]:
    return {
        "date": date.isoformat(),
        "description": desc,
        "quantity": qty,
        "unit_price": round(price, 2),
        "amount": round(price * qty, 2),
    }


def generate_case(seed: int) -> Case:
    rng = random.Random(seed)
    p = rng.choice(PROVIDERS)
    customer = rng.choice(["Maria Alvarez", "Sam Okafor", "Priya Raman", "Lena Fischer", "Tomas Herrera"])
    account = f"{p['acct']}-{rng.randint(10000, 99999)}"
    statement = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(30, 300))
    period = statement.replace(day=1)
    agreed = statement - dt.timedelta(days=rng.randint(60, 200))
    rate = round(rng.uniform(*p["rate"]), 2)
    fee_cap = float(rng.choice([10, 15, 25, 50]))
    window = rng.choice([30, 60, 90])

    errors = rng.sample(ERROR_TYPES, k=rng.randint(0, 3))
    decoys = rng.sample(DECOY_TYPES, k=rng.randint(0, 2))
    if "rate" in errors and "rate_after_lock" in decoys:
        decoys.remove("rate_after_lock")
    if "fee" in errors and "fee_within_cap" in decoys:
        decoys.remove("fee_within_cap")

    lock = (
        statement - dt.timedelta(days=rng.randint(10, 60))
        if "rate_after_lock" in decoys
        else statement + dt.timedelta(days=rng.randint(90, 400))
    )
    extra_name, extra_price = p["extras"][0]
    services = [p["plan"], extra_name]

    lines: list[dict[str, Any]] = []
    planted: list[dict[str, Any]] = []

    plan_price = rate
    if "rate" in errors:
        plan_price = round(rate + rng.choice([5.0, 10.0, 15.0, 20.0, 30.0]), 2)
    elif "rate_after_lock" in decoys:
        plan_price = round(rate + rng.choice([5.0, 10.0]), 2)
    lines.append(_line(period, p["plan"], plan_price))
    if "rate" in errors:
        planted.append({"type": "rate", "line_indexes": [0], "amount": round(plan_price - rate, 2)})

    extra_desc = f"{extra_name} - monthly" if "service_alias" in decoys else extra_name
    lines.append(_line(period, extra_desc, extra_price))

    if "duplicate" in errors:
        lines.append(_line(period, extra_desc, extra_price))
        planted.append({"type": "duplicate", "line_indexes": [1, 2], "amount": extra_price})

    if "unauthorized" in errors:
        addon = rng.choice(p["addons"])
        price = rng.choice([4.99, 7.99, 9.99, 14.99])
        lines.append(_line(period, addon, price))
        planted.append({"type": "unauthorized", "line_indexes": [len(lines) - 1], "amount": price})

    if "installment" in decoys:
        services.append("Equipment installment")
        amount = rng.choice([25.0, 40.0, 60.0])
        prev = (period - dt.timedelta(days=1)).replace(day=1)
        lines.append(_line(prev, "Equipment installment 1 of 2", amount))
        lines.append(_line(period, "Equipment installment 2 of 2", amount))

    if "fee" in errors:
        over = rng.choice([5.0, 10.0, 20.0])
        lines.append(_line(period + dt.timedelta(days=14), "Late payment fee", fee_cap + over))
        planted.append({"type": "fee", "line_indexes": [len(lines) - 1], "amount": over})
    elif "fee_within_cap" in decoys:
        lines.append(_line(period + dt.timedelta(days=14), "Late payment fee", fee_cap - rng.choice([0.0, 5.0])))

    line_sum = round(sum(li["amount"] for li in lines), 2)
    subtotal = line_sum
    math_check = None
    if "math" in errors:
        math_check = rng.choice(["subtotal", "total"])
        delta = rng.choice([1.0, 5.0, 10.0, 12.5])
        if math_check == "subtotal":
            subtotal = round(line_sum + delta, 2)
        planted.append({"type": "math", "line_indexes": [], "check": math_check, "amount": delta})

    tax_rate = p["tax"]
    tax = round(subtotal * tax_rate, 2)
    if "tax_rounding" in decoys and tax_rate:
        tax = round(tax + 0.01, 2)
    elif "tax_rounding" in decoys:
        decoys.remove("tax_rounding")
    total = round(subtotal + tax, 2)
    if math_check == "total":
        total = round(total + planted[-1]["amount"], 2)

    return Case(
        case_id=f"case-{seed:04d}",
        contract_text=render_contract(p, account, customer, agreed, rate, lock, services, fee_cap, window),
        bill_text=render_bill(p["name"], account, customer, statement, lines, subtotal, tax_rate, tax, total),
        planted=planted,
        decoys=decoys,
    )


def generate_suite(n: int = 50, start_seed: int = 1000) -> list[Case]:
    return [generate_case(start_seed + i) for i in range(n)]
