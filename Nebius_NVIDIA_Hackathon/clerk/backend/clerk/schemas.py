"""Plain data structures shared by the agents, the API and the evals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FINDING_TYPES = ("duplicate", "math", "rate", "fee", "unauthorized", "other")


@dataclass
class LineItem:
    index: int
    description: str
    amount: float
    date: str | None = None
    quantity: float = 1.0
    unit_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Bill:
    provider: str
    account: str | None
    statement_date: str | None
    due_date: str | None
    line_items: list[LineItem]
    subtotal: float | None = None
    tax: float | None = None
    tax_rate: float | None = None
    total: float | None = None
    currency: str = "USD"
    customer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["line_items"] = [li.to_dict() for li in self.line_items]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bill:
        items = [LineItem(**li) for li in data.get("line_items", [])]
        rest = {k: v for k, v in data.items() if k != "line_items"}
        return cls(line_items=items, **rest)


@dataclass
class ContractFacts:
    provider: str
    account: str | None = None
    plan_name: str | None = None
    monthly_rate: float | None = None
    rate_locked_until: str | None = None
    included_services: list[str] = field(default_factory=list)
    late_fee_cap: float | None = None
    dispute_window_days: int | None = None
    other: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContractFacts:
        known: dict[str, Any] = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        known["included_services"] = list(known.get("included_services") or [])
        known["other"] = list(known.get("other") or [])
        return cls(**known)


@dataclass
class Finding:
    id: str
    type: str
    title: str
    line_indexes: list[int]
    claimed_amount: float
    rationale: str
    source: str  # "detector" or "llm"
    verified: bool | None = None
    verified_amount: float | None = None
    verification_note: str | None = None
    verdict: str | None = None  # "upheld" | "dismissed"
    verdict_reason: str | None = None
    confidence: float | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def amount(self) -> float:
        return self.verified_amount if self.verified_amount is not None else self.claimed_amount

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})
