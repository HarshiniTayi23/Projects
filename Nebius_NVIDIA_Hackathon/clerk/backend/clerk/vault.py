"""The Vault: Clerk's long-term memory of the user's agreements.

Contract facts are stored as editable key/value rows (one per fact) with an
embedding for semantic recall. When a bill is audited, the facts for the
matching provider are reassembled into a ContractFacts object, so any edit the
user makes in the UI immediately changes how future bills are checked.
"""

from __future__ import annotations

import math
from typing import Any

from .agents.matching import norm_provider
from .agents.runtime import AgentContext
from .schemas import Bill, ContractFacts

FACT_KEYS = (
    "account",
    "plan_name",
    "monthly_rate",
    "rate_locked_until",
    "included_services",
    "late_fee_cap",
    "dispute_window_days",
)
LIST_SEP = "; "


def _fact_text(provider: str, key: str, value: str) -> str:
    return f"{provider} {key.replace('_', ' ')} {value}"


def remember_contract(ctx: AgentContext, document_id: str, facts: ContractFacts) -> list[str]:
    rows: list[tuple[str, str]] = []
    for key in FACT_KEYS:
        value = getattr(facts, key)
        if value is None or value == []:
            continue
        rows.append((key, LIST_SEP.join(value) if isinstance(value, list) else str(value)))
    rows.extend(("note", note) for note in facts.other)
    if not rows:
        return []
    embeddings = ctx.llm.embed([_fact_text(facts.provider, k, v) for k, v in rows])
    return [
        ctx.db.add_fact(ctx.workspace, document_id, facts.provider, key, value, emb)
        for (key, value), emb in zip(rows, embeddings, strict=True)
    ]


def _parse(key: str, value: str) -> Any:
    if key in ("monthly_rate", "late_fee_cap"):
        try:
            return round(float(value.replace("$", "").replace(",", "")), 2)
        except ValueError:
            return None
    if key == "dispute_window_days":
        try:
            return int(float(value))
        except ValueError:
            return None
    if key == "included_services":
        return [s.strip() for s in value.split(";") if s.strip()]
    return value


def contracts_by_provider(facts: list[dict[str, Any]]) -> dict[str, ContractFacts]:
    grouped: dict[str, ContractFacts] = {}
    for row in facts:
        key = norm_provider(row["provider"])
        contract = grouped.setdefault(key, ContractFacts(provider=row["provider"]))
        if row["key"] == "note":
            contract.other.append(row["value"])
        elif row["key"] in FACT_KEYS:
            setattr(contract, row["key"], _parse(row["key"], row["value"]))
    return grouped


def find_contract(ctx: AgentContext, bill: Bill) -> ContractFacts | None:
    contracts = contracts_by_provider(ctx.db.list_facts(ctx.workspace))
    if bill.account:
        for contract in contracts.values():
            if contract.account and contract.account.strip().lower() == bill.account.strip().lower():
                return contract
    return contracts.get(norm_provider(bill.provider))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def search(ctx: AgentContext, query: str, k: int = 6) -> list[dict[str, Any]]:
    facts = ctx.db.list_facts(ctx.workspace)
    if not facts:
        return []
    [q] = ctx.llm.embed([query])
    scored = []
    for fact in facts:
        emb = fact.get("embedding")
        score = _cosine(q, emb) if emb else 0.0
        scored.append((score, fact))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**fact, "score": round(score, 4), "embedding": None} for score, fact in scored[:k] if score > 0]


def update_fact(ctx: AgentContext, fact_id: str, value: str) -> bool:
    row = next((f for f in ctx.db.list_facts(ctx.workspace) if f["id"] == fact_id), None)
    if row is None:
        return False
    [emb] = ctx.llm.embed([_fact_text(row["provider"], row["key"], value)])
    return ctx.db.update_fact(ctx.workspace, fact_id, value, emb)
