"""The Researcher: finds current, citable guidance behind each upheld finding (Tavily)."""

from __future__ import annotations

from ..schemas import Finding
from ..search import SearchError
from .runtime import AgentContext

QUERIES = {
    "duplicate": "how to dispute a duplicate charge on a bill consumer rights",
    "math": "billing error dispute incorrect total consumer rights how to request correction",
    "rate": "company charged more than contract promotional rate locked price dispute",
    "fee": "late fee higher than contract allows dispute consumer rights",
    "unauthorized": "charged for service I did not order unauthorized add-on charge dispute",
    "other": "how to dispute a billing error with a company consumer rights",
}


def research(ctx: AgentContext, finding: Finding) -> list[dict[str, object]]:
    if ctx.search.mode != "live":
        return []
    query = QUERIES.get(finding.type, QUERIES["other"])
    try:
        results = ctx.search.search(query, max_results=3)
    except SearchError as exc:
        ctx.emit({"type": "notice", "message": f"Web research unavailable: {exc}"})
        return []
    return [r.to_dict() | {"query": query} for r in results]
