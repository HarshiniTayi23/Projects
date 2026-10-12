"""Deterministic stand-ins for each Nemotron task (offline mode only).

Each simulator returns text in the same format the real model is asked for, so
the agents' parsing, validation and fallback code paths are exercised exactly
as they are in live mode.
"""

from __future__ import annotations

import json

from ..llm.base import LLMRequest
from ..llm.offline import register
from .matching import norm, similarity
from .textformat import classify, parse_bill_text, parse_contract_text


@register("reader")
def reader(request: LLMRequest) -> str:
    text = str(request.payload.get("text", ""))
    kind = classify(text)
    if kind == "bill":
        bill = parse_bill_text(text)
        if bill:
            return json.dumps({"kind": "bill", "provider": bill["provider"], "bill": bill, "contract": None})
    if kind == "contract":
        contract = parse_contract_text(text)
        if contract:
            return json.dumps(
                {"kind": "contract", "provider": contract["provider"], "bill": None, "contract": contract}
            )
    first = next((line.strip().title() for line in text.splitlines() if line.strip()), "Unknown provider")
    return json.dumps({"kind": "unknown", "provider": first, "bill": None, "contract": None})


@register("ai_auditor")
def ai_auditor(request: LLMRequest) -> str:
    # The open-ended auditor needs a real model; offline it reports nothing extra.
    return json.dumps({"findings": []})


@register("skeptic")
def skeptic(request: LLMRequest) -> str:
    p = request.payload
    acc = p["accusation"]
    lines = p["cited_lines"]
    agreement = p.get("agreement") or {}
    statement_date = p["statement"].get("statement_date") or ""
    type_ = acc["type"]

    def out(verdict: str, reason: str, confidence: float) -> str:
        return f"<think>Weighing the evidence.</think>\n{json.dumps({'verdict': verdict, 'reason': reason, 'confidence': confidence})}"

    if type_ == "duplicate":
        dates = {li.get("date") for li in lines}
        descs = {li["description"] for li in lines}
        if len(dates) > 1 and len(descs) > 1:
            return out(
                "dismissed",
                "The lines have different dates and numbered descriptions, so they look like separate installments, not a double charge.",
                0.8,
            )
        return out("upheld", "Same description, same amount, same date: this is the same charge billed twice.", 0.92)

    if type_ == "rate":
        lock = agreement.get("rate_locked_until")
        if lock and statement_date and statement_date > lock:
            return out(
                "dismissed",
                f"The promotional rate was only locked until {lock}; this statement is dated {statement_date}, so a higher rate may be allowed.",
                0.75,
            )
        return out(
            "upheld",
            f"The agreement locks the rate until {lock or 'further notice'} and this statement falls inside that period.",
            0.9,
        )

    if type_ == "unauthorized":
        services = agreement.get("included_services") or []
        for line in lines:
            match = next((s for s in services if similarity(line["description"], s) >= 0.6), None)
            if match:
                return out(
                    "dismissed", f"'{line['description']}' matches '{match}', which your agreement includes.", 0.8
                )
        names = ", ".join(services) or "none listed"
        return out("upheld", f"Nothing in the agreement covers this charge (included: {names}).", 0.85)

    if type_ == "math":
        amount = float(acc.get("amount") or 0)
        if amount <= 0.011:
            return out("dismissed", "A one-cent difference is ordinary rounding, not overbilling.", 0.7)
        return out("upheld", "The verifier's recomputation shows the bill overstates the amount.", 0.95)

    if type_ == "fee":
        return out("upheld", "The fee exceeds the cap written into the agreement.", 0.9)

    # Model-found findings can't be judged meaningfully offline; stay cautious.
    cited = ", ".join(norm(li["description"]) for li in lines) or "no lines"
    return out("dismissed", f"Offline simulator can't evaluate open-ended findings ({cited}).", 0.3)


@register("advocate")
def advocate(request: LLMRequest) -> str:
    # The template letter is what the offline demo shows; live mode asks Nemotron Super to write it.
    return str(request.payload.get("fallback_letter", ""))


@register("ask")
def ask(request: LLMRequest) -> str:
    facts = request.payload.get("facts") or []
    if not facts:
        return "I don't have anything about that in your Vault yet. Upload the agreement and I'll remember it."
    lines = [f"- {f['provider']}: {f['fact'].replace('_', ' ')} = {f['value']}" for f in facts]
    return "Here's what I remember that looks relevant:\n" + "\n".join(lines)
