"""Agents that call Nemotron through the provider: Reader, AI Auditor, Skeptic, Advocate, Ask."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from ..llm.base import LLMError, LLMRequest
from ..llm.jsonutil import strip_reasoning
from ..schemas import FINDING_TYPES, Bill, ContractFacts, Finding, LineItem
from . import prompts
from .runtime import AgentContext, as_json

# --------------------------------------------------------------------------- helpers


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2) if not isinstance(value, bool) else None
    try:
        return round(float(str(value).replace("$", "").replace(",", "").strip()), 2)
    except ValueError:
        return None


def _str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce_bill(provider: str, data: dict[str, Any]) -> Bill | None:
    items: list[LineItem] = []
    for raw in data.get("line_items") or []:
        if not isinstance(raw, dict):
            continue
        amount = _num(raw.get("amount"))
        desc = _str(raw.get("description"))
        if amount is None or not desc:
            continue
        qty = _num(raw.get("quantity"))
        items.append(
            LineItem(
                index=len(items),
                description=desc,
                amount=amount,
                date=_str(raw.get("date")),
                quantity=qty if qty and qty > 0 else 1.0,
                unit_price=_num(raw.get("unit_price")),
            )
        )
    if not items:
        return None
    tax_rate = data.get("tax_rate")
    rate = float(tax_rate) if isinstance(tax_rate, (int, float)) and not isinstance(tax_rate, bool) else None
    if rate is not None and rate > 1:  # model wrote 5 for 5%
        rate = rate / 100
    return Bill(
        provider=provider,
        account=_str(data.get("account")),
        customer=_str(data.get("customer")),
        statement_date=_str(data.get("statement_date")),
        due_date=_str(data.get("due_date")),
        line_items=items,
        subtotal=_num(data.get("subtotal")),
        tax=_num(data.get("tax")),
        tax_rate=rate,
        total=_num(data.get("total")),
    )


def coerce_contract(provider: str, data: dict[str, Any]) -> ContractFacts:
    window = data.get("dispute_window_days")
    services = data.get("included_services") or []
    other = data.get("other") or []
    return ContractFacts(
        provider=provider,
        account=_str(data.get("account")),
        plan_name=_str(data.get("plan_name")),
        monthly_rate=_num(data.get("monthly_rate")),
        rate_locked_until=_str(data.get("rate_locked_until")),
        included_services=[s for s in (_str(x) for x in services) if s] if isinstance(services, list) else [],
        late_fee_cap=_num(data.get("late_fee_cap")),
        dispute_window_days=int(window) if isinstance(window, (int, float)) and not isinstance(window, bool) else None,
        other=[s for s in (_str(x) for x in other) if s] if isinstance(other, list) else [],
    )


# --------------------------------------------------------------------------- Reader


@dataclass
class ReadResult:
    kind: str
    provider: str
    bill: Bill | None
    contract: ContractFacts | None
    text: str


def read_document(ctx: AgentContext, text: str | None = None, image: bytes | None = None, mime: str = "") -> ReadResult:
    if image is not None:
        if ctx.llm.mode != "live":
            raise LLMError(
                "Reading photos needs live mode (Nemotron vision on Nebius). Upload a text or PDF bill instead."
            )
        data_url = f"data:{mime or 'image/png'};base64,{base64.b64encode(image).decode()}"
        request = LLMRequest(
            task="reader",
            role="vision",
            agent="Reader",
            max_tokens=3000,
            messages=[
                {"role": "system", "content": prompts.READER + prompts.READER_IMAGE_SUFFIX},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this document."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
    else:
        assert text is not None
        request = LLMRequest(
            task="reader",
            role="nano",
            agent="Reader",
            max_tokens=3000,
            payload={"text": text},
            messages=[{"role": "system", "content": prompts.READER}, {"role": "user", "content": text}],
        )
    data = ctx.call_json(request)
    if not isinstance(data, dict):
        raise LLMError("Reader returned an unexpected format")
    provider = _str(data.get("provider")) or "Unknown provider"
    kind = str(data.get("kind")) if data.get("kind") in ("bill", "contract") else "unknown"
    transcript = text if text is not None else str(data.get("transcript") or "")
    bill = coerce_bill(provider, data.get("bill") or {}) if kind == "bill" else None
    contract = coerce_contract(provider, data.get("contract") or {}) if kind == "contract" else None
    if kind == "bill" and bill is None:
        kind = "unknown"
    return ReadResult(kind=kind, provider=provider, bill=bill, contract=contract, text=transcript)


# --------------------------------------------------------------------------- AI Auditor


def ai_audit(ctx: AgentContext, bill: Bill, facts: ContractFacts | None, existing: list[Finding]) -> list[Finding]:
    payload = {
        "bill": bill.to_dict(),
        "agreement": facts.to_dict() if facts else None,
        "already_flagged": [{"type": f.type, "line_indexes": f.line_indexes, "title": f.title} for f in existing],
    }
    data = ctx.call_json(
        LLMRequest(
            task="ai_auditor",
            role="nano",
            agent="AI Auditor",
            payload=payload,
            messages=[{"role": "system", "content": prompts.AUDITOR}, {"role": "user", "content": as_json(payload)}],
        )
    )
    raw_findings = data.get("findings") if isinstance(data, dict) else None
    findings = []
    for n, raw in enumerate(raw_findings or []):
        if not isinstance(raw, dict):
            continue
        lines = [
            int(i) for i in raw.get("line_indexes") or [] if isinstance(i, (int, float)) and not isinstance(i, bool)
        ]
        amount = _num(raw.get("claimed_amount")) or 0.0
        type_ = str(raw.get("type")) if raw.get("type") in FINDING_TYPES else "other"
        findings.append(
            Finding(
                id=f"ai-{n}",
                type=type_ if type_ != "duplicate" or len(lines) > 1 else "other",
                title=_str(raw.get("title")) or "Possible billing problem",
                line_indexes=lines,
                claimed_amount=amount,
                rationale=_str(raw.get("rationale")) or "",
                source="llm",
            )
        )
    return findings[:6]


# --------------------------------------------------------------------------- Skeptic


def cross_examine(
    ctx: AgentContext, finding: Finding, bill: Bill, facts: ContractFacts | None
) -> tuple[str, str, float]:
    payload = {
        "accusation": {
            "type": finding.type,
            "title": finding.title,
            "rationale": finding.rationale,
            "amount": finding.amount,
            "verifier_note": finding.verification_note,
        },
        "cited_lines": [bill.line_items[i].to_dict() for i in finding.line_indexes if 0 <= i < len(bill.line_items)],
        "statement": {
            "provider": bill.provider,
            "statement_date": bill.statement_date,
            "all_lines": [li.to_dict() for li in bill.line_items],
        },
        "agreement": facts.to_dict() if facts else None,
    }
    data = ctx.call_json(
        LLMRequest(
            task="skeptic",
            role="super",
            agent="Skeptic",
            max_tokens=2000,
            payload=payload,
            messages=[{"role": "system", "content": prompts.SKEPTIC}, {"role": "user", "content": as_json(payload)}],
        )
    )
    if not isinstance(data, dict):
        raise LLMError("Skeptic returned an unexpected format")
    verdict = str(data.get("verdict", "")).lower()
    if verdict not in ("upheld", "dismissed"):
        verdict = "dismissed"  # when in doubt, don't accuse
    conf = data.get("confidence")
    confidence = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.5
    reason = _str(data.get("reason")) or "No reason given."
    return verdict, reason, max(0.0, min(1.0, confidence))


# --------------------------------------------------------------------------- Advocate


def draft_letter(
    ctx: AgentContext, bill: Bill, facts: ContractFacts | None, findings: list[Finding], fallback: str
) -> str:
    total = round(sum(f.amount for f in findings), 2)
    payload = {
        "customer": bill.customer,
        "provider": bill.provider,
        "account": bill.account,
        "statement_date": bill.statement_date,
        "findings": [
            {
                "title": f.title,
                "amount": f.amount,
                "explanation": f.rationale,
                "sources": [s.get("url") for s in f.sources],
            }
            for f in findings
        ],
        "total_credit_requested": f"${total:,.2f}",
        "dispute_window_days": facts.dispute_window_days if facts else None,
        "fallback_letter": fallback,
    }
    result = ctx.call(
        LLMRequest(
            task="advocate",
            role="super",
            agent="Advocate",
            max_tokens=2500,
            temperature=0.3,
            payload=payload,
            messages=[
                {"role": "system", "content": prompts.ADVOCATE},
                {"role": "user", "content": as_json({k: v for k, v in payload.items() if k != "fallback_letter"})},
            ],
        )
    )

    letter = strip_reasoning(result.text)
    # Guardrail: the letter must state the exact amount the Verifier confirmed.
    if f"{total:,.2f}" not in letter or len(letter) < 200:
        return fallback
    return letter


# --------------------------------------------------------------------------- Ask the Vault


def answer_question(ctx: AgentContext, question: str, facts: list[dict[str, Any]]) -> str:
    payload = {
        "question": question,
        "facts": [{"provider": f["provider"], "fact": f["key"], "value": f["value"]} for f in facts],
    }
    result = ctx.call(
        LLMRequest(
            task="ask",
            role="nano",
            agent="Ask Clerk",
            max_tokens=800,
            payload=payload,
            messages=[{"role": "system", "content": prompts.ASK}, {"role": "user", "content": as_json(payload)}],
        )
    )

    return strip_reasoning(result.text)
