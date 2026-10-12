"""The Tribunal: every accusation is proposed, verified, cross-examined, researched, then acted on.

    Auditors (code + Nemotron Nano)  ->  Verifier (code)  ->  Skeptic (Nemotron Super)
        ->  Researcher (Tavily)  ->  Advocate (Nemotron Super + templates)

Each step emits events, which the API streams to the browser as Server-Sent Events.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from .. import paperwork, vault
from ..llm.base import LLMError
from ..schemas import Bill, ContractFacts, Finding
from .detectors import run_detectors
from .llm_agents import ai_audit, cross_examine, draft_letter
from .researcher import research
from .runtime import AgentContext, BudgetExceeded
from .verifier import verify

MAX_FINDINGS = 12


def _stage(ctx: AgentContext, name: str, status: str, detail: str = "") -> None:
    ctx.emit({"type": "stage", "stage": name, "status": status, "detail": detail})


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    unique = []
    for f in findings:
        key = (f.type, tuple(sorted(f.line_indexes)), str(f.meta.get("check", "")))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def run_tribunal(ctx: AgentContext, bill: Bill, today: dt.date | None = None) -> dict[str, Any]:
    """Run the full audit. Returns the summary that is stored with the audit."""
    _stage(ctx, "reader", "done", f"{len(bill.line_items)} line items read from {bill.provider}")
    ctx.emit({"type": "bill", "bill": bill.to_dict()})

    # 1. Memory
    _stage(ctx, "vault", "start")
    facts: ContractFacts | None = vault.find_contract(ctx, bill)
    ctx.emit({"type": "facts", "facts": facts.to_dict() if facts else None})
    _stage(
        ctx,
        "vault",
        "done",
        f"Found your {facts.provider} agreement"
        if facts
        else "No agreement on file: rate, fee and service checks are limited",
    )

    # 2. Auditors
    _stage(ctx, "auditors", "start")
    candidates = run_detectors(bill, facts)
    for f in candidates:
        ctx.emit({"type": "candidate", "finding": f.to_dict()})
    if ctx.allowed("ai_auditor"):
        try:
            extra = ai_audit(ctx, bill, facts, candidates)
        except LLMError as exc:
            extra = []
            ctx.emit({"type": "notice", "message": f"AI Auditor skipped: {exc}"})
        for f in extra:
            ctx.emit({"type": "candidate", "finding": f.to_dict()})
        candidates += extra
    candidates = _dedupe(candidates)[:MAX_FINDINGS]
    _stage(ctx, "auditors", "done", f"{len(candidates)} possible problem{'s' if len(candidates) != 1 else ''} raised")

    # 3. Verifier
    _stage(ctx, "verifier", "start")
    for f in candidates:
        v = verify(f, bill, facts)
        f.verified, f.verified_amount, f.verification_note = v.ok, v.amount, v.note
        if not v.ok:
            f.verdict, f.verdict_reason, f.confidence = "dismissed", f"Failed verification: {v.note}", 1.0
        ctx.emit({"type": "verified", "finding": f.to_dict()})
    _stage(ctx, "verifier", "done", f"{sum(1 for f in candidates if f.verified)} passed the arithmetic check")

    # 4. Skeptic
    _stage(ctx, "skeptic", "start")
    for f in candidates:
        if not f.verified:
            continue
        try:
            f.verdict, f.verdict_reason, f.confidence = cross_examine(ctx, f, bill, facts)
        except BudgetExceeded:
            raise
        except LLMError as exc:
            # Never accuse without cross-examination.
            f.verdict, f.verdict_reason, f.confidence = "dismissed", f"Could not be cross-examined ({exc}).", 0.0
        ctx.emit({"type": "verdict", "finding": f.to_dict()})
    upheld = [f for f in candidates if f.verdict == "upheld"]
    _stage(ctx, "skeptic", "done", f"{len(upheld)} upheld, {len(candidates) - len(upheld)} dismissed")

    # 5. Researcher
    if upheld and ctx.allowed("web_research"):
        _stage(ctx, "researcher", "start")
        for f in upheld:
            f.sources = research(ctx, f)
            ctx.emit({"type": "sources", "finding": f.to_dict()})
        detail = "Sources attached" if ctx.search.mode == "live" else "Live web research is off in offline mode"
        _stage(ctx, "researcher", "done", detail)
    else:
        _stage(ctx, "researcher", "skipped", "Nothing to research" if not upheld else "Web research is turned off")

    # 6. Advocate
    actions: list[dict[str, Any]] = []
    total = round(sum(f.amount for f in upheld), 2)
    if upheld:
        _stage(ctx, "advocate", "start")
        actions = _prepare_actions(ctx, bill, facts, upheld, total, today)
        for action in actions:
            ctx.emit({"type": "action", "action": action})
        _stage(
            ctx,
            "advocate",
            "done",
            f"{len(actions)} action{'s' if len(actions) != 1 else ''} waiting for your approval",
        )
    else:
        _stage(ctx, "advocate", "skipped", "No upheld findings, so nothing to dispute")

    summary = {
        "provider": bill.provider,
        "statement_date": bill.statement_date,
        "recoverable_total": total,
        "upheld": len(upheld),
        "dismissed": len(candidates) - len(upheld),
        "findings": [f.to_dict() for f in candidates],
        "facts_used": facts.to_dict() if facts else None,
        "action_ids": [a["id"] for a in actions],
    }
    ctx.emit({"type": "summary", "summary": summary})
    return summary


def _prepare_actions(
    ctx: AgentContext,
    bill: Bill,
    facts: ContractFacts | None,
    upheld: list[Finding],
    total: float,
    today: dt.date | None,
) -> list[dict[str, Any]]:
    created: list[str] = []
    base = paperwork.slug(f"{bill.provider} {bill.statement_date or ''}")
    if ctx.allowed("draft_letters"):
        fallback = paperwork.template_letter(bill, upheld, today)
        try:
            letter = draft_letter(ctx, bill, facts, upheld, fallback)
        except LLMError as exc:
            ctx.emit({"type": "notice", "message": f"Advocate used the template letter: {exc}"})
            letter = fallback
        created.append(
            ctx.db.add_action(
                ctx.workspace,
                ctx.audit_id,
                "letter",
                f"Dispute letter to {bill.provider} ({paperwork.money(total)})",
                letter,
                f"dispute-{base}.txt",
                "text/plain",
            )
        )
        created.append(
            ctx.db.add_action(
                ctx.workspace,
                ctx.audit_id,
                "call_script",
                f"Phone script for {bill.provider}",
                paperwork.call_script(bill, upheld),
                f"call-script-{base}.txt",
                "text/plain",
            )
        )
    if ctx.allowed("reminders"):
        deadline = paperwork.dispute_deadline(bill, facts, today)
        source = (
            "your agreement's dispute window" if facts and facts.dispute_window_days else "a suggested 30-day window"
        )
        ics = paperwork.reminder_ics(
            f"Dispute deadline: {bill.provider} ({paperwork.money(total)})",
            f"Clerk found {len(upheld)} billing error(s) on the {bill.statement_date} statement. "
            f"Deadline based on {source}. Verify the deadline in your agreement.",
            deadline,
        )
        created.append(
            ctx.db.add_action(
                ctx.workspace,
                ctx.audit_id,
                "reminder",
                f"Reminder: dispute {bill.provider} by {deadline.isoformat()}",
                ics,
                f"reminder-{base}.ics",
                "text/calendar",
            )
        )
    return [_public_action(a) for a in (ctx.db.get_action(ctx.workspace, i) for i in created) if a]


def _public_action(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ("id", "audit_id", "type", "status", "title", "content", "filename", "mime", "created")}
