"""Templates for the Advocate's outputs: dispute letter, call script, calendar reminder."""

from __future__ import annotations

import datetime as dt
import re
import uuid

from .schemas import Bill, ContractFacts, Finding

DEFAULT_RESPONSE_DAYS = 30


def money(x: float) -> str:
    return f"${x:,.2f}"


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def dispute_deadline(bill: Bill, facts: ContractFacts | None, today: dt.date | None = None) -> dt.date:
    """The date by which the dispute should be raised.

    Uses the agreement's dispute window when we have one; otherwise suggests
    responding within 30 days of the statement (labelled as a suggestion in the UI).
    """
    start = _parse_date(bill.statement_date) or today or dt.date.today()
    days = facts.dispute_window_days if facts and facts.dispute_window_days else DEFAULT_RESPONSE_DAYS
    return start + dt.timedelta(days=days)


def template_letter(bill: Bill, findings: list[Finding], today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    total = round(sum(f.amount for f in findings), 2)
    lines = [
        today.strftime("%B %d, %Y").replace(" 0", " "),
        "",
        bill.provider,
        "Billing Disputes",
        f"Re: Account {bill.account or '[account number]'}, statement dated {bill.statement_date or '[date]'}",
        "",
        "To whom it may concern,",
        "",
        f"I am writing to dispute {len(findings)} charge{'s' if len(findings) != 1 else ''} on the statement above. "
        "After comparing it with my agreement and checking the arithmetic, I found the following errors:",
        "",
    ]
    for n, f in enumerate(findings, 1):
        lines.append(f"{n}. {f.title}: {money(f.amount)}")
        lines.append(f"   {f.rationale}")
        lines.append("")
    lines += [
        f"Please correct these charges, credit {money(total)} to my account, and confirm the correction in writing.",
        "",
    ]
    sources = [s["url"] for f in findings for s in f.sources if s.get("url")]
    if sources:
        lines += ["For reference:"] + [f"- {url}" for url in dict.fromkeys(sources)] + [""]
    lines += [
        "I have kept copies of the statement, my agreement and this letter.",
        "",
        "Sincerely,",
        bill.customer or "[Your name]",
    ]
    return "\n".join(lines)


def call_script(bill: Bill, findings: list[Finding]) -> str:
    total = round(sum(f.amount for f in findings), 2)
    out = [
        f"PHONE SCRIPT: {bill.provider} billing",
        "",
        f"Have ready: account {bill.account or '[account number]'}, statement dated {bill.statement_date or '[date]'}, your agreement.",
        "",
        "OPENING",
        f"\"Hi, I'm calling about my statement dated {bill.statement_date or '[date]'}. "
        f"I found {len(findings)} billing error{'s' if len(findings) != 1 else ''} totalling {money(total)} and I'd like them corrected.\"",
        "",
        "THE ERRORS",
    ]
    for n, f in enumerate(findings, 1):
        out.append(f'{n}. "{f.title}. That\'s {money(f.amount)}. {f.rationale}"')
    out += [
        "",
        "IF THEY PUSH BACK",
        '- "Can you show me where my agreement allows this charge?"',
        '- "I\'d like this escalated to a supervisor, please."',
        '- "Please note on my account that I disputed this today, and give me a reference number."',
        "",
        "BEFORE YOU HANG UP",
        "- Write down: the agent's name, the date and time, the reference number, and what they promised.",
        f"- Confirm the credit amount: {money(total)}.",
    ]
    return "\n".join(out)


def _ics_escape(text: str) -> str:
    return re.sub(r"([,;\\])", r"\\\1", text).replace("\n", "\\n")


def reminder_ics(title: str, description: str, when: dt.date, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.UTC)
    end = when + dt.timedelta(days=1)
    return "\r\n".join(
        [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Clerk//Bill Auditor//EN",
            "CALSCALE:GREGORIAN",
            "BEGIN:VEVENT",
            f"UID:{uuid.uuid4()}@clerk",
            f"DTSTAMP:{now.strftime('%Y%m%dT%H%M%SZ')}",
            f"DTSTART;VALUE=DATE:{when.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{end.strftime('%Y%m%d')}",
            f"SUMMARY:{_ics_escape(title)}",
            f"DESCRIPTION:{_ics_escape(description)}",
            "BEGIN:VALARM",
            "TRIGGER:-P3D",
            "ACTION:DISPLAY",
            f"DESCRIPTION:{_ics_escape(title)}",
            "END:VALARM",
            "END:VEVENT",
            "END:VCALENDAR",
            "",
        ]
    )


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "document"
