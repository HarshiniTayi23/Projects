"""Lenient parsing of dates, times, phones and fuzzy names.

Small models often pass "tomorrow", "8pm" or "Indira nagar" instead of the
exact format asked for. Normalising here (instead of failing) makes the agent
far more robust without extra LLM round-trips.
"""
from __future__ import annotations

import difflib
import re
from datetime import date, datetime, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d/%m/%y", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"]
NO_YEAR_FORMATS = ["%d %B", "%d %b", "%B %d", "%b %d", "%d/%m", "%d-%m"]


class ParseError(ValueError):
    pass


def parse_date(value: str, today: date) -> date:
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", str(value).strip().lower()).replace(",", " ")
    text = re.sub(r"\s+", " ", text)
    if text in ("today", "tonight", "this evening"):
        return today
    if text in ("tomorrow", "tmrw", "tomorrow night"):
        return today + timedelta(days=1)
    if text == "day after tomorrow":
        return today + timedelta(days=2)

    m = re.fullmatch(r"(this |next |coming )?(" + "|".join(WEEKDAYS) + r")", text)
    if m:
        delta = (WEEKDAYS.index(m.group(2)) - today.weekday()) % 7
        if m.group(1) == "next " and delta == 0:
            delta = 7
        return today + timedelta(days=delta)

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    for fmt in NO_YEAR_FORMATS:
        try:
            d = datetime.strptime(f"{text} {today.year}", f"{fmt} %Y").date()
            return d if d >= today else d.replace(year=today.year + 1)
        except ValueError:
            pass
    raise ParseError(f"Could not understand the date '{value}'. Use YYYY-MM-DD, e.g. {today.isoformat()}.")


_TIME = re.compile(r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?")


def parse_time(value: str) -> str:
    """Return HH:MM (24h). Bare hours 1-6 are read as PM (dinner context)."""
    text = str(value).strip().lower().replace(" ", "")
    if text in ("noon", "midday", "lunch"):
        return "12:30" if text == "lunch" else "12:00"
    if text in ("dinner",):
        return "20:00"
    m = _TIME.fullmatch(text)
    if not m:
        raise ParseError(f"Could not understand the time '{value}'. Use 24-hour HH:MM, e.g. 19:30.")
    hour, minute, suffix = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "")
    if minute > 59 or hour > 23:
        raise ParseError(f"'{value}' is not a valid time.")
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    elif not suffix and 1 <= hour <= 6:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def add_minutes(hhmm: str, minutes: int) -> str:
    """Add minutes within the same day (clamped to 00:00-23:59)."""
    h, m = map(int, hhmm.split(":"))
    total = max(0, min(h * 60 + m + minutes, 23 * 60 + 59))
    return f"{total // 60:02d}:{total % 60:02d}"


def normalize_phone(value: str) -> str:
    digits = re.sub(r"[^\d+]", "", str(value))
    plain = digits.lstrip("+")
    if len(plain) == 10 and plain[0] in "6789":
        return "+91" + plain
    if len(plain) == 12 and plain.startswith("91") and plain[2] in "6789":
        return "+" + plain
    if digits.startswith("+") and 8 <= len(plain) <= 15:
        return digits
    raise ParseError("Please provide a valid 10-digit mobile number (e.g. 98765 43210).")


def normalize_name(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value)).strip()
    if not (2 <= len(name) <= 60) or not re.fullmatch(r"[A-Za-z][A-Za-z .'\-]*", name):
        raise ParseError("Please provide the guest's name (letters only, 2-60 characters).")
    return name.title()


def fuzzy_pick(value: str, options: list[str], cutoff: float = 0.72) -> str | None:
    """Case/spacing-insensitive best match, e.g. 'indira nagar' -> 'Indiranagar'."""
    if not value:
        return None
    squash = lambda s: re.sub(r"[^a-z]", "", s.lower())
    lookup = {squash(o): o for o in options}
    key = squash(value)
    if key in lookup:
        return lookup[key]
    for k, original in lookup.items():
        if key and (key in k or k in key) and len(key) >= 4:
            return original
    close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=cutoff)
    return lookup[close[0]] if close else None
