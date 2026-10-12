"""Small text-matching helpers shared by detectors, the verifier and the skeptic."""

from __future__ import annotations

import re

_WORDS = re.compile(r"[a-z]+")
_STOP = {"the", "a", "an", "of", "for", "and", "per", "monthly", "month", "fee", "charge", "service", "plan"}


def norm(text: str) -> str:
    return " ".join(_WORDS.findall(text.lower()))


def norm_provider(name: str | None) -> str:
    return "".join(_WORDS.findall((name or "").lower()))


def tokens(text: str) -> set[str]:
    return {w for w in _WORDS.findall(text.lower()) if w not in _STOP and len(w) > 1}


def similarity(a: str, b: str) -> float:
    """Token overlap relative to the smaller set (1.0 = one contains the other)."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def is_tax_line(description: str) -> bool:
    return bool(re.search(r"\b(tax|vat|gst|surcharge for tax)\b", description, re.IGNORECASE))


def is_late_fee(description: str) -> bool:
    return bool(re.search(r"\blate\b.*\bfee\b|\blate payment\b", description, re.IGNORECASE))


def mentions_plan(description: str, plan_name: str | None) -> bool:
    if not plan_name:
        return False
    return similarity(description, plan_name) >= 0.99
