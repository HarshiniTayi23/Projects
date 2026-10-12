"""Robust JSON extraction from model output.

Reasoning models may wrap answers in <think> blocks or markdown fences, or add a
sentence before the JSON. We strip all of that and parse the first complete
JSON object or array.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JSONExtractionError(ValueError):
    pass


def strip_reasoning(text: str) -> str:
    text = _THINK.sub("", text)
    # An unterminated <think> (truncated output) - drop everything before the last '</think>' or the tag itself.
    if "<think>" in text.lower():
        idx = text.lower().rfind("</think>")
        text = text[idx + len("</think>") :] if idx >= 0 else re.sub(r"(?is)<think>.*", "", text)
    return text.strip()


def _first_balanced(text: str) -> str | None:
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
            if not stack:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Any:
    cleaned = strip_reasoning(text)
    candidates = [m.group(1).strip() for m in _FENCE.finditer(cleaned)] + [cleaned]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        block = _first_balanced(candidate)
        if block is not None:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                # Trailing commas are the most common model mistake.
                try:
                    return json.loads(re.sub(r",\s*([}\]])", r"\1", block))
                except json.JSONDecodeError:
                    continue
    raise JSONExtractionError(f"no valid JSON found in model output: {text[:200]!r}")
