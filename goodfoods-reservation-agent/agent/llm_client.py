"""Minimal client for any OpenAI-compatible /chat/completions endpoint.

Written directly on top of httpx (no SDK, no agent framework) so every part of
the request/response cycle is visible and under our control.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent.config import Settings, settings as default_settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM provider cannot produce a usable response."""


class LLMConfigError(LLMError):
    """Raised when the client is missing configuration (e.g. API key)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as produced by the model

    def to_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0

    def to_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_message() for tc in self.tool_calls]
        return msg


# Small models sometimes write the tool call as text instead of using the
# structured field. These patterns recover the most common shapes.
_FUNC_TAG = re.compile(r"<function=([\w\-]+)>?\s*(\{.*?\})\s*</function>", re.S)
_JSON_CALL = re.compile(r'\{\s*"(?:name|function)"\s*:\s*"([\w\-]+)"\s*,\s*"(?:parameters|arguments)"\s*:\s*(\{.*\})\s*\}', re.S)


def parse_text_tool_calls(text: str | None, known_tools: set[str]) -> list[ToolCall]:
    """Extract tool calls that a model emitted as plain text."""
    if not text:
        return []
    calls: list[ToolCall] = []
    for pattern in (_FUNC_TAG, _JSON_CALL):
        for name, args in pattern.findall(text):
            if name not in known_tools:
                continue
            try:
                json.loads(args)
            except json.JSONDecodeError:
                continue
            calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=args))
        if calls:
            break
    return calls


class LLMClient:
    def __init__(self, cfg: Settings = default_settings, http: httpx.Client | None = None):
        self.cfg = cfg
        self._http = http or httpx.Client(timeout=cfg.request_timeout)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        if not self.cfg.api_key:
            raise LLMConfigError("LLM_API_KEY is not set. Copy .env.example to .env and add your key.")

        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        known = {t["function"]["name"] for t in tools or []}
        data = self._post_with_retries(payload, known)
        return self._parse(data, known)

    # ------------------------------------------------------------------ internals
    def _post_with_retries(self, payload: dict[str, Any], known: set[str]) -> dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        last_err: Exception | None = None

        for attempt in range(self.cfg.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self._http.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:  # network error / timeout
                last_err = exc
                log.warning("LLM request failed (attempt %s): %s", attempt + 1, exc)
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    data["_latency"] = time.perf_counter() - start
                    return data
                body = _safe_json(resp)
                # Groq returns 400 tool_use_failed when the model writes a malformed
                # tool call. Try to salvage it; otherwise retry the generation.
                if resp.status_code == 400 and _error_code(body) == "tool_use_failed":
                    failed = (body.get("error") or {}).get("failed_generation", "")
                    recovered = parse_text_tool_calls(failed, known)
                    if recovered:
                        log.info("Recovered %d tool call(s) from failed generation", len(recovered))
                        return _synthetic_response(recovered, time.perf_counter() - start)
                    last_err = LLMError("Model produced a malformed tool call.")
                elif resp.status_code in (408, 409, 429) or resp.status_code >= 500:
                    last_err = LLMError(f"Provider returned {resp.status_code}: {_error_message(body)}")
                elif resp.status_code in (401, 403):
                    raise LLMConfigError("The LLM provider rejected the API key.")
                else:
                    raise LLMError(f"Provider returned {resp.status_code}: {_error_message(body)}")

            if attempt < self.cfg.max_retries:
                time.sleep(min(8.0, (2**attempt) * 0.5 + random.random() * 0.3))

        raise LLMError(f"LLM unavailable after {self.cfg.max_retries + 1} attempts: {last_err}")

    @staticmethod
    def _parse(data: dict[str, Any], known: set[str]) -> LLMResponse:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {exc}") from exc

        calls = [
            ToolCall(
                id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "{}",
            )
            for tc in msg.get("tool_calls") or []
        ]
        content = msg.get("content")
        if not calls:
            calls = parse_text_tool_calls(content, known)
            if calls:
                content = None
        return LLMResponse(content=content, tool_calls=calls, latency_s=data.get("_latency", 0.0))


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except ValueError:
        return {"error": {"message": resp.text[:300]}}


def _error_code(body: dict[str, Any]) -> str | None:
    err = body.get("error")
    return err.get("code") if isinstance(err, dict) else None


def _error_message(body: dict[str, Any]) -> str:
    err = body.get("error")
    return err.get("message", str(err)) if isinstance(err, dict) else str(body)[:300]


def _synthetic_response(calls: list[ToolCall], latency: float) -> dict[str, Any]:
    return {
        "_latency": latency,
        "choices": [{"message": {"content": None, "tool_calls": [c.to_message() for c in calls]}}],
    }
