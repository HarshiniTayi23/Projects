"""Test doubles: a scripted LLM and a toy tool provider."""
from __future__ import annotations

import json
import uuid
from typing import Any

from agent.llm_client import LLMResponse, ToolCall


def tool_call(name: str, args: dict | str) -> ToolCall:
    raw = args if isinstance(args, str) else json.dumps(args)
    return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=raw)


class ScriptedLLM:
    """Returns pre-programmed responses in order and records what it was sent."""

    def __init__(self, *responses: LLMResponse | Exception):
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class ToyTools:
    """A single get_time tool, used to prove the loop before the real tools exist."""

    def __init__(self):
        self.invocations: list[tuple[str, dict]] = []

    def schemas(self):
        return [{
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current local time in a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }]

    def call(self, name, arguments):
        self.invocations.append((name, arguments))
        if "city" not in arguments:
            return json.dumps({"ok": False, "error": "MISSING_FIELD", "message": "city is required"})
        return json.dumps({"ok": True, "city": arguments["city"], "time": "19:30"})
