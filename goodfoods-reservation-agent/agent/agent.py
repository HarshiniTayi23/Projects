"""The tool-calling agent loop, written from scratch.

Flow per user turn:
    1. append user message
    2. send history + tool schemas to the LLM
    3. if the LLM asks for tools -> execute each, append results, go to 2
    4. if the LLM answers in text -> return it
The LLM alone decides which tool (if any) matches the user's intent.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agent.config import Settings, settings as default_settings
from agent.llm_client import LLMConfigError, LLMError, LLMResponse

log = logging.getLogger(__name__)


class ToolProvider(Protocol):
    def schemas(self) -> list[dict[str, Any]]: ...
    def call(self, name: str, arguments: dict[str, Any]) -> str: ...


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any] | str
    result: str
    ok: bool
    duration_s: float


@dataclass
class AgentTurn:
    reply: str
    tool_events: list[ToolEvent] = field(default_factory=list)
    error: str | None = None


FALLBACK_LOOP = (
    "Sorry, I got a bit tangled up working that out. Could you tell me again what you'd like, "
    "for example the restaurant, date, time and number of guests?"
)
FALLBACK_LLM_DOWN = (
    "I'm having trouble reaching my language service right now. Please try again in a moment."
)


class Agent:
    def __init__(
        self,
        llm: ChatModel,
        tools: ToolProvider,
        system_prompt: Callable[[], str] | str,
        cfg: Settings = default_settings,
    ):
        self.llm = llm
        self.tools = tools
        self._system_prompt = system_prompt
        self.cfg = cfg
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ public
    def chat(self, user_text: str) -> AgentTurn:
        user_text = (user_text or "").strip()
        if not user_text:
            return AgentTurn(reply="Could you type your request? I can help you find and book a table.")

        self.history.append({"role": "user", "content": user_text[:2000]})
        events: list[ToolEvent] = []
        schemas = self.tools.schemas()

        for step in range(self.cfg.max_tool_iterations):
            try:
                response = self.llm.chat(self._build_messages(), tools=schemas)
            except LLMConfigError as exc:
                return self._finish(str(exc), events, error=str(exc))
            except LLMError as exc:
                log.error("LLM failure: %s", exc)
                return self._finish(FALLBACK_LLM_DOWN, events, error=str(exc))

            if not response.tool_calls:
                reply = (response.content or "").strip() or "Is there anything else I can help you with?"
                return self._finish(reply, events)

            self.history.append(response.to_message())
            for call in response.tool_calls:
                events.append(self._execute(call.id, call.name, call.arguments))
            log.info("step %d: %d tool call(s)", step + 1, len(response.tool_calls))

        return self._finish(FALLBACK_LOOP, events, error="max_tool_iterations reached")

    def reset(self) -> None:
        self.history.clear()

    # ------------------------------------------------------------------ helpers
    def _execute(self, call_id: str, name: str, raw_args: str) -> ToolEvent:
        start = time.perf_counter()
        known = {s["function"]["name"] for s in self.tools.schemas()}
        args: dict[str, Any] | str = raw_args

        if name not in known:
            result = _error_json("UNKNOWN_TOOL", f"No tool named '{name}'. Available: {sorted(known)}")
        else:
            try:
                parsed = json.loads(raw_args) if raw_args else {}
                if not isinstance(parsed, dict):
                    raise ValueError("arguments must be a JSON object")
                args = {k: v for k, v in parsed.items() if v is not None}
                result = self.tools.call(name, args)
            except (json.JSONDecodeError, ValueError) as exc:
                result = _error_json("INVALID_ARGUMENTS", f"Could not parse arguments: {exc}. Send valid JSON.")
            except Exception as exc:  # tool crashed: tell the model, never the user's screen
                log.exception("Tool %s crashed", name)
                result = _error_json("TOOL_FAILURE", f"The {name} tool failed: {exc}")

        self.history.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": result})
        ok = not _is_error(result)
        return ToolEvent(name, args, result, ok, time.perf_counter() - start)

    def _finish(self, reply: str, events: list[ToolEvent], error: str | None = None) -> AgentTurn:
        self.history.append({"role": "assistant", "content": reply})
        return AgentTurn(reply=reply, tool_events=events, error=error)

    def _build_messages(self) -> list[dict[str, Any]]:
        prompt = self._system_prompt() if callable(self._system_prompt) else self._system_prompt
        return [{"role": "system", "content": prompt}, *self._trimmed_history()]

    def _trimmed_history(self) -> list[dict[str, Any]]:
        """Keep the last N user turns; always cut at a user message so tool
        call/result pairs are never split."""
        user_idx = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(user_idx) <= self.cfg.history_turns:
            return list(self.history)
        return self.history[user_idx[-self.cfg.history_turns]:]


def _error_json(code: str, message: str) -> str:
    return json.dumps({"ok": False, "error": code, "message": message})


def _is_error(result: str) -> bool:
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, dict) and data.get("ok") is False
