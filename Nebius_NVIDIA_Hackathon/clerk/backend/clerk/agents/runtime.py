"""Shared runtime for agents: budget checks, tracing and JSON calls."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..db import Database
from ..llm.base import LLMError, LLMProvider, LLMRequest, LLMResult
from ..llm.jsonutil import JSONExtractionError, extract_json
from ..search import SearchClient

Emit = Callable[[dict[str, Any]], None]

SKILLS: dict[str, str] = {
    "remember_documents": "Remember facts from your agreements in the Vault",
    "ai_auditor": "Let Nemotron look for problems the rule-based auditors can't catch",
    "web_research": "Search the web (Tavily) for the rules behind each finding",
    "draft_letters": "Draft dispute letters and phone-call scripts for your approval",
    "reminders": "Prepare calendar reminders for dispute deadlines",
}


class BudgetExceeded(LLMError):
    pass


def _noop(_: dict[str, Any]) -> None:
    return None


@dataclass
class AgentContext:
    settings: Settings
    llm: LLMProvider
    search: SearchClient
    db: Database
    workspace: str
    audit_id: str | None = None
    emit: Emit = _noop
    permissions: dict[str, bool] = field(default_factory=dict)

    def allowed(self, skill: str) -> bool:
        return self.permissions.get(skill, True)

    def call(self, request: LLMRequest) -> LLMResult:
        if self.llm.mode == "live" and self.db.count_usage_today("llm_call") >= self.settings.daily_llm_call_limit:
            raise BudgetExceeded("Clerk has reached today's model-call budget. Please try again tomorrow.")
        result = self.llm.complete(request)
        self.db.record_usage(self.workspace, "llm_call")
        cost = None
        if result.mode == "live":
            cost = self.settings.price_for(result.model).cost(result.prompt_tokens, result.completion_tokens)
        step = {
            "agent": request.agent or request.task,
            "task": request.task,
            "model": result.model,
            "mode": result.mode,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
            "cost_usd": cost,
        }
        self.db.add_trace(self.workspace, self.audit_id, step)
        self.emit({"type": "trace", "step": step})
        return result

    def call_json(self, request: LLMRequest) -> Any:
        result = self.call(request)
        try:
            return extract_json(result.text)
        except JSONExtractionError:
            retry = LLMRequest(
                task=request.task,
                role=request.role,
                agent=request.agent,
                payload=request.payload,
                max_tokens=request.max_tokens,
                temperature=0.0,
                messages=[
                    *request.messages,
                    {"role": "assistant", "content": result.text[:4000]},
                    {
                        "role": "user",
                        "content": "That reply was not valid JSON. Reply again with ONLY the JSON object.",
                    },
                ],
            )
            try:
                return extract_json(self.call(retry).text)
            except JSONExtractionError as exc:
                raise LLMError(f"{request.task}: model did not return valid JSON") from exc


def as_json(data: Any) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False)
