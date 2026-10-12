"""Provider interface shared by the live Nebius client and the offline simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

ROLES = ("super", "nano", "vision")


@dataclass
class LLMRequest:
    task: str  # e.g. "reader", "skeptic"; the offline simulator routes on this
    role: str  # "super" | "nano" | "vision"
    messages: list[dict[str, Any]]
    payload: dict[str, Any] = field(default_factory=dict)  # structured inputs (used by the simulator)
    max_tokens: int = 1500
    temperature: float = 0.1
    agent: str | None = None  # display name for the trace panel

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")


@dataclass
class LLMResult:
    text: str
    model: str
    mode: str  # "live" | "offline"
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class LLMError(RuntimeError):
    """The provider failed after retries, or the budget is exhausted."""


class LLMProvider(Protocol):
    mode: str

    def complete(self, request: LLMRequest) -> LLMResult: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def model_for(self, role: str) -> str: ...
