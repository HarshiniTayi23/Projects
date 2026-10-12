"""Offline provider: a deterministic simulator that stands in for Nemotron.

It exists so that the full pipeline, the UI, the test suite and the eval
harness run with no network and no API keys. Each agent task has a simulator
(see ``clerk.agents.simulators``) that returns text in the *same format* the
real model is prompted to produce, so the agents' parsing code is exercised
end to end. Results are labelled ``mode="offline"`` everywhere they surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable

from .base import LLMError, LLMRequest, LLMResult

Simulator = Callable[[LLMRequest], str]
SIMULATORS: dict[str, Simulator] = {}

OFFLINE_MODELS = {
    "super": "offline-sim/nemotron-super-standin",
    "nano": "offline-sim/nemotron-nano-standin",
    "vision": "offline-sim/nemotron-vision-standin",
}

EMBED_DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


def register(task: str) -> Callable[[Simulator], Simulator]:
    def decorator(fn: Simulator) -> Simulator:
        SIMULATORS[task] = fn
        return fn

    return decorator


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def hashed_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Feature-hashed bag of words. Good enough for keyword-ish recall offline."""
    vec = [0.0] * dim
    for token in _TOKEN.findall(text.lower()):
        digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class OfflineProvider:
    mode = "offline"

    def __init__(self) -> None:
        # Importing registers every simulator.
        from ..agents import simulators  # noqa: F401

    def model_for(self, role: str) -> str:
        return OFFLINE_MODELS[role]

    def complete(self, request: LLMRequest) -> LLMResult:
        simulator = SIMULATORS.get(request.task)
        if simulator is None:
            raise LLMError(f"no offline simulator for task {request.task!r}")
        started = time.perf_counter()
        text = simulator(request)
        prompt_text = json.dumps(request.messages)
        return LLMResult(
            text=text,
            model=self.model_for(request.role),
            mode=self.mode,
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens(text),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [hashed_embedding(t) for t in texts]
