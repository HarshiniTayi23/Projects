"""Live provider: NVIDIA Nemotron models served by Nebius Token Factory.

Token Factory exposes an OpenAI-compatible API, so this is a thin httpx client
with retries. No SDK dependency keeps the container small.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..config import Settings
from .base import LLMError, LLMRequest, LLMResult

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class NebiusProvider:
    mode = "live"

    def __init__(self, settings: Settings, client: httpx.Client | None = None, max_attempts: int = 3):
        if not settings.nebius_api_key:
            raise LLMError("NEBIUS_API_KEY is not set")
        self.settings = settings
        self.max_attempts = max_attempts
        self.client = client or httpx.Client(
            base_url=settings.nebius_base_url,
            timeout=settings.llm_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.nebius_api_key}"},
        )

    def model_for(self, role: str) -> str:
        model = {
            "super": self.settings.model_super,
            "nano": self.settings.model_nano,
            # Fall back to the Super model for vision if no dedicated model is configured.
            "vision": self.settings.model_vision or self.settings.model_super,
        }[role]
        if not model:
            raise LLMError(f"no model configured for role {role!r}")
        return model

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self.client.post(path, json=body)
            except httpx.TransportError as exc:  # timeouts, connection resets
                last_error = exc
            else:
                if response.status_code < 400:
                    data: dict[str, Any] = response.json()
                    return data
                if response.status_code not in RETRY_STATUS:
                    raise LLMError(f"Token Factory returned {response.status_code}: {response.text[:300]}")
                last_error = LLMError(f"Token Factory returned {response.status_code}")
            if attempt < self.max_attempts - 1:
                time.sleep(min(2**attempt, 8))
        raise LLMError(f"Token Factory request failed after {self.max_attempts} attempts: {last_error}")

    def complete(self, request: LLMRequest) -> LLMResult:
        model = self.model_for(request.role)
        started = time.perf_counter()
        data = self._post(
            "chat/completions",
            {
                "model": model,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected Token Factory response shape: {str(data)[:300]}") from exc
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            model=model,
            mode=self.mode,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.model_embed:
            from .offline import hashed_embedding

            return [hashed_embedding(t) for t in texts]
        data = self._post("embeddings", {"model": self.settings.model_embed, "input": texts})
        return [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
