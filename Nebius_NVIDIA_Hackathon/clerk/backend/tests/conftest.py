from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from clerk.agents.runtime import AgentContext
from clerk.api import create_app
from clerk.config import Settings, load_settings
from clerk.db import Database
from clerk.llm.base import LLMRequest, LLMResult
from clerk.llm.offline import OfflineProvider, hashed_embedding
from clerk.search import OfflineSearch

DEMO = Path(__file__).resolve().parents[1] / "clerk" / "demo"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for var in ("CLERK_MODE", "NEBIUS_API_KEY", "TAVILY_API_KEY", "CLERK_MODEL_SUPER", "CLERK_MODEL_NANO"):
        monkeypatch.delenv(var, raising=False)
    return replace(load_settings(), db_path=Path(":memory:"))


@pytest.fixture
def db() -> Database:
    return Database(":memory:")


@pytest.fixture
def ctx(settings: Settings, db: Database) -> AgentContext:
    return AgentContext(settings=settings, llm=OfflineProvider(), search=OfflineSearch(), db=db, workspace="test")


@pytest.fixture
def client(settings: Settings, db: Database) -> Iterator[TestClient]:
    with TestClient(create_app(settings, db=db)) as c:
        yield c


def read_events(client: TestClient, doc_id: str) -> list[dict[str, Any]]:
    events = []
    with client.stream("GET", f"/api/documents/{doc_id}/audit/stream") as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
    return events


def demo_text(name: str) -> str:
    return (DEMO / name).read_text()


class FakeProvider:
    """Scripted 'live' provider: returns queued responses per task, records every request."""

    mode = "live"

    def __init__(self, responses: dict[str, list[str] | Callable[[LLMRequest], str]]):
        self.responses = responses
        self.requests: list[LLMRequest] = []

    def model_for(self, role: str) -> str:
        return f"fake/{role}"

    def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        source = self.responses.get(request.task)
        if source is None:
            raise AssertionError(f"unexpected task {request.task}")
        text = source(request) if callable(source) else source.pop(0)
        return LLMResult(text, self.model_for(request.role), "live", 100, 20, 5)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [hashed_embedding(t) for t in texts]
