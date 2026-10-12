"""Web research for the Researcher agent: Tavily when live, nothing when offline.

The offline client deliberately returns *no* sources rather than invented ones:
citations must always be real links.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..config import Settings


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "score": self.score}


class SearchError(RuntimeError):
    pass


class SearchClient(Protocol):
    mode: str

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]: ...


class TavilyClient:
    mode = "live"
    url = "https://api.tavily.com/search"

    def __init__(self, api_key: str, client: httpx.Client | None = None, timeout: float = 20.0):
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=timeout)

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        try:
            response = self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                },
            )
        except httpx.TransportError as exc:
            raise SearchError(f"Tavily request failed: {exc}") from exc
        if response.status_code >= 400:
            raise SearchError(f"Tavily returned {response.status_code}: {response.text[:200]}")
        results = response.json().get("results") or []
        return [
            SearchResult(
                title=str(r.get("title") or r.get("url") or ""),
                url=str(r.get("url") or ""),
                snippet=str(r.get("content") or "")[:400],
                score=r.get("score"),
            )
            for r in results
            if r.get("url")
        ][:max_results]


class OfflineSearch:
    mode = "offline"

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        return []


def build_search(settings: Settings) -> SearchClient:
    if settings.live and settings.tavily_api_key:
        return TavilyClient(settings.tavily_api_key)
    return OfflineSearch()
