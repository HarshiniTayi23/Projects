"""Contract tests: the live clients send exactly what Token Factory and Tavily expect."""

import json
from dataclasses import replace

import httpx
import pytest

from clerk.config import ModelPrice, load_settings
from clerk.llm.base import LLMError, LLMRequest
from clerk.llm.nebius import NebiusProvider
from clerk.search import SearchError, TavilyClient


@pytest.fixture
def live_settings(settings):
    return replace(
        settings,
        mode="live",
        nebius_api_key="nb-test",
        model_super="nvidia/super-test",
        model_nano="nvidia/nano-test",
        model_vision=None,
        model_embed="nvidia/embed-test",
    )


def provider(live_settings, handler, monkeypatch):
    monkeypatch.setattr("clerk.llm.nebius.time.sleep", lambda s: None)
    client = httpx.Client(
        base_url=live_settings.nebius_base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {live_settings.nebius_api_key}"},
    )
    return NebiusProvider(live_settings, client=client)


def chat_reply(text="hi", prompt=12, completion=3):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
    )


def test_chat_request_shape(live_settings, monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return chat_reply("ready")

    p = provider(live_settings, handler, monkeypatch)
    result = p.complete(
        LLMRequest(task="t", role="nano", messages=[{"role": "user", "content": "x"}], max_tokens=50, temperature=0.2)
    )
    assert seen["url"] == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert seen["auth"] == "Bearer nb-test"
    assert seen["body"] == {
        "model": "nvidia/nano-test",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 50,
        "temperature": 0.2,
    }
    assert (result.text, result.model, result.mode, result.prompt_tokens, result.completion_tokens) == (
        "ready",
        "nvidia/nano-test",
        "live",
        12,
        3,
    )


def test_vision_falls_back_to_super_model(live_settings, monkeypatch):
    p = provider(live_settings, lambda r: chat_reply(), monkeypatch)
    assert p.model_for("vision") == "nvidia/super-test"


def test_retries_on_429_and_5xx_then_succeeds(live_settings, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return [httpx.Response(429), httpx.Response(503), chat_reply("ok")][len(calls) - 1]

    p = provider(live_settings, handler, monkeypatch)
    assert p.complete(LLMRequest(task="t", role="super", messages=[])).text == "ok"
    assert len(calls) == 3


def test_retries_on_transport_errors(live_settings, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectTimeout("slow")
        return chat_reply("ok")

    assert (
        provider(live_settings, handler, monkeypatch).complete(LLMRequest(task="t", role="super", messages=[])).text
        == "ok"
    )


def test_gives_up_after_max_attempts(live_settings, monkeypatch):
    p = provider(live_settings, lambda r: httpx.Response(502), monkeypatch)
    with pytest.raises(LLMError, match="after 3 attempts"):
        p.complete(LLMRequest(task="t", role="super", messages=[]))


def test_does_not_retry_client_errors(live_settings, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(LLMError, match="401"):
        provider(live_settings, handler, monkeypatch).complete(LLMRequest(task="t", role="super", messages=[]))
    assert len(calls) == 1


def test_unexpected_response_shape(live_settings, monkeypatch):
    p = provider(live_settings, lambda r: httpx.Response(200, json={"oops": True}), monkeypatch)
    with pytest.raises(LLMError, match="unexpected"):
        p.complete(LLMRequest(task="t", role="super", messages=[]))


def test_embeddings_request_and_order(live_settings, monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path.endswith("/embeddings")
        assert body == {"model": "nvidia/embed-test", "input": ["a", "b"]}
        return httpx.Response(
            200, json={"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}
        )

    assert provider(live_settings, handler, monkeypatch).embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_embeddings_fall_back_to_hashing_without_model(live_settings, monkeypatch):
    p = provider(replace(live_settings, model_embed=None), lambda r: pytest.fail("no HTTP expected"), monkeypatch)
    [vec] = p.embed(["late fee"])
    assert len(vec) == 256


def test_missing_key_is_rejected(live_settings):
    with pytest.raises(LLMError):
        NebiusProvider(replace(live_settings, nebius_api_key=None))


def test_live_mode_requires_keys(monkeypatch):
    monkeypatch.setenv("CLERK_MODE", "live")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NEBIUS_API_KEY"):
        load_settings()


def test_invalid_mode(monkeypatch):
    monkeypatch.setenv("CLERK_MODE", "turbo")
    with pytest.raises(ValueError):
        load_settings()


def test_prices_from_env(monkeypatch):
    monkeypatch.setenv("CLERK_MODE", "live")
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.setenv("CLERK_MODEL_SUPER", "s")
    monkeypatch.setenv("CLERK_MODEL_NANO", "n")
    monkeypatch.setenv("CLERK_PRICE_SUPER_IN", "0.3")
    monkeypatch.setenv("CLERK_PRICE_SUPER_OUT", "0.9")
    s = load_settings()
    assert s.price_for("s").cost(1_000_000, 1_000_000) == pytest.approx(1.2)
    assert s.price_for("n").cost(10, 10) is None
    assert ModelPrice().cost(1, 1) is None


def test_tavily_request_shape():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "T", "url": "https://example.org/a", "content": "c" * 600, "score": 0.9},
                    {"title": "no url"},
                ]
            },
        )

    client = TavilyClient("tv-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    [result] = client.search("late fee dispute", max_results=2)
    assert seen["auth"] == "Bearer tv-key"
    assert seen["body"]["query"] == "late fee dispute" and seen["body"]["max_results"] == 2
    assert result.url == "https://example.org/a" and len(result.snippet) == 400


def test_tavily_errors():
    client = TavilyClient(
        "k", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(432, text="limit")))
    )
    with pytest.raises(SearchError, match="432"):
        client.search("q")
