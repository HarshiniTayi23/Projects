import json

import httpx
import pytest

from agent.config import Settings
from agent.llm_client import LLMClient, LLMConfigError, LLMError, parse_text_tool_calls

TOOLS = [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}]


def make_client(handler, **overrides):
    cfg = Settings(api_key="test", max_retries=2, **overrides)
    return LLMClient(cfg, http=httpx.Client(transport=httpx.MockTransport(handler)))


def ok(message):
    return httpx.Response(200, json={"choices": [{"message": message}]})


def test_plain_text_reply():
    client = make_client(lambda req: ok({"content": "Hello there friend"}))
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "Hello there friend" and resp.tool_calls == []


def test_structured_tool_call_is_parsed():
    msg = {"content": None, "tool_calls": [{"id": "c1", "type": "function",
           "function": {"name": "get_time", "arguments": '{"city": "Mumbai"}'}}]}
    resp = make_client(lambda req: ok(msg)).chat([], tools=TOOLS)
    assert resp.tool_calls[0].name == "get_time"
    assert json.loads(resp.tool_calls[0].arguments) == {"city": "Mumbai"}


def test_tool_call_written_as_text_is_recovered():
    msg = {"content": '<function=get_time>{"city": "Pune"}</function>'}
    resp = make_client(lambda req: ok(msg)).chat([], tools=TOOLS)
    assert resp.tool_calls and resp.tool_calls[0].name == "get_time"
    assert resp.content is None


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = {"n": 0}

    def handler(req):
        attempts["n"] += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}}) if attempts["n"] < 2 else ok({"content": "done"})

    assert make_client(handler).chat([]).content == "done"
    assert attempts["n"] == 2


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(LLMError):
        make_client(lambda req: httpx.Response(503, json={})).chat([])


def test_groq_tool_use_failed_is_salvaged():
    body = {"error": {"code": "tool_use_failed",
                      "failed_generation": '<function=get_time>{"city": "Delhi"}</function>'}}
    resp = make_client(lambda req: httpx.Response(400, json=body)).chat([], tools=TOOLS)
    assert resp.tool_calls[0].name == "get_time"


def test_missing_api_key_is_a_config_error():
    client = LLMClient(Settings(api_key=""))
    with pytest.raises(LLMConfigError):
        client.chat([])


def test_bad_key_is_a_config_error():
    with pytest.raises(LLMConfigError):
        make_client(lambda req: httpx.Response(401, json={})).chat([])


def test_text_parser_ignores_unknown_tools():
    assert parse_text_tool_calls('<function=rm_rf>{"x": 1}</function>', {"get_time"}) == []
