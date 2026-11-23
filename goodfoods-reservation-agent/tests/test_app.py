"""UI tests: run the real Streamlit script with a scripted LLM (Streamlit AppTest)."""
import json

import pytest
from streamlit.testing.v1 import AppTest

import agent.factory as factory
from agent.llm_client import LLMResponse
from mcp_server import db
from tests.fakes import ScriptedLLM, tool_call


@pytest.fixture
def app(tmp_path, monkeypatch):
    db_path = tmp_path / "ui.db"
    monkeypatch.setenv("GOODFOODS_DB", str(db_path))      # MCP server subprocess
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", db_path)     # browse tab
    llm = ScriptedLLM()
    monkeypatch.setattr(factory, "LLMClient", lambda: llm)
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    yield at, llm
    if at.session_state["agent"]:
        at.session_state["bridge"].close()


def test_home_renders_with_quick_starts(app):
    at, _ = app
    assert not at.exception
    assert any("Hi, I'm Gia" in m.value for m in at.markdown)
    assert len(at.button) >= 5


def test_booking_shows_in_sidebar(app):
    at, llm = app
    llm.responses += [
        LLMResponse(None, [tool_call("make_reservation", {"restaurant": 1, "date": "tomorrow", "time": "20:00",
                                                          "party_size": 2, "customer_name": "Nina",
                                                          "phone": "9876543210"})]),
        LLMResponse("You're booked!"),
    ]
    at.chat_input[0].set_value("yes, confirm").run()
    assert not at.exception
    assert any("You're booked!" in m.value for m in at.markdown)
    codes = list(at.session_state["bookings"])
    assert len(codes) == 1 and codes[0].startswith("GF-")
    assert any(codes[0] in m.value for m in at.sidebar.markdown)


def test_quick_start_button_sends_message(app):
    at, llm = app
    llm.responses += [LLMResponse(None, [tool_call("search_restaurants", {"cuisine": "Italian", "area": "Indiranagar"})]),
                      LLMResponse("Here are Italian places in Indiranagar.")]
    next(b for b in at.button if "Italian" in b.label).click().run()
    assert at.session_state["messages"][0]["content"] == "Show me Italian restaurants in Indiranagar"
    assert at.session_state["messages"][1]["events"][0].name == "search_restaurants"


def test_new_conversation_clears_chat(app):
    at, llm = app
    llm.responses += [LLMResponse("Hello!")]
    at.chat_input[0].set_value("hi").run()
    next(b for b in at.sidebar.button if "New conversation" in b.label).click().run()
    assert at.session_state["messages"] == []
