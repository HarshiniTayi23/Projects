"""Agent + real MCP server + scripted LLM: a full booking journey through every layer."""
import json

import pytest

from agent.factory import build_agent
from agent.llm_client import LLMResponse
from agent.mcp_bridge import MCPToolBridge
from agent.prompts import build_system_prompt
from tests.fakes import ScriptedLLM, tool_call


@pytest.fixture
def bridge(tmp_path):
    with MCPToolBridge(env={"GOODFOODS_DB": str(tmp_path / "e2e.db")}) as b:
        yield b


def test_full_booking_journey(bridge):
    llm = ScriptedLLM(
        # turn 1: recommend
        LLMResponse(None, [tool_call("recommend_restaurants", {"party_size": 2, "occasion": "anniversary"})]),
        LLMResponse("Here are three romantic options..."),
        # turn 2: check then ask for confirmation
        LLMResponse(None, [tool_call("check_availability", {"restaurant": 3, "date": "tomorrow", "time": "20:00", "party_size": 2})]),
        LLMResponse("A table is free. Shall I confirm for Meera, 9876543210?"),
        # turn 3: book
        LLMResponse(None, [tool_call("make_reservation", {"restaurant": 3, "date": "tomorrow", "time": "20:00",
                                                          "party_size": 2, "customer_name": "Meera", "phone": "9876543210"})]),
        LLMResponse("Booked!"),
    )
    agent, _ = build_agent(llm=llm, bridge=bridge)

    t1 = agent.chat("Somewhere nice for our anniversary, 2 people")
    assert t1.tool_events[0].name == "recommend_restaurants" and t1.tool_events[0].ok
    t2 = agent.chat("Option 1 tomorrow at 8pm, I'm Meera 9876543210")
    assert json.loads(t2.tool_events[0].result)["available"] is True
    t3 = agent.chat("yes please")
    booking = json.loads(t3.tool_events[0].result)
    assert booking["ok"] and booking["confirmation_code"].startswith("GF-")

    # the tool result reached the model on the follow-up request
    assert booking["confirmation_code"] in llm.calls[-1][-1]["content"]


def test_system_prompt_contains_live_date():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    p = build_system_prompt(now=datetime(2026, 11, 23, 15, 0, tzinfo=ZoneInfo("Asia/Kolkata")))
    assert "Monday, 2026-11-23" in p and "Tomorrow is 2026-11-24" in p
