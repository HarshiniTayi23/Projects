"""Integration tests: real MCP server subprocess over stdio, via the bridge."""
import json

import pytest

from agent.mcp_bridge import MCPToolBridge

EXPECTED = {"list_options", "search_restaurants", "recommend_restaurants", "get_restaurant_details",
            "check_availability", "make_reservation", "get_reservation", "modify_reservation", "cancel_reservation"}


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("mcp") / "mcp.db"
    with MCPToolBridge(env={"GOODFOODS_DB": str(db_path)}) as b:
        yield b


def test_all_tools_listed(bridge):
    assert {s["function"]["name"] for s in bridge.schemas()} == EXPECTED


def test_schemas_are_flat_and_described(bridge):
    for s in bridge.schemas():
        fn = s["function"]
        assert fn["description"], fn["name"]
        for prop in fn["parameters"]["properties"].values():
            assert "anyOf" not in prop and "type" in prop


def test_required_fields_exposed(bridge):
    mk = next(s for s in bridge.schemas() if s["function"]["name"] == "make_reservation")
    assert set(mk["function"]["parameters"]["required"]) == {
        "restaurant", "date", "time", "party_size", "customer_name", "phone"}


def test_search_over_mcp(bridge):
    data = json.loads(bridge.call("search_restaurants", {"cuisine": "Thai"}))
    assert data["ok"] and all(r["cuisine"] == "Thai" for r in data["restaurants"])


def test_validation_error_is_returned_not_raised(bridge):
    data = json.loads(bridge.call("check_availability", {"restaurant": 1}))
    assert data["ok"] is False and data["error"] == "TOOL_ERROR"


def test_book_lookup_cancel_over_mcp(bridge):
    rec = json.loads(bridge.call("recommend_restaurants", {"party_size": 2, "date": "tomorrow", "time": "20:00"}))
    rid = rec["recommendations"][0]["id"]
    made = json.loads(bridge.call("make_reservation", {"restaurant": rid, "date": "tomorrow", "time": "20:00",
                                                        "party_size": 2, "customer_name": "Test Guest",
                                                        "phone": "9876500000"}))
    code = made["confirmation_code"]
    assert json.loads(bridge.call("get_reservation", {"code": code}))["reservation"]["status"] == "confirmed"
    assert json.loads(bridge.call("cancel_reservation", {"code": code, "phone": "9876500000"}))["ok"]
