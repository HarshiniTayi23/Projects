"""GoodFoods MCP server: exposes the reservation tools over the Model Context Protocol.

Run standalone (stdio transport):
    python -m mcp_server.server

The tool docstrings and parameter descriptions below are exactly what the LLM
reads when deciding which tool fits the user's intent, so they are written as
"when to use" guidance, not just API docs.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Annotated

from pydantic import Field

from mcp.server.mcpserver import MCPServer
from data.seed import seed
from mcp_server import db
from mcp_server.tools import ReservationService

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)  # stdout is reserved for MCP messages

_conn = db.connect(os.getenv("GOODFOODS_DB") or None)
seed(_conn)  # no-op if already seeded
service = ReservationService(_conn, tz=os.getenv("APP_TIMEZONE", "Asia/Kolkata"))

mcp = MCPServer(
    name="goodfoods-reservations",
    instructions="Search, recommend and book tables at GoodFoods restaurants in Bengaluru.",
)

Restaurant = Annotated[int | str, Field(description="Restaurant id (preferred, from search results) or its name")]
Date = Annotated[str, Field(description="Date as YYYY-MM-DD. Words like 'tomorrow' or 'saturday' are also accepted")]
Time = Annotated[str, Field(description="Time as 24-hour HH:MM, e.g. 19:30")]
PartySize = Annotated[int, Field(description="Number of guests", ge=1, le=50)]


def _out(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def list_options() -> str:
    """List the cuisines, neighbourhoods (areas), features and price levels GoodFoods offers.
    Use when the user asks what is available, or when you need valid filter values."""
    return _out(service.list_options())


@mcp.tool()
def search_restaurants(
    cuisine: Annotated[str | None, Field(description="Cuisine, e.g. Italian, North Indian, Japanese")] = None,
    area: Annotated[str | None, Field(description="Neighbourhood, e.g. Indiranagar, Koramangala")] = None,
    party_size: Annotated[int | None, Field(description="Only show places that can seat this many")] = None,
    max_price_tier: Annotated[int | str | None, Field(description="1=budget .. 4=fine dining, or a word like 'moderate'")] = None,
    features: Annotated[str | None, Field(description="Comma-separated needs, e.g. 'outdoor, vegan, parking'")] = None,
    name: Annotated[str | None, Field(description="Part of a restaurant name")] = None,
) -> str:
    """Find GoodFoods restaurants matching explicit filters (cuisine, area, price, features).
    Use when the user names concrete criteria or asks to list/browse restaurants.
    Does NOT check table availability."""
    return _out(service.search_restaurants(cuisine, area, party_size, max_price_tier, features, name))


@mcp.tool()
def recommend_restaurants(
    party_size: Annotated[int | None, Field(description="Number of guests")] = None,
    date: Annotated[str | None, Field(description="Optional date; if given with time, only available places are returned")] = None,
    time: Annotated[str | None, Field(description="Optional time HH:MM")] = None,
    cuisine: Annotated[str | None, Field(description="Preferred cuisine")] = None,
    area: Annotated[str | None, Field(description="Preferred neighbourhood")] = None,
    budget: Annotated[int | str | None, Field(description="1-4 or budget/moderate/upscale/fine dining")] = None,
    occasion: Annotated[str | None, Field(description="e.g. anniversary, birthday, business meeting, family dinner")] = None,
    preferences: Annotated[str | None, Field(description="Comma-separated wishes, e.g. 'romantic, rooftop, live music'")] = None,
) -> str:
    """Suggest the best 3 restaurants with reasons. Use when the user is undecided, asks for
    suggestions or 'somewhere good/nice for X', or describes a mood or occasion."""
    return _out(service.recommend_restaurants(party_size, date, time, cuisine, area, budget, occasion, preferences))


@mcp.tool()
def get_restaurant_details(restaurant: Restaurant) -> str:
    """Full details of one restaurant: address, hours, features, phone, description."""
    return _out(service.get_restaurant_details(restaurant))


@mcp.tool()
def check_availability(restaurant: Restaurant, date: Date, time: Time, party_size: PartySize) -> str:
    """Check if a table is free at a restaurant for a date, time and party size.
    Always call this before make_reservation. If full, it returns alternative times and venues."""
    return _out(service.check_availability(restaurant, date, time, party_size))


@mcp.tool()
def make_reservation(
    restaurant: Restaurant,
    date: Date,
    time: Time,
    party_size: PartySize,
    customer_name: Annotated[str, Field(description="Guest's name as given by the user")],
    phone: Annotated[str, Field(description="Guest's mobile number as given by the user")],
    occasion: Annotated[str | None, Field(description="Optional occasion, e.g. birthday")] = None,
    special_requests: Annotated[str | None, Field(description="Optional requests, e.g. window seat, high chair")] = None,
) -> str:
    """Create a confirmed booking. Only call after the user has explicitly confirmed the
    summary (restaurant, date, time, guests, name, phone). Never invent name or phone."""
    return _out(service.make_reservation(restaurant, date, time, party_size, customer_name, phone, occasion, special_requests))


@mcp.tool()
def get_reservation(code: Annotated[str, Field(description="Confirmation code, e.g. GF-7K2PQ")]) -> str:
    """Look up an existing booking by its confirmation code."""
    return _out(service.get_reservation(code))


@mcp.tool()
def modify_reservation(
    code: Annotated[str, Field(description="Confirmation code")],
    phone: Annotated[str, Field(description="Phone number used for the booking (for verification)")],
    date: Annotated[str | None, Field(description="New date, if changing")] = None,
    time: Annotated[str | None, Field(description="New time, if changing")] = None,
    party_size: Annotated[int | None, Field(description="New number of guests, if changing")] = None,
    special_requests: Annotated[str | None, Field(description="New special requests, if changing")] = None,
) -> str:
    """Change the date, time, party size or requests of an existing booking.
    Requires the confirmation code and the phone number used when booking."""
    return _out(service.modify_reservation(code, phone, date, time, party_size, special_requests))


@mcp.tool()
def cancel_reservation(
    code: Annotated[str, Field(description="Confirmation code")],
    phone: Annotated[str, Field(description="Phone number used for the booking (for verification)")],
) -> str:
    """Cancel an existing booking. Requires the confirmation code and the booking phone number.
    Only call after the user confirms they want to cancel."""
    return _out(service.cancel_reservation(code, phone))


if __name__ == "__main__":
    mcp.run("stdio")
