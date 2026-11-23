"""GoodFoods reservation assistant: Streamlit frontend.

    streamlit run app.py
"""
from __future__ import annotations

import json
import logging

import pandas as pd
import streamlit as st

from agent.config import settings
from agent.factory import build_agent
from agent.mcp_bridge import MCPBridgeError
from data.seed import seed
from mcp_server import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    filename="goodfoods.log")

st.set_page_config(page_title="GoodFoods Reservations", page_icon="🍽️", layout="wide")

QUICK_STARTS = [
    ("✨ Recommend a place", "Can you recommend a nice restaurant for dinner?"),
    ("🍝 Italian in Indiranagar", "Show me Italian restaurants in Indiranagar"),
    ("🔎 Check my booking", "I want to check my reservation"),
    ("❌ Cancel a booking", "I need to cancel a reservation"),
]


# ------------------------------------------------------------------ state
def init_state() -> None:
    if "agent" not in st.session_state:
        try:
            agent, bridge = build_agent()
            st.session_state.agent, st.session_state.bridge = agent, bridge
            st.session_state.startup_error = None
        except (MCPBridgeError, OSError, TimeoutError) as exc:
            st.session_state.agent = None
            st.session_state.startup_error = f"Could not start the reservation service: {exc}"
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("bookings", {})
    st.session_state.setdefault("pending", None)


@st.cache_data(ttl=60)
def restaurant_table() -> pd.DataFrame:
    conn = db.connect()
    seed(conn)
    rows = db.all_restaurants(conn)
    conn.close()
    df = pd.DataFrame(rows)
    df["price"] = df["price_tier"].map(lambda t: "₹" * t)
    df["hours"] = df["open_time"] + "–" + df["close_time"]
    df["features"] = df["features"].map(lambda f: ", ".join(x.replace("_", " ") for x in f))
    df["ambience"] = df["ambience"].map(", ".join)
    return df[["id", "name", "cuisine", "area", "rating", "price", "avg_cost_for_two", "hours",
               "total_seats", "max_party_size", "ambience", "features"]]


def track_bookings(events) -> None:
    """Keep the sidebar 'Your bookings' list in sync with tool results."""
    for ev in events:
        if not ev.ok:
            continue
        data = json.loads(ev.result)
        if ev.name in ("make_reservation", "modify_reservation", "get_reservation") and data.get("reservation"):
            res = data["reservation"]
            if ev.name != "get_reservation" or res["code"] in st.session_state.bookings:
                st.session_state.bookings[res["code"]] = res
        elif ev.name == "cancel_reservation":
            code = data.get("cancelled_code")
            if code in st.session_state.bookings:
                st.session_state.bookings[code]["status"] = "cancelled"


# ------------------------------------------------------------------ rendering
def render_tool_events(events) -> None:
    for ev in events:
        icon = "✅" if ev.ok else "⚠️"
        with st.expander(f"{icon} `{ev.name}` · {ev.duration_s * 1000:.0f} ms", expanded=False):
            st.caption("Arguments chosen by the model")
            st.json(ev.arguments if isinstance(ev.arguments, dict) else {"raw": ev.arguments})
            st.caption("Tool result")
            try:
                st.json(json.loads(ev.result))
            except json.JSONDecodeError:
                st.code(ev.result)


def render_sidebar() -> bool:
    with st.sidebar:
        st.title("🍽️ GoodFoods")
        st.caption(f"{len(restaurant_table())} restaurants across Bengaluru")
        show_tools = st.toggle("Show tool calls", value=True, help="See which tools the model decided to call")

        if st.button("🔄 New conversation", width="stretch"):
            if st.session_state.get("agent"):
                st.session_state.agent.reset()
            st.session_state.messages = []
            st.rerun()

        st.subheader("Your bookings")
        if not st.session_state.bookings:
            st.caption("Bookings you make in this chat appear here.")
        for code, b in st.session_state.bookings.items():
            status = "🟢" if b["status"] == "confirmed" else "⚪ cancelled"
            st.markdown(f"**{code}** {status}  \n{b['restaurant']}  \n{b['weekday']} {b['date']} · {b['time']} · "
                        f"{b['party_size']} guests")

        st.divider()
        st.caption(f"Model: `{settings.model}`  \nTools served over MCP (stdio)")
        if not settings.api_key:
            st.error("LLM_API_KEY is missing. Copy `.env.example` to `.env` and add your key.")
    return show_tools


def handle(prompt: str, show_tools: bool) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt, "events": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🍽️"):
        with st.spinner("Checking with our restaurants..."):
            turn = st.session_state.agent.chat(prompt)
        if show_tools and turn.tool_events:
            render_tool_events(turn.tool_events)
        st.markdown(turn.reply)
    before = json.dumps(st.session_state.bookings, sort_keys=True)
    track_bookings(turn.tool_events)
    st.session_state.messages.append({"role": "assistant", "content": turn.reply, "events": turn.tool_events})
    if json.dumps(st.session_state.bookings, sort_keys=True) != before:
        st.rerun()  # the sidebar was drawn before this turn; redraw it with the new booking


# ------------------------------------------------------------------ page
def main() -> None:
    init_state()
    show_tools = render_sidebar()
    # chat_input at top level so Streamlit pins it to the bottom of the page.
    typed = st.chat_input("e.g. Table for 4 tomorrow at 8pm, somewhere with outdoor seating")
    pending, st.session_state.pending = st.session_state.pending, None
    prompt = typed or pending
    chat_tab, browse_tab = st.tabs(["💬 Reserve a table", "📋 Browse restaurants"])

    with browse_tab:
        df = restaurant_table()
        c1, c2, c3 = st.columns(3)
        cuisines = c1.multiselect("Cuisine", sorted(df["cuisine"].unique()))
        areas = c2.multiselect("Area", sorted(df["area"].unique()))
        min_rating = c3.slider("Minimum rating", 3.5, 5.0, 3.5, 0.1)
        view = df[df["rating"] >= min_rating]
        if cuisines:
            view = view[view["cuisine"].isin(cuisines)]
        if areas:
            view = view[view["area"].isin(areas)]
        st.dataframe(view, hide_index=True, width="stretch")

    with chat_tab:
        if st.session_state.startup_error:
            st.error(st.session_state.startup_error)
            return
        if not st.session_state.messages and not prompt:
            st.markdown("### Hi, I'm Gia 👋\nI can find you a table at any GoodFoods restaurant, "
                        "suggest a place for an occasion, or change an existing booking.")
            cols = st.columns(len(QUICK_STARTS))
            for col, (label, text) in zip(cols, QUICK_STARTS):
                if col.button(label, width="stretch"):
                    st.session_state.pending = text
                    st.rerun()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🍽️" if msg["role"] == "assistant" else None):
                if show_tools and msg["events"]:
                    render_tool_events(msg["events"])
                st.markdown(msg["content"])

        if prompt:
            handle(prompt, show_tools)


main()
