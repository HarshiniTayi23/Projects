"""SQLite storage for restaurants and reservations."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = Path(os.getenv("GOODFOODS_DB", PROJECT_ROOT / "data" / "goodfoods.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS restaurants (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    area           TEXT NOT NULL,
    address        TEXT NOT NULL,
    cuisine        TEXT NOT NULL,
    price_tier     INTEGER NOT NULL CHECK (price_tier BETWEEN 1 AND 4),
    avg_cost_for_two INTEGER NOT NULL,
    rating         REAL NOT NULL,
    open_time      TEXT NOT NULL,   -- HH:MM
    close_time     TEXT NOT NULL,   -- HH:MM (same day)
    total_seats    INTEGER NOT NULL,
    max_party_size INTEGER NOT NULL,
    features       TEXT NOT NULL,   -- JSON list
    ambience       TEXT NOT NULL,   -- JSON list
    description    TEXT NOT NULL,
    phone          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reservations (
    code             TEXT PRIMARY KEY,
    restaurant_id    INTEGER NOT NULL REFERENCES restaurants(id),
    customer_name    TEXT NOT NULL,
    phone            TEXT NOT NULL,
    party_size       INTEGER NOT NULL,
    date             TEXT NOT NULL,   -- YYYY-MM-DD
    start_time       TEXT NOT NULL,   -- HH:MM
    end_time         TEXT NOT NULL,   -- HH:MM
    status           TEXT NOT NULL DEFAULT 'confirmed',
    occasion         TEXT,
    special_requests TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_res_slot ON reservations(restaurant_id, date, status);
CREATE INDEX IF NOT EXISTS idx_rest_search ON restaurants(cuisine, area);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def restaurant_from_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["features"] = json.loads(data["features"])
    data["ambience"] = json.loads(data["ambience"])
    return data


def get_restaurant(conn: sqlite3.Connection, restaurant_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM restaurants WHERE id = ?", (restaurant_id,)).fetchone()
    return restaurant_from_row(row) if row else None


def all_restaurants(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [restaurant_from_row(r) for r in conn.execute("SELECT * FROM restaurants ORDER BY id")]


def find_restaurants_by_name(conn: sqlite3.Connection, text: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM restaurants WHERE name LIKE ? ORDER BY rating DESC LIMIT 10", (f"%{text}%",)
    )
    return [restaurant_from_row(r) for r in rows]


def booked_seats(conn: sqlite3.Connection, restaurant_id: int, date: str, start: str, end: str,
                 exclude_code: str | None = None) -> int:
    """Seats taken by confirmed bookings whose time window overlaps [start, end)."""
    row = conn.execute(
        """SELECT COALESCE(SUM(party_size), 0) FROM reservations
           WHERE restaurant_id = ? AND date = ? AND status = 'confirmed'
             AND start_time < ? AND end_time > ? AND code != COALESCE(?, '')""",
        (restaurant_id, date, end, start, exclude_code),
    ).fetchone()
    return int(row[0])


def get_reservation(conn: sqlite3.Connection, code: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM reservations WHERE code = ?", (code,)).fetchone()
    return dict(row) if row else None


def count_restaurants(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM restaurants").fetchone()[0]
