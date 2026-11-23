import pytest

from data.seed import seed
from mcp_server import db


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    seed(connection, n=75, with_bookings=False)
    yield connection
    connection.close()


from datetime import datetime
from zoneinfo import ZoneInfo

from mcp_server.tools import ReservationService

FIXED_NOW = datetime(2026, 11, 23, 15, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # a Monday


@pytest.fixture
def svc(conn):
    return ReservationService(conn, clock=lambda: FIXED_NOW)


@pytest.fixture
def small(svc):
    """The smallest evening restaurant (easy to fill up)."""
    rows = [r for r in svc._restaurants if r["open_time"] <= "19:00" and r["close_time"] >= "22:00"]
    return min(rows, key=lambda r: (r["total_seats"], r["id"]))
