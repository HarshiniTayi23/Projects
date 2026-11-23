from data.seed import CUISINES, AREAS, generate_restaurants, seed
from mcp_server import db


def test_seed_count_in_required_range(conn):
    assert 50 <= db.count_restaurants(conn) <= 100


def test_seed_twice_does_not_duplicate(conn):
    before = db.count_restaurants(conn)
    seed(conn)
    assert db.count_restaurants(conn) == before


def test_reset_regenerates(conn):
    assert seed(conn, n=60, reset=True) == 60


def test_hours_and_capacity_are_valid(conn):
    for r in db.all_restaurants(conn):
        assert r["open_time"] < r["close_time"], r["name"]
        assert 20 <= r["total_seats"] <= 200
        assert 1 <= r["max_party_size"] <= min(20, r["total_seats"])
        assert r["features"] and r["ambience"]


def test_every_cuisine_and_area_is_represented():
    rows = generate_restaurants(75)
    assert {r["cuisine"] for r in rows} == set(CUISINES)
    assert {r["area"] for r in rows} == set(AREAS)


def test_generation_is_deterministic():
    assert generate_restaurants(75) == generate_restaurants(75)


def test_names_are_unique(conn):
    names = [r["name"] for r in db.all_restaurants(conn)]
    assert len(names) == len(set(names))
