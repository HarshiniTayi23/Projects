"""Generate GoodFoods restaurant locations (deterministic, reproducible).

    python -m data.seed            # create data/goodfoods.db if empty
    python -m data.seed --reset    # wipe and regenerate
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from mcp_server import db

AREAS = {
    "Indiranagar": "100 Feet Road",
    "Koramangala": "80 Feet Road",
    "HSR Layout": "27th Main",
    "Whitefield": "ITPL Main Road",
    "Jayanagar": "11th Main",
    "MG Road": "Church Street",
    "Malleshwaram": "Sampige Road",
    "JP Nagar": "24th Main",
    "Electronic City": "Hosur Road",
    "Hebbal": "Bellary Road",
}

# cuisine -> (concept names, price tiers, typical ambience, signature line)
CUISINES = {
    "North Indian": (["Tandoor House", "Punjab Grill Room", "Dhaba Junction"], [2, 3], ["family", "lively"], "smoky tandoori platters and slow-cooked dal makhani"),
    "South Indian": (["Tiffin Room", "Udupi Kitchen", "Chettinad Table"], [1, 2], ["family", "casual"], "crisp dosas, filter coffee and Chettinad curries"),
    "Italian": (["Trattoria", "Pasta Bar", "Wood-fire Pizzeria"], [3, 4], ["romantic", "quiet"], "hand-rolled pasta and wood-fired pizza"),
    "Chinese": (["Wok Street", "Dim Sum House"], [2, 3], ["family", "casual"], "dim sum, Hakka noodles and Sichuan specials"),
    "Japanese": (["Sushi Counter", "Izakaya"], [3, 4], ["quiet", "business"], "omakase sushi and robata grills"),
    "Continental": (["Bistro", "Grill & Co"], [3, 4], ["business", "romantic"], "steaks, salads and seasonal European plates"),
    "Mexican": (["Cantina", "Taqueria"], [2, 3], ["lively", "casual"], "tacos, burritos and fresh guacamole"),
    "Thai": (["Thai Garden", "Bangkok Kitchen"], [2, 3], ["quiet", "romantic"], "green curries, pad thai and som tam"),
    "Mughlai": (["Nawabi Darbar", "Biryani Mahal"], [2, 3], ["family", "lively"], "dum biryani and rich Awadhi kebabs"),
    "Coastal": (["Coastal Catch", "Mangalore Shack"], [2, 3], ["casual", "family"], "Mangalorean fish curries and ghee roast"),
    "Mediterranean": (["Mezze Terrace", "Olive Room"], [3, 4], ["romantic", "quiet"], "mezze platters, falafel and grilled halloumi"),
    "Cafe": (["Brew & Bake", "Garden Cafe"], [1, 2], ["casual", "quiet"], "all-day breakfast, coffee and desserts"),
}

FEATURES = ["outdoor_seating", "vegetarian_friendly", "vegan_options", "bar", "live_music", "parking",
            "private_dining", "kid_friendly", "wheelchair_accessible", "rooftop", "pet_friendly", "wifi"]

HOURS = [("07:30", "22:00"), ("11:00", "23:00"), ("12:00", "23:30"), ("12:00", "15:30"), ("18:00", "23:30"), ("11:30", "22:30")]
COST_BY_TIER = {1: (300, 600), 2: (700, 1200), 3: (1300, 2200), 4: (2400, 4000)}
SEAT_CHOICES = [20, 24, 30, 36, 40, 48, 60, 72, 80, 100, 120, 150, 180, 200]


def generate_restaurants(n: int = 75, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    combos = [(c, concept, a) for c, (concepts, *_rest) in CUISINES.items() for concept in concepts for a in AREAS]
    rng.shuffle(combos)

    # Guarantee every cuisine and every area appears at least once.
    chosen, seen_c, seen_a = [], set(), set()
    for combo in combos:
        if combo[0] not in seen_c or combo[2] not in seen_a:
            chosen.append(combo)
            seen_c.add(combo[0])
            seen_a.add(combo[2])
    for combo in combos:
        if len(chosen) >= n:
            break
        if combo not in chosen:
            chosen.append(combo)

    restaurants = []
    for i, (cuisine, concept, area) in enumerate(chosen[:n], start=1):
        _, tiers, ambience, signature = CUISINES[cuisine]
        tier = rng.choice(tiers)
        open_t, close_t = ("07:30", "22:00") if cuisine == "Cafe" else rng.choice(HOURS[1:])
        seats = rng.choice(SEAT_CHOICES)
        features = sorted(rng.sample(FEATURES, rng.randint(2, 6)))
        if cuisine in ("South Indian", "North Indian", "Cafe") and "vegetarian_friendly" not in features:
            features = sorted(features + ["vegetarian_friendly"])
        if seats >= 80 and "private_dining" not in features and rng.random() < 0.6:
            features = sorted(features + ["private_dining"])
        amb = sorted(set(ambience + ([rng.choice(["romantic", "family", "casual", "business", "lively", "quiet"])] if rng.random() < 0.4 else [])))
        lo, hi = COST_BY_TIER[tier]
        restaurants.append({
            "id": i,
            "name": f"GoodFoods {concept} - {area}",
            "area": area,
            "address": f"{rng.randint(1, 400)}, {AREAS[area]}, {area}, Bengaluru",
            "cuisine": cuisine,
            "price_tier": tier,
            "avg_cost_for_two": rng.randrange(lo, hi, 50),
            "rating": round(rng.uniform(3.6, 4.9), 1),
            "open_time": open_t,
            "close_time": close_t,
            "total_seats": seats,
            "max_party_size": min(20, max(6, seats // 4)),
            "features": features,
            "ambience": amb,
            "description": f"{concept} in {area} serving {signature}.",
            "phone": f"+91 80 {rng.randint(2000, 4999)} {rng.randint(1000, 9999)}",
        })
    return restaurants


def _demo_reservations(restaurants: list[dict], seed: int = 7) -> list[dict]:
    """A handful of existing bookings so availability is realistic.
    The smallest restaurant is fully booked tomorrow at 20:00 for demos."""
    rng = random.Random(seed)
    today = date.today()
    rows = []
    smallest = min(restaurants, key=lambda r: (r["total_seats"], r["id"]))
    tomorrow = (today + timedelta(days=1)).isoformat()
    if smallest["open_time"] <= "20:00" and smallest["close_time"] >= "21:30":
        rows.append((smallest["id"], tomorrow, "20:00", smallest["total_seats"]))
    for _ in range(120):
        r = rng.choice(restaurants)
        d = (today + timedelta(days=rng.randint(0, 14))).isoformat()
        h = rng.choice(["13:00", "13:30", "19:00", "19:30", "20:00", "20:30", "21:00"])
        if r["open_time"] <= h and _add(h, 90) <= r["close_time"]:
            rows.append((r["id"], d, h, rng.randint(2, max(2, r["max_party_size"] // 2))))
    out = []
    for n, (rid, d, h, size) in enumerate(rows):
        out.append({"code": f"GF-SEED{n:03d}", "restaurant_id": rid, "customer_name": "Walk-in Seed",
                    "phone": "+919000000000", "party_size": size, "date": d, "start_time": h,
                    "end_time": _add(h, 90), "status": "confirmed"})
    return out


def _add(hhmm: str, minutes: int) -> str:
    t = datetime.strptime(hhmm, "%H:%M") + timedelta(minutes=minutes)
    return t.strftime("%H:%M")


def seed(conn: sqlite3.Connection, n: int = 75, reset: bool = False, with_bookings: bool = True) -> int:
    if reset:
        conn.execute("DELETE FROM reservations")
        conn.execute("DELETE FROM restaurants")
    if db.count_restaurants(conn) > 0:
        return db.count_restaurants(conn)

    restaurants = generate_restaurants(n)
    conn.execute("BEGIN")
    try:
        conn.executemany(
            """INSERT INTO restaurants (id, name, area, address, cuisine, price_tier, avg_cost_for_two, rating,
               open_time, close_time, total_seats, max_party_size, features, ambience, description, phone)
               VALUES (:id, :name, :area, :address, :cuisine, :price_tier, :avg_cost_for_two, :rating,
               :open_time, :close_time, :total_seats, :max_party_size, :features, :ambience, :description, :phone)""",
            [{**r, "features": json.dumps(r["features"]), "ambience": json.dumps(r["ambience"])} for r in restaurants],
        )
        if with_bookings:
            conn.executemany(
                """INSERT INTO reservations (code, restaurant_id, customer_name, phone, party_size, date,
                   start_time, end_time, status) VALUES (:code, :restaurant_id, :customer_name, :phone,
                   :party_size, :date, :start_time, :end_time, :status)""",
                _demo_reservations(restaurants),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(restaurants)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the GoodFoods database")
    parser.add_argument("--reset", action="store_true", help="wipe and regenerate all data")
    parser.add_argument("--count", type=int, default=75)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()
    conn = db.connect(args.db)
    total = seed(conn, n=args.count, reset=args.reset)
    print(f"Database ready with {total} restaurants at {args.db or db.DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()
