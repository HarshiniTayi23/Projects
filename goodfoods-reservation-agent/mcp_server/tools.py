"""Reservation business logic.

Pure Python, independent of MCP and of the LLM, so every rule is unit-tested.
All hard rules (capacity, opening hours, booking window, validation) live here,
never in the prompt: the model cannot talk its way past them.

Every method returns a JSON-serialisable dict with "ok": True/False. Errors carry
a machine-readable "error" code, a human "message", and where useful
"alternatives" the agent can offer.
"""
from __future__ import annotations

import secrets
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from mcp_server import db
from mcp_server.parsing import (
    ParseError, add_minutes, fuzzy_pick, normalize_name, normalize_phone, parse_date, parse_time,
)

SLOT_MINUTES = 90           # how long a table is held per booking
MAX_DAYS_AHEAD = 60
MIN_LEAD_MINUTES = 30       # same-day bookings need this much notice
MAX_PARTY = 20
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

PRICE_WORDS = {"cheap": 1, "budget": 1, "affordable": 2, "moderate": 2, "mid": 2, "mid-range": 2,
               "upscale": 3, "expensive": 4, "premium": 4, "fine dining": 4, "luxury": 4}
OCCASION_AMBIENCE = {
    "anniversary": ["romantic", "quiet"], "date": ["romantic", "quiet"], "romantic": ["romantic"],
    "birthday": ["lively", "family"], "celebration": ["lively"], "party": ["lively"],
    "family": ["family"], "kids": ["family"], "business": ["business", "quiet"],
    "meeting": ["business", "quiet"], "casual": ["casual"], "friends": ["lively", "casual"],
}
FEATURE_SYNONYMS = {
    "outdoor": "outdoor_seating", "terrace": "outdoor_seating", "al fresco": "outdoor_seating",
    "veg": "vegetarian_friendly", "vegetarian": "vegetarian_friendly", "vegan": "vegan_options",
    "drinks": "bar", "alcohol": "bar", "cocktails": "bar", "music": "live_music", "live music": "live_music",
    "parking": "parking", "private": "private_dining", "private room": "private_dining",
    "kids": "kid_friendly", "children": "kid_friendly", "wheelchair": "wheelchair_accessible",
    "accessible": "wheelchair_accessible", "rooftop": "rooftop", "view": "rooftop", "pets": "pet_friendly",
    "dog": "pet_friendly", "wifi": "wifi",
}


def ok(**data: Any) -> dict[str, Any]:
    return {"ok": True, **data}


def fail(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": code, "message": message, **extra}


class ReservationService:
    def __init__(self, conn: sqlite3.Connection, tz: str = "Asia/Kolkata",
                 clock: Callable[[], datetime] | None = None):
        self.conn = conn
        self.tz = ZoneInfo(tz)
        self._clock = clock or (lambda: datetime.now(self.tz))
        self._restaurants = db.all_restaurants(conn)
        self.cuisines = sorted({r["cuisine"] for r in self._restaurants})
        self.areas = sorted({r["area"] for r in self._restaurants})
        self.features = sorted({f for r in self._restaurants for f in r["features"]})
        self.ambiences = sorted({a for r in self._restaurants for a in r["ambience"]})

    def now(self) -> datetime:
        return self._clock()

    # ================================================================ catalogue
    def list_options(self) -> dict[str, Any]:
        return ok(cuisines=self.cuisines, areas=self.areas, features=self.features,
                  ambience=self.ambiences, price_tiers={"1": "budget", "2": "moderate", "3": "upscale", "4": "fine dining"},
                  total_restaurants=len(self._restaurants))

    def search_restaurants(self, cuisine: str | None = None, area: str | None = None,
                           party_size: int | None = None, max_price_tier: int | str | None = None,
                           features: list[str] | str | None = None, name: str | None = None,
                           limit: int = 5) -> dict[str, Any]:
        filters, err = self._resolve_filters(cuisine, area, max_price_tier, features)
        if err:
            return err
        rows = self._filter(**filters, party_size=party_size)
        if name:
            needle = name.lower()
            rows = [r for r in rows if needle in r["name"].lower()]
        rows.sort(key=lambda r: (-r["rating"], r["name"]))
        limit = max(1, min(int(limit or 5), 10))
        return ok(count=len(rows), applied_filters={k: v for k, v in filters.items() if v},
                  restaurants=[self._summary(r) for r in rows[:limit]],
                  note=None if rows else "No exact match. Try removing a filter (area or price) or ask for a recommendation.")

    def get_restaurant_details(self, restaurant: int | str) -> dict[str, Any]:
        r, err = self._resolve_restaurant(restaurant)
        if err:
            return err
        return ok(restaurant={k: v for k, v in r.items()})

    # ============================================================ recommendation
    def recommend_restaurants(self, party_size: int | None = None, date: str | None = None,
                              time: str | None = None, cuisine: str | None = None, area: str | None = None,
                              budget: int | str | None = None, occasion: str | None = None,
                              preferences: list[str] | str | None = None, limit: int = 3) -> dict[str, Any]:
        wanted_features, wanted_ambience = self._interpret_preferences(preferences, occasion)
        filters, err = self._resolve_filters(cuisine, area, budget, None)
        if err:
            return err

        slot = None
        if date or time:
            if not (date and time):
                return fail("MISSING_FIELD", "Provide both date and time to check live availability, or neither.")
            slot, err = self._resolve_slot(date, time)
            if err:
                return err

        relaxed: list[str] = []
        candidates = self._recommend_candidates(filters, party_size, slot)
        for key in ("area", "max_price_tier", "cuisine"):
            if candidates or not filters.get(key):
                continue
            filters[key] = None
            relaxed.append(key)
            candidates = self._recommend_candidates(filters, party_size, slot)

        scored = []
        for r in candidates:
            score, reasons = r["rating"] * 10, [f"rated {r['rating']}"]
            if filters.get("cuisine") and r["cuisine"] == filters["cuisine"]:
                score += 15; reasons.append(f"{r['cuisine']} as requested")
            if filters.get("area") and r["area"] == filters["area"]:
                score += 12; reasons.append(f"in {r['area']}")
            hits_f = [f for f in wanted_features if f in r["features"]]
            hits_a = [a for a in wanted_ambience if a in r["ambience"]]
            score += 8 * len(hits_f) + 10 * len(hits_a)
            if hits_a:
                reasons.append(f"{'/'.join(hits_a)} ambience")
            if hits_f:
                reasons.append("has " + ", ".join(h.replace("_", " ") for h in hits_f))
            if party_size and party_size >= 8 and "private_dining" in r["features"]:
                score += 5; reasons.append("private dining for groups")
            scored.append((score, r, reasons))

        scored.sort(key=lambda x: (-x[0], x[1]["name"]))
        limit = max(1, min(int(limit or 3), 5))
        picks = []
        for score, r, reasons in scored[:limit]:
            item = self._summary(r)
            item["why"] = "; ".join(reasons)
            if slot:
                item["seats_left_at_requested_time"] = r["_seats_left"]
            picks.append(item)

        if not picks:
            return fail("NO_MATCH", "No restaurant can host this request. Try a different time or a smaller party.",
                        relaxed_filters=relaxed)
        return ok(recommendations=picks, relaxed_filters=relaxed,
                  note=(f"No exact match, so I ignored: {', '.join(relaxed)}." if relaxed else None),
                  checked_slot=slot and {"date": slot[0], "time": slot[1]})

    def _recommend_candidates(self, filters, party_size, slot):
        rows = self._filter(**filters, party_size=party_size)
        out = []
        for r in rows:
            if slot:
                d, t = slot
                if self._hours_problem(r, t):
                    continue
                left = r["total_seats"] - db.booked_seats(self.conn, r["id"], d, t, add_minutes(t, SLOT_MINUTES))
                if left < (party_size or 1):
                    continue
                r = {**r, "_seats_left": left}
            out.append(r)
        return out

    # ============================================================== availability
    def check_availability(self, restaurant: int | str, date: str, time: str, party_size: int) -> dict[str, Any]:
        r, err = self._resolve_restaurant(restaurant)
        if err:
            return err
        slot, err = self._resolve_slot(date, time)
        if err:
            return err
        d, t = slot
        err = self._party_problem(r, party_size) or self._hours_problem(r, t)
        if err:
            return err

        left = self._seats_left(r, d, t)
        result = {"restaurant": self._summary(r), "date": d, "time": t, "party_size": party_size,
                  "available": left >= party_size, "seats_left": max(left, 0)}
        if left < party_size:
            result["alternative_times"] = self._alternative_times(r, d, t, party_size)
            result["alternative_restaurants"] = self._alternative_restaurants(r, d, t, party_size)
            result["message"] = "That slot is full. Offer the alternatives."
        return ok(**result)

    # ================================================================ booking
    def make_reservation(self, restaurant: int | str, date: str, time: str, party_size: int,
                         customer_name: str, phone: str, occasion: str | None = None,
                         special_requests: str | None = None) -> dict[str, Any]:
        r, err = self._resolve_restaurant(restaurant)
        if err:
            return err
        slot, err = self._resolve_slot(date, time)
        if err:
            return err
        d, t = slot
        try:
            name = normalize_name(customer_name)
            phone_n = normalize_phone(phone)
        except ParseError as exc:
            return fail("INVALID_CONTACT", str(exc))
        err = self._party_problem(r, party_size) or self._hours_problem(r, t)
        if err:
            return err

        end = add_minutes(t, SLOT_MINUTES)
        self.conn.execute("BEGIN IMMEDIATE")  # lock: check-then-insert must be atomic
        try:
            duplicate = self.conn.execute(
                "SELECT code FROM reservations WHERE phone=? AND restaurant_id=? AND date=? AND start_time=? AND status='confirmed'",
                (phone_n, r["id"], d, t)).fetchone()
            if duplicate:
                self.conn.execute("ROLLBACK")
                return fail("DUPLICATE_BOOKING", f"This guest already has booking {duplicate[0]} for that exact slot.",
                            existing_code=duplicate[0])
            left = r["total_seats"] - db.booked_seats(self.conn, r["id"], d, t, end)
            if left < party_size:
                self.conn.execute("ROLLBACK")
                return fail("FULLY_BOOKED", f"Only {max(left, 0)} seats left at {t}.",
                            alternative_times=self._alternative_times(r, d, t, party_size),
                            alternative_restaurants=self._alternative_restaurants(r, d, t, party_size))
            code = self._new_code()
            self.conn.execute(
                """INSERT INTO reservations (code, restaurant_id, customer_name, phone, party_size, date,
                   start_time, end_time, occasion, special_requests) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (code, r["id"], name, phone_n, party_size, d, t, end, _clean(occasion), _clean(special_requests, 300)))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return ok(confirmation_code=code, reservation=self._reservation_view(db.get_reservation(self.conn, code)),
                  message="Booking confirmed. Share the confirmation code with the guest.")

    def get_reservation(self, code: str) -> dict[str, Any]:
        res = db.get_reservation(self.conn, _code(code))
        if not res:
            return fail("NOT_FOUND", f"No reservation with code '{code}'. Codes look like GF-AB12C.")
        return ok(reservation=self._reservation_view(res))

    def modify_reservation(self, code: str, phone: str, date: str | None = None, time: str | None = None,
                           party_size: int | None = None, special_requests: str | None = None) -> dict[str, Any]:
        res, err = self._owned_active_reservation(code, phone)
        if err:
            return err
        if not any([date, time, party_size, special_requests]):
            return fail("NOTHING_TO_CHANGE", "Tell me what to change: date, time, party size or special requests.")
        r = db.get_restaurant(self.conn, res["restaurant_id"])

        new_date, new_time = res["date"], res["start_time"]
        if date or time:
            slot, err = self._resolve_slot(date or res["date"], time or res["start_time"])
            if err:
                return err
            new_date, new_time = slot
        new_size = int(party_size or res["party_size"])
        err = self._party_problem(r, new_size) or self._hours_problem(r, new_time)
        if err:
            return err

        end = add_minutes(new_time, SLOT_MINUTES)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            left = r["total_seats"] - db.booked_seats(self.conn, r["id"], new_date, new_time, end, exclude_code=res["code"])
            if left < new_size:
                self.conn.execute("ROLLBACK")
                return fail("FULLY_BOOKED", "The new slot is full. Your original booking is unchanged.",
                            original=self._reservation_view(res),
                            alternative_times=self._alternative_times(r, new_date, new_time, new_size, exclude=res["code"]))
            self.conn.execute(
                """UPDATE reservations SET date=?, start_time=?, end_time=?, party_size=?,
                   special_requests=COALESCE(?, special_requests), updated_at=datetime('now') WHERE code=?""",
                (new_date, new_time, end, new_size, _clean(special_requests, 300), res["code"]))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return ok(reservation=self._reservation_view(db.get_reservation(self.conn, res["code"])),
                  message="Reservation updated.")

    def cancel_reservation(self, code: str, phone: str) -> dict[str, Any]:
        res, err = self._owned_active_reservation(code, phone)
        if err:
            return err
        self.conn.execute("UPDATE reservations SET status='cancelled', updated_at=datetime('now') WHERE code=?",
                          (res["code"],))
        return ok(cancelled_code=res["code"], message="Reservation cancelled. The table has been released.")

    # ================================================================= helpers
    def _resolve_restaurant(self, restaurant: int | str | None):
        if restaurant is None or str(restaurant).strip() == "":
            return None, fail("MISSING_FIELD", "Which restaurant? Give its id or name.")
        text = str(restaurant).strip()
        if text.isdigit():
            r = db.get_restaurant(self.conn, int(text))
            return (r, None) if r else (None, fail("NOT_FOUND", f"No restaurant with id {text}."))
        matches = db.find_restaurants_by_name(self.conn, text)
        if not matches:
            names = [r["name"] for r in self._restaurants]
            best = fuzzy_pick(text, names, cutoff=0.6)
            matches = [r for r in self._restaurants if r["name"] == best] if best else []
        exact = [m for m in matches if m["name"].lower() == text.lower()]
        if exact or len(matches) == 1:
            return (exact or matches)[0], None
        if not matches:
            return None, fail("NOT_FOUND", f"No restaurant matching '{text}'. Use search_restaurants to find one.")
        return None, fail("AMBIGUOUS_RESTAURANT", f"'{text}' matches several restaurants. Ask which one.",
                          options=[{"id": m["id"], "name": m["name"]} for m in matches[:6]])

    def _resolve_slot(self, date_s: str | None, time_s: str | None):
        if not date_s or not time_s:
            return None, fail("MISSING_FIELD", "Both a date and a time are needed.")
        now = self.now()
        try:
            d = parse_date(date_s, now.date())
            t = parse_time(time_s)
        except ParseError as exc:
            return None, fail("INVALID_DATETIME", str(exc), today=now.date().isoformat())
        if d < now.date():
            return None, fail("DATE_IN_PAST", f"{d.isoformat()} is in the past. Today is {now.date().isoformat()}.")
        if d > now.date() + timedelta(days=MAX_DAYS_AHEAD):
            return None, fail("TOO_FAR_AHEAD", f"Bookings open only {MAX_DAYS_AHEAD} days ahead.")
        if d == now.date():
            earliest = (now + timedelta(minutes=MIN_LEAD_MINUTES)).strftime("%H:%M")
            if t < earliest:
                return None, fail("TIME_TOO_SOON", f"Same-day bookings need {MIN_LEAD_MINUTES} min notice. "
                                  f"Earliest today is {earliest}.", earliest_time=earliest)
        return (d.isoformat(), t), None

    def _hours_problem(self, r: dict, t: str):
        last = add_minutes(r["close_time"], -SLOT_MINUTES)
        if r["open_time"] <= t <= last:
            return None
        hint = None
        h = int(t[:2])
        if h < 12 and r["open_time"] <= f"{h + 12:02d}{t[2:]}" <= last:
            hint = f"Did the guest mean {h + 12:02d}{t[2:]}?"
        return fail("OUTSIDE_OPENING_HOURS", f"{r['name']} takes bookings from {r['open_time']} to {last} "
                    f"(closes {r['close_time']}).", hint=hint)

    def _party_problem(self, r: dict, size):
        try:
            size = int(size)
        except (TypeError, ValueError):
            return fail("INVALID_PARTY_SIZE", "Party size must be a whole number.")
        if size < 1:
            return fail("INVALID_PARTY_SIZE", "Party size must be at least 1.")
        if size > min(MAX_PARTY, r["max_party_size"]):
            larger = [self._summary(x) for x in self._restaurants
                      if x["max_party_size"] >= size and "private_dining" in x["features"]][:3]
            return fail("PARTY_TOO_LARGE", f"{r['name']} accepts online bookings up to {r['max_party_size']} guests.",
                        suggestion=("For groups above 20 please call our events team." if size > MAX_PARTY
                                    else "These venues have private dining for larger groups."),
                        alternative_restaurants=larger)
        return None

    def _seats_left(self, r, d, t, exclude=None):
        return r["total_seats"] - db.booked_seats(self.conn, r["id"], d, t, add_minutes(t, SLOT_MINUTES), exclude)

    def _alternative_times(self, r, d, t, size, exclude=None) -> list[str]:
        base = int(t[:2]) * 60 + int(t[3:])
        out = []
        for delta in (-30, 30, -60, 60, -90, 90, 120):
            m = base + delta
            if m < 0 or m >= 24 * 60:
                continue
            cand = f"{m // 60:02d}:{m % 60:02d}"
            if self._hours_problem(r, cand) or self._resolve_slot(d, cand)[1]:
                continue
            if self._seats_left(r, d, cand, exclude) >= size:
                out.append(cand)
            if len(out) == 3:
                break
        return sorted(out)

    def _alternative_restaurants(self, r, d, t, size) -> list[dict]:
        pool = [x for x in self._restaurants if x["id"] != r["id"] and x["max_party_size"] >= size
                and not self._hours_problem(x, t)]
        pool.sort(key=lambda x: (x["cuisine"] != r["cuisine"], x["area"] != r["area"], -x["rating"]))
        out = []
        for x in pool:
            if self._seats_left(x, d, t) >= size:
                out.append(self._summary(x))
            if len(out) == 3:
                break
        return out

    def _owned_active_reservation(self, code, phone):
        res = db.get_reservation(self.conn, _code(code))
        if not res:
            return None, fail("NOT_FOUND", f"No reservation with code '{code}'.")
        try:
            phone_n = normalize_phone(phone)
        except ParseError as exc:
            return None, fail("INVALID_CONTACT", str(exc))
        if phone_n != res["phone"]:
            return None, fail("VERIFICATION_FAILED", "That phone number doesn't match this booking.")
        if res["status"] != "confirmed":
            return None, fail("ALREADY_CANCELLED", "This reservation was already cancelled.")
        if res["date"] < self.now().date().isoformat():
            return None, fail("PAST_RESERVATION", "This reservation is in the past and can't be changed.")
        return res, None

    def _resolve_filters(self, cuisine, area, price, features):
        filters: dict[str, Any] = {"cuisine": None, "area": None, "max_price_tier": None, "features": []}
        if cuisine:
            filters["cuisine"] = fuzzy_pick(cuisine, self.cuisines)
            if not filters["cuisine"]:
                return None, fail("UNKNOWN_CUISINE", f"We don't have '{cuisine}' cuisine.", valid_cuisines=self.cuisines)
        if area:
            filters["area"] = fuzzy_pick(area, self.areas)
            if not filters["area"]:
                return None, fail("UNKNOWN_AREA", f"No GoodFoods outlet in '{area}'.", valid_areas=self.areas)
        if price not in (None, ""):
            tier = PRICE_WORDS.get(str(price).lower().strip()) if not str(price).strip().isdigit() else int(price)
            if not tier or not 1 <= tier <= 4:
                return None, fail("INVALID_PRICE", "Budget must be 1-4 or a word like budget/moderate/upscale.")
            filters["max_price_tier"] = tier
        if features:
            wanted, _ = self._interpret_preferences(features, None)
            filters["features"] = wanted
        return filters, None

    def _interpret_preferences(self, prefs, occasion):
        items = [prefs] if isinstance(prefs, str) else list(prefs or [])
        items = [p for chunk in items for p in str(chunk).replace(";", ",").split(",") if p.strip()]
        feats, amb = set(), set()
        for raw in items:
            p = raw.strip().lower()
            key = p.replace(" ", "_")
            if key in self.features:
                feats.add(key)
            elif p in self.ambiences:
                amb.add(p)
            elif p in FEATURE_SYNONYMS:
                feats.add(FEATURE_SYNONYMS[p])
            elif p in OCCASION_AMBIENCE:
                amb.update(OCCASION_AMBIENCE[p])
            else:
                for word, f in FEATURE_SYNONYMS.items():
                    if word in p:
                        feats.add(f)
        if occasion:
            for word, a in OCCASION_AMBIENCE.items():
                if word in occasion.lower():
                    amb.update(a)
        return sorted(feats), sorted(amb)

    def _filter(self, cuisine=None, area=None, max_price_tier=None, features=None, party_size=None):
        out = []
        for r in self._restaurants:
            if cuisine and r["cuisine"] != cuisine:
                continue
            if area and r["area"] != area:
                continue
            if max_price_tier and r["price_tier"] > max_price_tier:
                continue
            if features and not all(f in r["features"] for f in features):
                continue
            if party_size and int(party_size) > r["max_party_size"]:
                continue
            out.append(r)
        return out

    def _new_code(self) -> str:
        while True:
            code = "GF-" + "".join(secrets.choice(CODE_ALPHABET) for _ in range(5))
            if not db.get_reservation(self.conn, code):
                return code

    @staticmethod
    def _summary(r: dict) -> dict[str, Any]:
        return {"id": r["id"], "name": r["name"], "cuisine": r["cuisine"], "area": r["area"],
                "rating": r["rating"], "price": "₹" * r["price_tier"], "cost_for_two": r["avg_cost_for_two"],
                "hours": f"{r['open_time']}-{r['close_time']}", "features": r["features"]}

    def _reservation_view(self, res: dict) -> dict[str, Any]:
        r = db.get_restaurant(self.conn, res["restaurant_id"])
        return {"code": res["code"], "status": res["status"], "restaurant": r["name"], "restaurant_id": r["id"],
                "address": r["address"], "date": res["date"],
                "weekday": datetime.strptime(res["date"], "%Y-%m-%d").strftime("%A"),
                "time": res["start_time"], "party_size": res["party_size"], "name": res["customer_name"],
                "phone": "******" + res["phone"][-4:], "occasion": res["occasion"],
                "special_requests": res["special_requests"]}


def _clean(text: str | None, limit: int = 60) -> str | None:
    if not text:
        return None
    return " ".join(str(text).split())[:limit] or None


def _code(code: str) -> str:
    c = str(code or "").strip().upper().replace(" ", "")
    return c if c.startswith("GF-") else ("GF-" + c[2:] if c.startswith("GF") else c)
