"""Business-rule tests. No LLM involved."""


def book(svc, r, size=2, time="20:00", date="2026-11-24", phone="9876543210", name="Asha Rao"):
    return svc.make_reservation(r["id"], date, time, size, name, phone)


# ---------------------------------------------------------------- search
def test_search_filters_by_cuisine_and_area(svc):
    res = svc.search_restaurants(cuisine="italian", limit=10)
    assert res["ok"] and res["restaurants"]
    assert all(r["cuisine"] == "Italian" for r in res["restaurants"])


def test_search_tolerates_spelling(svc):
    assert svc.search_restaurants(area="indira nagar")["applied_filters"]["area"] == "Indiranagar"


def test_search_unknown_cuisine_lists_valid_options(svc):
    res = svc.search_restaurants(cuisine="Martian")
    assert res["error"] == "UNKNOWN_CUISINE" and "Italian" in res["valid_cuisines"]


def test_search_feature_synonyms(svc):
    res = svc.search_restaurants(features=["outdoor", "vegan"], limit=10)
    assert all({"outdoor_seating", "vegan_options"} <= set(r["features"]) for r in res["restaurants"])


# ---------------------------------------------------------- availability
def test_booking_reduces_availability(svc, small):
    before = svc.check_availability(small["id"], "2026-11-24", "20:00", 2)["seats_left"]
    assert book(svc, small, size=4)["ok"]
    after = svc.check_availability(small["id"], "2026-11-24", "20:00", 2)["seats_left"]
    assert after == before - 4


def test_overlapping_slot_counts_but_non_overlapping_does_not(svc, small):
    book(svc, small, size=4, time="19:00")
    assert svc.check_availability(small["id"], "2026-11-24", "20:00", 1)["seats_left"] == small["total_seats"] - 4
    assert svc.check_availability(small["id"], "2026-11-24", "20:30", 1)["seats_left"] == small["total_seats"]


def test_overbooking_fails_with_alternatives(svc, small):
    seats, phone = small["total_seats"], 9000000001
    while seats > 0:
        size = min(seats, small["max_party_size"])
        assert book(svc, small, size=size, phone=str(phone))["ok"]
        seats -= size
        phone += 1
    res = book(svc, small, size=2, phone="9811111111")
    assert res["error"] == "FULLY_BOOKED"
    assert res["alternative_times"] or res["alternative_restaurants"]
    chk = svc.check_availability(small["id"], "2026-11-24", "20:00", 2)
    assert chk["available"] is False and chk["alternative_restaurants"]


# ------------------------------------------------------------- validation
def test_outside_opening_hours_rejected_with_hint(svc, small):
    res = svc.make_reservation(small["id"], "2026-11-24", "08:00", 2, "Asha", "9876543210")
    assert res["error"] == "OUTSIDE_OPENING_HOURS" and "20:00" in (res["hint"] or "")


def test_last_seating_is_90_min_before_close(svc):
    r = next(x for x in svc._restaurants if x["close_time"] == "23:00")
    assert svc.check_availability(r["id"], "2026-11-24", "21:30", 2)["ok"]
    assert svc.check_availability(r["id"], "2026-11-24", "21:45", 2)["error"] == "OUTSIDE_OPENING_HOURS"


def test_past_date_and_too_far_and_too_soon(svc, small):
    assert book(svc, small, date="2026-11-01")["error"] == "DATE_IN_PAST"
    assert book(svc, small, date="2027-06-01")["error"] == "TOO_FAR_AHEAD"
    r = next(x for x in svc._restaurants if x["open_time"] <= "15:00" and x["close_time"] >= "18:00")
    assert svc.make_reservation(r["id"], "today", "15:10", 2, "Asha", "9876543210")["error"] == "TIME_TOO_SOON"


def test_party_size_limits(svc, small):
    res = book(svc, small, size=small["max_party_size"] + 1)
    assert res["error"] == "PARTY_TOO_LARGE" and "alternative_restaurants" in res
    assert book(svc, small, size=0)["error"] == "INVALID_PARTY_SIZE"


def test_invalid_contact_rejected(svc, small):
    assert book(svc, small, phone="123")["error"] == "INVALID_CONTACT"
    assert book(svc, small, name="R2D2!")["error"] == "INVALID_CONTACT"


def test_natural_language_date_time_and_name_lookup(svc, small):
    res = svc.make_reservation(small["name"], "tomorrow", "8pm", 2, "asha rao", "98765 43210")
    assert res["ok"] and res["reservation"]["date"] == "2026-11-24" and res["reservation"]["time"] == "20:00"
    assert res["reservation"]["name"] == "Asha Rao" and res["reservation"]["phone"].endswith("3210")


def test_ambiguous_restaurant_name(svc):
    res = svc.check_availability("Trattoria", "2026-11-24", "20:00", 2)
    assert res["error"] in ("AMBIGUOUS_RESTAURANT",) and len(res["options"]) > 1


def test_duplicate_booking_detected(svc, small):
    code = book(svc, small)["confirmation_code"]
    res = book(svc, small)
    assert res["error"] == "DUPLICATE_BOOKING" and res["existing_code"] == code


# ------------------------------------------------- lookup / modify / cancel
def test_get_reservation_accepts_sloppy_code(svc, small):
    code = book(svc, small)["confirmation_code"]
    assert svc.get_reservation(code.lower().replace("-", " "))["ok"] or svc.get_reservation(code.lower())["ok"]
    assert svc.get_reservation("GF-ZZZZZ")["error"] == "NOT_FOUND"


def test_cancel_frees_seats_and_requires_phone(svc, small):
    code = book(svc, small, size=6)["confirmation_code"]
    assert svc.cancel_reservation(code, "9000000000")["error"] == "VERIFICATION_FAILED"
    before = svc.check_availability(small["id"], "2026-11-24", "20:00", 1)["seats_left"]
    assert svc.cancel_reservation(code, "9876543210")["ok"]
    assert svc.check_availability(small["id"], "2026-11-24", "20:00", 1)["seats_left"] == before + 6
    assert svc.cancel_reservation(code, "9876543210")["error"] == "ALREADY_CANCELLED"


def test_modify_party_size_and_time(svc, small):
    code = book(svc, small)["confirmation_code"]
    res = svc.modify_reservation(code, "9876543210", time="19:30", party_size=3)
    assert res["ok"] and res["reservation"]["time"] == "19:30" and res["reservation"]["party_size"] == 3


def test_modify_does_not_double_count_own_seats(svc, small):
    code = book(svc, small, size=small["max_party_size"])["confirmation_code"]
    assert svc.modify_reservation(code, "9876543210", special_requests="window seat")["ok"]


def test_modify_into_full_slot_keeps_original(svc, small):
    code = book(svc, small, time="19:00")["confirmation_code"]
    seats, phone = small["total_seats"], 9100000001
    while seats > 0:
        size = min(seats, small["max_party_size"])
        assert book(svc, small, time="21:00", size=size, phone=str(phone))["ok"]
        seats -= size
        phone += 1
    res = svc.modify_reservation(code, "9876543210", time="21:00")
    assert res["error"] == "FULLY_BOOKED"
    assert svc.get_reservation(code)["reservation"]["time"] == "19:00"


def test_modify_needs_something_to_change(svc, small):
    code = book(svc, small)["confirmation_code"]
    assert svc.modify_reservation(code, "9876543210")["error"] == "NOTHING_TO_CHANGE"


# ---------------------------------------------------------- recommendation
def test_recommend_respects_party_size(svc):
    res = svc.recommend_restaurants(party_size=12, limit=5)
    assert all(svc.get_restaurant_details(r["id"])["restaurant"]["max_party_size"] >= 12 for r in res["recommendations"])


def test_recommend_uses_occasion_and_gives_reasons(svc):
    res = svc.recommend_restaurants(party_size=2, occasion="anniversary", limit=3)
    assert res["ok"] and all(r["why"] for r in res["recommendations"])
    assert any("romantic" in r["why"] or "quiet" in r["why"] for r in res["recommendations"])


def test_recommend_relaxes_impossible_filters(svc):
    res = svc.recommend_restaurants(party_size=2, cuisine="Japanese", area="Malleshwaram", budget="budget")
    assert res["ok"] and res["relaxed_filters"]


def test_recommend_with_slot_skips_full_restaurants(svc, small):
    seats, phone = small["total_seats"], 9200000001
    while seats > 0:
        size = min(seats, small["max_party_size"])
        book(svc, small, size=size, phone=str(phone))
        seats -= size
        phone += 1
    res = svc.recommend_restaurants(party_size=2, date="2026-11-24", time="20:00", cuisine=small["cuisine"],
                                    area=small["area"], limit=5)
    assert small["id"] not in [r["id"] for r in res["recommendations"]]


def test_list_options(svc):
    res = svc.list_options()
    assert len(res["cuisines"]) == 12 and len(res["areas"]) == 10
