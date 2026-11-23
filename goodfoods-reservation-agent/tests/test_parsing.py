from datetime import date

import pytest

from mcp_server.parsing import ParseError, fuzzy_pick, normalize_name, normalize_phone, parse_date, parse_time

MON = date(2026, 11, 23)


@pytest.mark.parametrize("text,expected", [
    ("2026-11-25", date(2026, 11, 25)), ("today", MON), ("tonight", MON), ("tomorrow", date(2026, 11, 24)),
    ("saturday", date(2026, 11, 28)), ("next monday", date(2026, 11, 30)), ("this friday", date(2026, 11, 27)),
    ("25th December", date(2026, 12, 25)), ("Dec 5", date(2026, 12, 5)), ("5 jan", date(2027, 1, 5)),
    ("28/11/2026", date(2026, 11, 28)),
])
def test_parse_date(text, expected):
    assert parse_date(text, MON) == expected


@pytest.mark.parametrize("text,expected", [
    ("19:30", "19:30"), ("8pm", "20:00"), ("8:30 PM", "20:30"), ("6", "18:00"), ("7", "07:00"), ("11am", "11:00"),
    ("12pm", "12:00"), ("12am", "00:00"), ("20.15", "20:15"), ("noon", "12:00"),
])
def test_parse_time(text, expected):
    assert parse_time(text) == expected


@pytest.mark.parametrize("bad", ["sometime", "25:00", "8:75"])
def test_parse_time_rejects_garbage(bad):
    with pytest.raises(ParseError):
        parse_time(bad)


def test_parse_date_rejects_garbage():
    with pytest.raises(ParseError):
        parse_date("whenever", MON)


@pytest.mark.parametrize("text,expected", [
    ("98765 43210", "+919876543210"), ("+91-98765-43210", "+919876543210"), ("919876543210", "+919876543210"),
])
def test_phone(text, expected):
    assert normalize_phone(text) == expected


@pytest.mark.parametrize("bad", ["12345", "abc", "0123456789"])
def test_phone_rejects(bad):
    with pytest.raises(ParseError):
        normalize_phone(bad)


def test_name():
    assert normalize_name("  priya   sharma ") == "Priya Sharma"
    with pytest.raises(ParseError):
        normalize_name("x1!")


def test_fuzzy_pick():
    opts = ["Indiranagar", "HSR Layout", "MG Road"]
    assert fuzzy_pick("indira nagar", opts) == "Indiranagar"
    assert fuzzy_pick("hsr", opts) is None or fuzzy_pick("hsr", opts) == "HSR Layout"
    assert fuzzy_pick("mg rd", opts) == "MG Road"
    assert fuzzy_pick("Paris", opts) is None
