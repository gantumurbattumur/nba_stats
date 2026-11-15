import json
from datetime import date

import responses

from conftest import load_fixture
from nba_predict.data.espn_client import (
    BASE,
    EspnClient,
    parse_pickcenter,
    parse_scoreboard,
)
from nba_predict.ingest import ingest_espn_lines


def test_parse_pickcenter_with_odds(summary_with_odds):
    p = parse_pickcenter(summary_with_odds)
    assert p["book"] == "Draft Kings"
    assert p["home_spread"] == -14.5 and p["total"] == 221.5
    assert p["home_ml"] == -1040 and p["away_ml"] == 670
    assert p["home_spread_price"] == -114 and p["away_spread_price"] == -106
    assert p["over_price"] == -110 and p["under_price"] == -110
    assert isinstance(p["home_spread_price"], int)


def test_summary_without_pickcenter(summary_no_odds):
    assert parse_pickcenter(summary_no_odds) is None
    assert parse_pickcenter({}) is None


def test_parse_pickcenter_missing_fields():
    s = {"pickcenter": [{"provider": {"name": "Draft Kings"}, "spread": -3.5,
                         "homeTeamOdds": {"moneyLine": -160}, "awayTeamOdds": {}}]}
    p = parse_pickcenter(s)
    assert p["home_spread"] == -3.5 and p["home_ml"] == -160
    assert p["away_ml"] is None and p["total"] is None and p["over_price"] is None
    # entry with no numbers at all is treated as no line
    assert parse_pickcenter({"pickcenter": [{"provider": {"name": "x"}}]}) is None


def test_parse_scoreboard_eastern_dates():
    ev = parse_scoreboard(load_fixture("espn_scoreboard_20261110.json"))
    assert len(ev) == 6
    first = next(e for e in ev if e["espn_event_id"] == "401909960")
    # tip 2026-11-11T00:30Z is 7:30pm ET on 2026-11-10
    assert first == dict(espn_event_id="401909960", game_date="2026-11-10",
                         home_espn_abbr="NY", away_espn_abbr="MIL", completed=True)
    assert all(e["game_date"] == "2026-11-10" for e in ev)  # incl. 04:00Z OKC @ POR


def test_ingest_matches_and_is_resumable(conn, fake_espn, seeded_games):
    n1 = ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    calls = fake_espn.summary_calls
    n2 = ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    assert n1 == 1 and n2 == 0
    assert calls == 2 and fake_espn.summary_calls == calls  # second run fetched nothing
    snap = conn.execute("select * from odds_snapshots").fetchall()
    assert len(snap) == 1
    s = dict(snap[0])
    assert s["game_id"] == "0022600211" and s["source"] == "espn" and s["is_closing"] == 1
    assert s["pulled_at"] == "2026-11-10T23:59:59Z" and s["book"] == "Draft Kings"
    assert (s["home_spread"], s["total"], s["home_ml"], s["away_ml"]) == (-14.5, 221.5, -1040, 670)
    ids = dict(conn.execute("select game_id, espn_event_id from games").fetchall())
    assert ids == {"0022600211": "401909960", "0022600212": "401909961"}  # no-line game still linked


def test_only_missing_false_refetches(conn, fake_espn, seeded_games):
    ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    n = ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10), only_missing=False)
    assert n == 1 and fake_espn.summary_calls == 4
    assert conn.execute("select count(*) from odds_snapshots").fetchone()[0] == 1  # upserted


def test_dates_without_db_games_make_no_calls(conn, fake_espn):
    assert ingest_espn_lines(conn, fake_espn, date(2026, 7, 1), date(2026, 7, 3)) == 0
    assert fake_espn.events_calls == 0 and fake_espn.summary_calls == 0


def test_incomplete_event_skipped(conn, fake_espn, seeded_games):
    fake_espn._events = [{**e, "completed": False} for e in fake_espn._events]
    assert ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10)) == 0
    assert fake_espn.summary_calls == 0
    assert conn.execute("select count(*) from games where espn_event_id is not null").fetchone()[0] == 0


@responses.activate
def test_client_urls(monkeypatch):
    from nba_predict.data import espn_client

    monkeypatch.setattr(espn_client.time, "sleep", lambda s: None)
    responses.get(f"{BASE}/scoreboard", json=load_fixture("espn_scoreboard_20261110.json"),
                  match=[responses.matchers.query_param_matcher({"dates": "20261110"})])
    responses.get(f"{BASE}/summary", json=load_fixture("espn_summary_with_odds.json"),
                  match=[responses.matchers.query_param_matcher({"event": "401909960"})])
    c = EspnClient(delay_s=1.0)
    assert len(c.events(date(2026, 11, 10))) == 6
    assert parse_pickcenter(c.summary("401909960"))["home_ml"] == -1040
