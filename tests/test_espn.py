import json
from datetime import date

import pytest
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


def test_spread_is_home_perspective_when_home_is_underdog():
    # home underdog: ESPN spread is positive (home +4.5), home ML positive
    s = {"pickcenter": [{"provider": {"name": "Draft Kings"}, "spread": 4.5, "overUnder": 230.0,
                         "homeTeamOdds": {"moneyLine": 160, "spreadOdds": -110, "underdog": True},
                         "awayTeamOdds": {"moneyLine": -190, "spreadOdds": -110, "favorite": True}}]}
    p = parse_pickcenter(s)
    assert p["home_spread"] == 4.5 and p["home_ml"] == 160 and p["away_ml"] == -190


def test_pickem_spread_zero_is_not_missing():
    s = {"pickcenter": [{"provider": {"name": "x"}, "spread": 0, "overUnder": 220,
                         "homeTeamOdds": {"moneyLine": -110}, "awayTeamOdds": {"moneyLine": -110}}]}
    assert parse_pickcenter(s)["home_spread"] == 0.0


def test_invalid_prices_become_none():
    # ESPN occasionally serves 0 / blank prices; they must not reach odds_math as real odds
    s = {"pickcenter": [{"provider": {"name": "x"}, "spread": -2.5, "overOdds": 0, "underOdds": "",
                         "homeTeamOdds": {"moneyLine": 0, "spreadOdds": "EVEN"},
                         "awayTeamOdds": {"moneyLine": -50}}]}
    p = parse_pickcenter(s)
    assert p["home_spread"] == -2.5
    assert p["home_ml"] is None and p["away_ml"] is None and p["over_price"] is None
    assert p["under_price"] is None


@pytest.mark.parametrize("iso,et", [
    ("2026-11-11T03:30Z", "2026-11-10"),   # 10:30pm EST tip, next day UTC
    ("2026-11-11T04:59Z", "2026-11-10"),
    ("2026-11-11T05:00Z", "2026-11-11"),   # midnight EST
    ("2026-10-22T03:30Z", "2026-10-21"),   # EDT (UTC-4): 11:30pm
    ("2026-10-22T04:00Z", "2026-10-22"),   # EDT midnight
    ("2026-11-02T04:30Z", "2026-11-01"),   # day after DST ends (EST, UTC-5)
])
def test_et_date_boundaries(iso, et):
    from nba_predict.data.espn_client import _et_date
    assert _et_date(iso) == et


@responses.activate
def test_client_retries_then_raises(monkeypatch):
    from nba_predict.data import espn_client
    from nba_predict.data.nba_client import IngestError

    monkeypatch.setattr(espn_client.time, "sleep", lambda s: None)
    responses.get(f"{BASE}/summary", status=503)
    with pytest.raises(IngestError):
        EspnClient(delay_s=0).summary("1")
    assert len(responses.calls) == 3


def test_ingest_warns_when_espn_home_away_flipped(conn, fake_espn, seeded_games, caplog):
    # ESPN lists PHI @ BOS while the NBA has BOS @ PHI (neutral-site style disagreement):
    # must not silently drop the game, and must not attach a line with the wrong sign
    fake_espn._events = [{**e, "home_espn_abbr": "BOS", "away_espn_abbr": "PHI"}
                         if e["espn_event_id"] == "401909961" else e for e in fake_espn._events]
    with caplog.at_level("WARNING"):
        ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    assert any("401909961" in r.getMessage() and "home/away" in r.getMessage() for r in caplog.records)
    row = conn.execute("select espn_event_id from games where game_id='0022600212'").fetchone()
    assert row[0] is None


@responses.activate
def test_client_404_is_not_retried(monkeypatch):
    from nba_predict.data import espn_client
    from nba_predict.data.espn_client import EspnNotFound

    monkeypatch.setattr(espn_client.time, "sleep", lambda s: None)
    responses.get(f"{BASE}/summary", status=404)
    with pytest.raises(EspnNotFound):
        EspnClient(delay_s=0).summary("401869391")
    assert len(responses.calls) == 1


def test_summary_404_is_skipped_and_marked(conn, fake_espn, seeded_games, caplog):
    # B, phase 1: a single 404 summary (e.g. 401869391) aborted `nba espn` and resume hit it again
    from nba_predict.data.espn_client import EspnNotFound

    orig = fake_espn.summary

    def summary(event_id):
        if event_id == "401909960":
            raise EspnNotFound("404 Not Found")
        return orig(event_id)

    fake_espn.summary = summary
    with caplog.at_level("WARNING"):
        n = ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    assert n == 0 and any("401909960" in r.getMessage() for r in caplog.records)
    ids = dict(conn.execute("select game_id, espn_event_id from games").fetchall())
    assert ids == {"0022600211": "401909960", "0022600212": "401909961"}  # both linked
    calls = fake_espn.summary_calls
    ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    assert fake_espn.summary_calls == calls  # resume skips the 404 game


@pytest.mark.parametrize("status,calls", [(400, 1), (403, 1), (429, 3), (500, 3)])
@responses.activate
def test_client_retries_only_transient_errors(monkeypatch, status, calls):
    from nba_predict.data import espn_client
    from nba_predict.data.nba_client import IngestError

    monkeypatch.setattr(espn_client.time, "sleep", lambda s: None)
    responses.get(f"{BASE}/summary", status=status)
    with pytest.raises(IngestError):
        EspnClient(delay_s=0).summary("1")
    assert len(responses.calls) == calls


# --- ESPN core odds API fallback (open + close lines) -------------------------------------------

from nba_predict.data.espn_client import CORE_URL, EspnNotFound, parse_core_odds  # noqa: E402

CORE_LAL = "espn_core_odds_401809244.json"  # GSW @ LAL 2025-10-21, ESPN BET + a live entry
CORE_NYK = "espn_core_odds_401909960.json"  # MIL @ NYK 2026-11-10, Draft Kings


def test_parse_core_odds_close_and_open():
    p = parse_core_odds(load_fixture(CORE_LAL))
    assert p["book"] == "ESPN BET"  # "ESPN Bet - Live Odds" skipped
    assert p["close"] == dict(home_ml=130, away_ml=-150, home_spread=2.5, home_spread_price=-110,
                              away_spread_price=-110, total=227.5, over_price=-110, under_price=-110)
    assert p["open"] == dict(home_ml=115, away_ml=-135, home_spread=2.5, home_spread_price=-120,
                             away_spread_price=100, total=224.5, over_price=-115, under_price=-105)


def test_parse_core_odds_home_favorite_sign():
    p = parse_core_odds(load_fixture(CORE_NYK))
    assert p["book"] == "Draft Kings" and p["close"]["home_spread"] == -14.5
    assert p["close"]["home_ml"] == -1040 and p["open"]["home_spread"] == -15.5
    assert p["open"]["total"] == 219.5


def test_parse_core_odds_live_only_or_empty():
    d = load_fixture(CORE_LAL)
    live = {**d, "items": [i for i in d["items"] if "Live" in i["provider"]["name"]]}
    assert parse_core_odds(live) is None
    assert parse_core_odds({"count": 0, "items": []}) is None


def _snaps(conn, gid):
    return [dict(r) for r in conn.execute(
        "select book, pulled_at, is_closing, home_spread, total, home_ml from odds_snapshots "
        "where game_id = ? order by is_closing", (gid,))]


def test_core_fallback_when_pickcenter_empty(conn, fake_espn, seeded_games):
    fake_espn.core = {"401909961": load_fixture(CORE_LAL)}  # BOS @ PHI is served without pickcenter
    n = ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    assert n == 3  # NYK pickcenter close + PHI core close + PHI core open
    assert fake_espn.core_calls == ["401909961"]  # pickcenter game not fetched from core
    s = _snaps(conn, "0022600212")
    assert s == [dict(book="ESPN BET", pulled_at="2026-11-10T12:00:00Z", is_closing=0, home_spread=2.5,
                      total=224.5, home_ml=115),
                 dict(book="ESPN BET", pulled_at="2026-11-10T23:59:59Z", is_closing=1, home_spread=2.5,
                      total=227.5, home_ml=130)]
    assert ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10)) == 0
    assert fake_espn.core_calls == ["401909961"]


def test_linked_game_without_line_gets_core_once(conn, fake_espn, seeded_games):
    ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))  # PHI linked, no line
    assert fake_espn.core_calls == ["401909961"]  # first run already tried core (empty answer)
    conn.execute("update games set espn_core_checked = 0 where game_id = '0022600212'")
    fake_espn.core = {"401909961": load_fixture(CORE_LAL)}
    ev_calls, sum_calls = fake_espn.events_calls, fake_espn.summary_calls
    assert ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10)) == 2
    assert (fake_espn.events_calls, fake_espn.summary_calls) == (ev_calls, sum_calls)  # core only
    assert ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10)) == 0
    assert fake_espn.core_calls == ["401909961", "401909961"]


def test_with_open_adds_open_for_pickcenter_games(conn, fake_espn, seeded_games):
    ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    fake_espn.core = {"401909960": load_fixture(CORE_NYK)}
    n = ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10), with_open=True)
    assert n == 1
    s = _snaps(conn, "0022600211")
    assert [(x["is_closing"], x["home_spread"]) for x in s] == [(0, -15.5), (1, -14.5)]
    assert ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10), with_open=True) == 0


def test_core_404_marks_checked(conn, fake_espn, seeded_games):
    fake_espn.core = {"401909961": EspnNotFound("HTTP 404")}
    ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    assert conn.execute("select espn_core_checked from games where game_id='0022600212'").fetchone()[0] == 1
    ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    assert fake_espn.core_calls == ["401909961"]


def test_open_lines_do_not_change_closing_views(conn, fake_espn, seeded_games):
    from nba_predict.analysis.enrich import closing_lines
    from nba_predict.flags import market_view

    fake_espn.core = {"401909961": load_fixture(CORE_LAL)}
    ingest_espn_lines(conn, fake_espn, date(2026, 11, 10), date(2026, 11, 10))
    cl = closing_lines(conn).set_index("game_id")
    assert cl.loc["0022600212", "total"] == 227.5
    assert market_view(conn, "0022600212")["total"] == 227.5  # latest pull = close
    assert market_view(conn, "0022600212", closing_only=True)["total"] == 227.5


@responses.activate
def test_client_core_url(monkeypatch):
    from nba_predict.data import espn_client

    monkeypatch.setattr(espn_client.time, "sleep", lambda s: None)
    responses.get(CORE_URL.format(event_id="401809244"), json=load_fixture(CORE_LAL))
    assert EspnClient(delay_s=0).core_odds("401809244")["count"] == 2
    assert responses.calls[0].request.url == (
        "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/401809244/"
        "competitions/401809244/odds")
