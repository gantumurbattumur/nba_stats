import logging

import pytest
import responses

from conftest import load_fixture
from nba_predict.data.odds_client import URL, OddsApiClient, OddsApiError, parse_events
from nba_predict.db import upsert
from nba_predict.ingest import ingest_odds

HDR = {"x-requests-used": "33", "x-requests-remaining": "467", "x-requests-last": "3"}
BOS, DAL, SAC, OKC = 1610612738, 1610612742, 1610612758, 1610612760


def _mock(status=200, headers=HDR, body=None):
    responses.get(URL, json=load_fixture("oddsapi_nba.json") if body is None else body,
                  status=status, headers=headers)


@responses.activate
def test_request_params():
    _mock()
    events, usage = OddsApiClient("k3y").nba_odds()
    q = responses.calls[0].request
    assert q.url.split("?")[0] == "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    assert "historical" not in q.url
    from urllib.parse import parse_qs, urlparse
    assert parse_qs(urlparse(q.url).query) == {"apiKey": ["k3y"], "regions": ["us"],
                                              "markets": ["h2h,spreads,totals"], "oddsFormat": ["american"]}
    assert len(events) == 3 and usage == {"used": 33, "remaining": 467, "last": 3}


def test_parse_events():
    rows = parse_events(load_fixture("oddsapi_nba.json"))
    assert len(rows) == 5  # (event, book) pairs
    bos_dk = next(r for r in rows if r["event_id"] == "e1bos" and r["book"] == "draftkings")
    assert (bos_dk["home_ml"], bos_dk["away_ml"]) == (-180, 155)
    assert bos_dk["home_spread"] == -4.5 and (bos_dk["home_spread_price"], bos_dk["away_spread_price"]) == (-110, -110)
    assert (bos_dk["total"], bos_dk["over_price"], bos_dk["under_price"]) == (224.5, -110, -110)
    sac = next(r for r in rows if r["event_id"] == "e2sac" and r["book"] == "fanduel")
    assert sac["home_spread"] == 7.0 and sac["home_name"] == "Sacramento Kings"  # home dog: positive
    assert sac["commence_time"] == "2026-11-16T03:30:00Z"


def test_parse_tolerates_missing_markets():
    ev = load_fixture("oddsapi_nba.json")[0]
    ev["bookmakers"][0]["markets"] = [m for m in ev["bookmakers"][0]["markets"] if m["key"] == "totals"]
    r = parse_events([ev])[0]
    assert r["home_ml"] is None and r["home_spread"] is None and r["total"] == 224.5


@pytest.fixture
def slate(conn):
    upsert(conn, "games", [
        dict(game_id="g_bos", season="2026-27", game_date="2026-11-15", home_team_id=BOS,
             away_team_id=DAL, status="scheduled"),
        dict(game_id="g_sac", season="2026-27", game_date="2026-11-15", home_team_id=SAC,
             away_team_id=OKC, status="scheduled"),
    ], ["game_id"])
    return conn


@responses.activate
def test_late_tip_matches_eastern_date(slate):
    _mock()
    n = ingest_odds(slate, OddsApiClient("k"))
    sac = slate.execute("select count(*) from odds_snapshots where game_id='g_sac'").fetchone()[0]
    assert sac == 2 and n == 4


@responses.activate
def test_unmapped_team_skipped_with_warning(slate, caplog):
    _mock()
    with caplog.at_level(logging.WARNING):
        ingest_odds(slate, OddsApiClient("k"))
    assert "Seattle SuperSonics" in caplog.text
    assert slate.execute("select count(distinct game_id) from odds_snapshots").fetchone()[0] == 2


@responses.activate
def test_unmatched_game_skipped(conn, caplog):
    _mock()
    with caplog.at_level(logging.WARNING):
        assert ingest_odds(conn, OddsApiClient("k")) == 0  # no games in DB
    assert "no game" in caplog.text


@responses.activate
def test_snapshot_fields_and_closing_flag(slate):
    _mock()
    ingest_odds(slate, OddsApiClient("k"), closing=True)
    rows = [dict(r) for r in slate.execute("select * from odds_snapshots order by game_id, book")]
    assert {r["source"] for r in rows} == {"oddsapi"} and {r["is_closing"] for r in rows} == {1}
    assert len({r["pulled_at"] for r in rows}) == 1  # one shared pull time
    dk = next(r for r in rows if r["game_id"] == "g_bos" and r["book"] == "draftkings")
    assert (dk["home_ml"], dk["home_spread"], dk["total"]) == (-180, -4.5, 224.5)


@responses.activate
@pytest.mark.parametrize("status,headers,body", [
    (429, HDR, {"message": "rate limited"}),
    (401, {**HDR, "x-requests-remaining": "0"}, {"message": "Usage quota has been reached",
                                                  "error_code": "OUT_OF_USAGE_CREDITS"}),
])
def test_quota_exhausted_raises_clean_error(status, headers, body):
    _mock(status, headers, body)
    with pytest.raises(OddsApiError, match="quota exhausted"):
        OddsApiClient("k").nba_odds()


@responses.activate
def test_invalid_key():
    _mock(401, HDR, {"message": "API key is not valid", "error_code": "INVALID_KEY"})
    with pytest.raises(OddsApiError, match="invalid key"):
        OddsApiClient("bad").nba_odds()


def test_missing_key_makes_no_call():
    with responses.RequestsMock() as rs:
        with pytest.raises(OddsApiError, match="ODDS_API_KEY not set"):
            OddsApiClient(None).nba_odds()
        assert len(rs.calls) == 0


@responses.activate
def test_usage_logged(slate):
    _mock()
    ingest_odds(slate, OddsApiClient("k"))
    r = slate.execute("select source, credits_used, credits_remaining from api_usage").fetchall()
    assert [tuple(x) for x in r] == [("oddsapi", 3, 467)]
