from datetime import date

import pytest
import requests

from nba_predict.data import nba_client
from nba_predict.data.nba_client import IngestError, NbaClient
from nba_predict.ingest import ingest_season, ingest_slate, parse_team_logs


def test_parse_team_logs(sample_logs):
    games, tgs = parse_team_logs(sample_logs, "2025-26")
    assert len(games) == 3 and len(tgs) == 6
    g = next(x for x in games if x["game_id"] == sample_logs.GAME_ID[0])
    home = next(t for t in tgs if t["game_id"] == g["game_id"] and t["is_home"] == 1)
    assert g["home_team_id"] == home["team_id"] and g["home_pts"] == home["pts"]
    assert all(t["opp_team_id"] != t["team_id"] for t in tgs)


def test_parse_team_logs_values(sample_logs):
    games, tgs = parse_team_logs(sample_logs, "2025-26")
    g = next(x for x in games if x["game_id"] == "0022500001")
    assert g == dict(game_id="0022500001", season="2025-26", game_date="2025-10-21",
                     home_team_id=1610612738, away_team_id=1610612752, home_pts=116,
                     away_pts=106, status="final")
    away = next(t for t in tgs if t["game_id"] == "0022500001" and t["is_home"] == 0)
    assert away["team_id"] == 1610612752 and away["opp_team_id"] == 1610612738
    assert away["wl"] == "L" and away["pts"] == 106 and away["plus_minus"] == -10
    assert isinstance(away["fga"], int)


def test_parse_skips_incomplete_game(sample_logs):
    # a game with only one team row (e.g. mid-ingest) is not emitted
    games, tgs = parse_team_logs(sample_logs.iloc[1:], "2025-26")
    assert {g["game_id"] for g in games} == {"0022500002", "0022500003"} and len(tgs) == 4


def test_ingest_season_idempotent(conn, fake_client):
    assert ingest_season(conn, fake_client, "2025-26") == 6
    ingest_season(conn, fake_client, "2025-26")
    assert conn.execute("select count(*) from team_games").fetchone()[0] == 6
    assert conn.execute("select count(*) from games where status='final'").fetchone()[0] == 3


def test_slate_does_not_downgrade_final(conn, fake_client):
    ingest_season(conn, fake_client, "2025-26")
    gid = "0022500003"
    fake_client.slate_rows = [dict(game_id=gid, game_date="2025-11-01", home_team_id=1610612752,
                                   away_team_id=1610612747, status="scheduled")]
    ingest_slate(conn, fake_client, date(2025, 11, 1))
    row = conn.execute("select status, home_pts from games where game_id=?", (gid,)).fetchone()
    assert tuple(row) == ("final", 116)


def test_slate_inserts_scheduled(conn, fake_client):
    fake_client.slate_rows = [dict(game_id="0022500099", game_date="2025-11-02",
                                   home_team_id=1610612738, away_team_id=1610612743,
                                   status="scheduled")]
    assert ingest_slate(conn, fake_client, date(2025, 11, 2)) == 1
    row = conn.execute("select season, status, espn_event_id from games where game_id='0022500099'").fetchone()
    assert tuple(row) == ("2025-26", "scheduled", None)


def test_client_retries_then_raises(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    class Boom:
        def __init__(self, *a, **k):
            calls["n"] += 1
            raise requests.Timeout("timed out")

    monkeypatch.setattr(nba_client.leaguegamelog, "LeagueGameLog", Boom)
    monkeypatch.setattr(nba_client.time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(IngestError):
        NbaClient(delay_s=0.6).season_team_logs("2025-26")
    assert calls["n"] == 3
    assert [s for s in sleeps if s >= 1] == [1, 2]   # backoff between attempts


def test_client_recovers_after_one_failure(monkeypatch, sample_logs):
    calls = {"n": 0}

    class Flaky:
        def __init__(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.ConnectionError("reset")

        def get_data_frames(self):
            return [sample_logs]

    monkeypatch.setattr(nba_client.leaguegamelog, "LeagueGameLog", Flaky)
    monkeypatch.setattr(nba_client.time, "sleep", lambda s: None)
    assert len(NbaClient(delay_s=0).season_team_logs("2025-26")) == 6


def test_season_for_date():
    from nba_predict.ingest import season_for_date
    assert season_for_date(date(2026, 11, 15)) == "2026-27"
    assert season_for_date(date(2027, 4, 1)) == "2026-27"
    assert season_for_date(date(2025, 10, 21)) == "2025-26"


def _neutralize(logs, gid="0022500002"):
    """Mexico City-style game: both MATCHUP strings use '@' (seen live: 0022600189 DEN/IND)."""
    logs = logs.copy()
    m = logs.GAME_ID == gid
    logs.loc[m & (logs.TEAM_ABBREVIATION == "DEN"), "MATCHUP"] = "DEN @ LAL"
    return logs


def test_parse_uses_home_ids_for_neutral_site(sample_logs):
    logs = _neutralize(sample_logs)
    games, _ = parse_team_logs(logs, "2025-26")
    assert "0022500002" not in {g["game_id"] for g in games}
    games, tgs = parse_team_logs(logs, "2025-26", home_ids={"0022500002": 1610612743})
    g = next(x for x in games if x["game_id"] == "0022500002")
    assert g["home_team_id"] == 1610612743 and g["home_pts"] == 121 and len(tgs) == 6


def test_ingest_season_resolves_neutral_site_via_slate(conn, fake_client):
    fake_client.logs = _neutralize(fake_client.logs)
    fake_client.slate_rows = [dict(game_id="0022500002", game_date="2025-10-22",
                                   home_team_id=1610612743, away_team_id=1610612747, status="final")]
    assert ingest_season(conn, fake_client, "2025-26") == 6
    row = conn.execute("select home_team_id from games where game_id='0022500002'").fetchone()
    assert row[0] == 1610612743
