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
                     away_pts=106, status="final", season_type="regular")
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


def test_reingest_keeps_espn_event_id(conn, fake_client):
    ingest_season(conn, fake_client, "2025-26")
    conn.execute("update games set espn_event_id='401' where game_id='0022500001'")
    conn.commit()
    ingest_season(conn, fake_client, "2025-26")
    fake_client.slate_rows = [dict(game_id="0022500001", game_date="2025-10-21",
                                   home_team_id=1610612738, away_team_id=1610612752, status="final")]
    ingest_slate(conn, fake_client, date(2025, 10, 21))
    r = conn.execute("select status, espn_event_id from games where game_id='0022500001'").fetchone()
    assert tuple(r) == ("final", "401")


def test_slate_final_before_log_then_season_promotes(conn, fake_client):
    # scoreboard says final but the box score isn't in the log yet → scheduled (no scores);
    # the next ingest_season run promotes it to final with scores
    fake_client.slate_rows = [dict(game_id="0022500003", game_date="2025-11-01",
                                   home_team_id=1610612752, away_team_id=1610612747, status="final")]
    ingest_slate(conn, fake_client, date(2025, 11, 1))
    r = conn.execute("select status, home_pts from games where game_id='0022500003'").fetchone()
    assert tuple(r) == ("scheduled", None)
    ingest_season(conn, fake_client, "2025-26")
    r = conn.execute("select status, home_pts from games where game_id='0022500003'").fetchone()
    assert tuple(r) == ("final", 116)


def test_slate_postponed_then_rescheduled(conn, fake_client):
    base = dict(game_id="0022500050", home_team_id=1610612738, away_team_id=1610612743)
    fake_client.slate_rows = [dict(base, game_date="2025-12-01", status="postponed")]
    ingest_slate(conn, fake_client, date(2025, 12, 1))
    fake_client.slate_rows = [dict(base, game_date="2026-01-15", status="scheduled")]
    ingest_slate(conn, fake_client, date(2026, 1, 15))
    r = conn.execute("select season, game_date, status from games where game_id='0022500050'").fetchone()
    assert tuple(r) == ("2025-26", "2026-01-15", "scheduled")


def test_parse_skips_in_progress_game(sample_logs):
    logs = sample_logs.copy()
    logs.loc[logs.GAME_ID == "0022500002", "WL"] = None
    games, tgs = parse_team_logs(logs, "2025-26")
    assert "0022500002" not in {g["game_id"] for g in games} and len(tgs) == 4


def test_status_mapping():
    from nba_predict.data.nba_client import _status
    assert _status(3, "Final") == "final"
    assert _status(3, "Final/OT") == "final"
    assert _status(1, "7:30 pm ET") == "scheduled"
    assert _status(2, "Q3 5:12") == "scheduled"
    assert _status(1, "PPD") == "postponed" and _status(1, "Postponed") == "postponed"


def test_live_slate_falls_back_when_board_is_stale(monkeypatch):
    # before the live board rolls over (late morning ET) it still shows yesterday
    monkeypatch.setattr(nba_client.time, "sleep", lambda s: None)
    monkeypatch.setattr(nba_client, "today_et", lambda: date(2026, 11, 15))

    class Board:
        def __init__(self, *a, **k): pass
        def get_dict(self):
            return {"scoreboard": {"gameDate": "2026-11-14", "games": [
                {"gameId": "old", "homeTeam": {"teamId": 1}, "awayTeam": {"teamId": 2},
                 "gameStatus": 3, "gameStatusText": "Final"}]}}

    monkeypatch.setattr(nba_client.scoreboard, "ScoreBoard", Board)
    v2 = [dict(game_id="new", game_date="2026-11-15", home_team_id=1, away_team_id=2,
               status="scheduled")]
    monkeypatch.setattr(NbaClient, "_v2_slate", lambda self, d: v2)
    assert NbaClient(delay_s=0).slate(date(2026, 11, 15)) == v2


def test_live_slate_used_for_today(monkeypatch):
    monkeypatch.setattr(nba_client.time, "sleep", lambda s: None)
    monkeypatch.setattr(nba_client, "today_et", lambda: date(2026, 11, 15))

    class Board:
        def __init__(self, *a, **k): pass
        def get_dict(self):
            return {"scoreboard": {"gameDate": "2026-11-15", "games": [
                {"gameId": "0022600240", "homeTeam": {"teamId": 1610612738},
                 "awayTeam": {"teamId": 1610612742}, "gameStatus": 1,
                 "gameStatusText": "7:30 pm ET"}]}}

    monkeypatch.setattr(nba_client.scoreboard, "ScoreBoard", Board)
    monkeypatch.setattr(NbaClient, "_v2_slate", lambda self, d: pytest.fail("V2 not expected"))
    assert NbaClient(delay_s=0).slate(date(2026, 11, 15)) == [dict(
        game_id="0022600240", game_date="2026-11-15", home_team_id=1610612738,
        away_team_id=1610612742, status="scheduled")]


def test_slate_skips_preseason_and_all_star(conn, fake_client):
    # LeagueGameLog (regular season) never finalizes these, so they'd sit 'scheduled' forever
    mk = lambda gid: dict(game_id=gid, game_date="2026-10-10", home_team_id=1610612738,
                          away_team_id=1610612752, status="final")
    fake_client.slate_rows = [mk("0012600001"), mk("0032600001"), mk("0022600001")]
    ingest_slate(conn, fake_client, date(2026, 10, 10))
    assert [r[0] for r in conn.execute("select game_id from games")] == ["0022600001"]


# --- season types (regular / play-in / playoffs) ---------------------------------------------

def _playoff_logs(sample_logs, prefix="004", gid_suffix="00101"):
    """Re-label sample game 0022500001 (NYK @ BOS) as a post-season game."""
    g = sample_logs[sample_logs.GAME_ID == "0022500001"].copy()
    g["GAME_ID"] = prefix + "25" + gid_suffix
    g["GAME_DATE"] = "2026-04-20"
    return g


def test_ingest_season_pulls_all_season_types(conn, fake_client, sample_logs):
    fake_client.other_logs = {"PlayIn": _playoff_logs(sample_logs, "005"),
                              "Playoffs": _playoff_logs(sample_logs, "004")}
    assert ingest_season(conn, fake_client, "2025-26") == 10
    assert [t for _, t in fake_client.log_calls] == ["Regular Season", "PlayIn", "Playoffs"]
    got = dict(conn.execute("select game_id, season_type from games").fetchall())
    assert got["0022500001"] == "regular" and got["0052500101"] == "playin"
    assert got["0042500101"] == "playoffs"
    tg = conn.execute("select season_type, count(*) from team_games group by 1 order by 1").fetchall()
    assert [tuple(r) for r in tg] == [("playin", 2), ("playoffs", 2), ("regular", 6)]


def test_ingest_season_types_subset_and_empty_logs(conn, fake_client):
    # no post-season yet: empty frames from nba_api must not crash
    assert ingest_season(conn, fake_client, "2025-26") == 6
    fake_client.log_calls.clear()
    assert ingest_season(conn, fake_client, "2025-26", season_types=("Regular Season",)) == 6
    assert fake_client.log_calls == [("2025-26", "Regular Season")]


def test_parse_sets_season_type(sample_logs):
    games, tgs = parse_team_logs(sample_logs, "2025-26", season_type="playoffs")
    assert {g["season_type"] for g in games} == {"playoffs"}
    assert {t["season_type"] for t in tgs} == {"playoffs"}
    games, _ = parse_team_logs(sample_logs, "2025-26")
    assert {g["season_type"] for g in games} == {"regular"}


def test_slate_sets_season_type_from_game_id(conn, fake_client):
    base = dict(game_date="2026-04-20", home_team_id=1610612738, away_team_id=1610612752,
                status="scheduled")
    fake_client.slate_rows = [{**base, "game_id": "0042500101"},
                              {**base, "game_id": "0052500101", "home_team_id": 1610612743},
                              {**base, "game_id": "0062500001", "home_team_id": 1610612747}]
    assert ingest_slate(conn, fake_client, date(2026, 4, 20)) == 2  # Cup final (006) skipped
    got = dict(conn.execute("select game_id, season_type from games").fetchall())
    assert got == {"0042500101": "playoffs", "0052500101": "playin"}


def test_playoff_slate_game_becomes_final_after_ingest(conn, fake_client, sample_logs):
    fake_client.slate_rows = [dict(game_id="0042500101", game_date="2026-04-20",
                                   home_team_id=1610612738, away_team_id=1610612752,
                                   status="scheduled")]
    ingest_slate(conn, fake_client, date(2026, 4, 20))
    fake_client.other_logs = {"Playoffs": _playoff_logs(sample_logs, "004")}
    ingest_season(conn, fake_client, "2025-26")
    row = conn.execute("select status, season_type, home_pts from games where game_id='0042500101'").fetchone()
    assert tuple(row) == ("final", "playoffs", 116)


def test_client_passes_season_type(monkeypatch, sample_logs):
    seen = {}

    class Rec:
        def __init__(self, **k):
            seen.update(k)

        def get_data_frames(self):
            return [sample_logs]

    monkeypatch.setattr(nba_client.leaguegamelog, "LeagueGameLog", Rec)
    monkeypatch.setattr(nba_client.time, "sleep", lambda s: None)
    NbaClient(delay_s=0).season_team_logs("2024-25", season_type="PlayIn")
    assert seen["season_type_all_star"] == "PlayIn" and seen["season"] == "2024-25"
