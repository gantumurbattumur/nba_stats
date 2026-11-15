import sqlite3

import pytest
from typer.testing import CliRunner

from nba_predict import cli
from nba_predict.config import get_settings


@pytest.fixture
def cli_db(tmp_path, monkeypatch):
    db = tmp_path / "cli.db"
    monkeypatch.setenv("NBA_DB_PATH", str(db))
    get_settings.cache_clear()
    yield db
    get_settings.cache_clear()


def test_init_db_uses_settings_path(cli_db):
    r = CliRunner().invoke(cli.app, ["init-db"])
    assert r.exit_code == 0 and str(cli_db) in r.output
    assert sqlite3.connect(cli_db).execute("select count(*) from games").fetchone()[0] == 0


def test_slate_command_and_clean_error(cli_db, monkeypatch, fake_client):
    from nba_predict.data.nba_client import IngestError

    fake_client.slate_rows = [dict(game_id="0022600240", game_date="2026-11-15",
                                   home_team_id=1610612738, away_team_id=1610612742,
                                   status="scheduled")]
    monkeypatch.setattr(cli, "NbaClient", lambda **k: fake_client)
    r = CliRunner().invoke(cli.app, ["slate", "--date", "2026-11-15"])
    assert r.exit_code == 0 and "1 games on slate" in r.output and "scheduled=1" in r.output

    class Down:
        def slate(self, d):
            raise IngestError("ScoreboardV2 failed after 3 attempts")

    monkeypatch.setattr(cli, "NbaClient", lambda **k: Down())
    r = CliRunner().invoke(cli.app, ["slate", "--date", "2026-11-15"])
    assert r.exit_code == 1 and "error:" in r.output and "Traceback" not in r.output


def test_espn_command_no_games_makes_no_calls(cli_db, monkeypatch):
    from nba_predict.data import espn_client

    class NoNet:
        def __init__(self, **k): pass
        def events(self, d): pytest.fail("no HTTP expected")
        def summary(self, e): pytest.fail("no HTTP expected")

    monkeypatch.setattr(espn_client, "EspnClient", NoNet)
    r = CliRunner().invoke(cli.app, ["espn", "--from", "2026-07-01", "--to", "2026-07-02"])
    assert r.exit_code == 0 and "wrote 0 ESPN snapshots" in r.output


def test_ingest_season_type_option(cli_db, monkeypatch, fake_client):
    monkeypatch.setattr(cli, "NbaClient", lambda **k: fake_client)
    r = CliRunner().invoke(cli.app, ["ingest", "--season", "2025-26", "--season-type", "playoffs"])
    assert r.exit_code == 0, r.output
    assert fake_client.log_calls == [("2025-26", "Playoffs")]
    r = CliRunner().invoke(cli.app, ["ingest", "--season", "2025-26"])
    assert r.exit_code == 0 and "regular=3" in r.output
    assert [t for _, t in fake_client.log_calls[1:]] == ["Regular Season", "PlayIn", "Playoffs"]
    r = CliRunner().invoke(cli.app, ["ingest", "--season-type", "cup"])
    assert r.exit_code == 2 and "unknown --season-type" in r.output


def test_coverage_and_status(cli_db, conn):
    from nba_predict.db import upsert
    from nba_predict.ingest import coverage

    g = dict(season="2026-27", game_date="2026-11-10", home_team_id=1, away_team_id=2,
             status="final", home_pts=100, away_pts=90)
    upsert(conn, "games", [
        {**g, "game_id": "a", "espn_event_id": "1"},
        {**g, "game_id": "b", "espn_event_id": "2"},
        {**g, "game_id": "c"},
        {**g, "game_id": "d", "status": "scheduled", "home_pts": None, "away_pts": None},
        {**g, "game_id": "e", "season": "2025-26", "season_type": "playoffs"},
    ], ["game_id"])
    upsert(conn, "team_games", [dict(game_id="a", team_id=1, season="2026-27"),
                                dict(game_id="a", team_id=2, season="2026-27")],
           ["game_id", "team_id"])
    snap = dict(source="espn", book="DK", pulled_at="t", is_closing=1)
    upsert(conn, "odds_snapshots", [
        {**snap, "game_id": "a", "home_spread": -3.5, "home_ml": -160, "away_ml": 135},  # agrees
        {**snap, "game_id": "b", "home_spread": 1.5, "home_ml": -101, "away_ml": -117},  # agrees
        {**snap, "game_id": "c", "home_spread": 1.5, "home_ml": -130, "away_ml": 110},   # disagrees
    ], ["game_id", "source", "book", "pulled_at"])
    df = coverage(conn).set_index(["season", "season_type"])
    r = df.loc[("2026-27", "regular")]
    assert (r.final_games, r.scheduled, r.team_games, r.espn_linked, r.espn_lines,
            r.fav_disagree) == (3, 1, 2, 2, 3, 1)
    assert df.loc[("2025-26", "playoffs")].final_games == 1


def test_status_command(cli_db):
    r = CliRunner().invoke(cli.app, ["init-db"])
    r = CliRunner().invoke(cli.app, ["status"])
    assert r.exit_code == 0 and "fav_disagree" in r.output and str(cli_db) in r.output


def test_features_command_always_rebuilds(cli_db, monkeypatch):
    import sqlite3 as sq

    import conftest
    from nba_predict.analysis import enrich
    from nba_predict.db import connect, init_schema
    from nba_predict.features import build

    c = connect(cli_db)
    init_schema(c)
    conftest.build_league(c)
    c.close()
    calls = []
    real_e, real_f = enrich.build_enriched, build.build_feature_table
    monkeypatch.setattr(enrich, "build_enriched", lambda conn: calls.append("e") or real_e(conn))
    monkeypatch.setattr(build, "build_feature_table",
                        lambda conn, w=(5, 8, 10, 15): calls.append(("f", w)) or real_f(conn, w))
    for _ in range(2):
        r = CliRunner().invoke(cli.app, ["features", "--windows", "5,10"])
        assert r.exit_code == 0, r.output
        assert "game_features: 50 rows" in r.output
    assert calls == ["e", ("f", (5, 10)), "e", ("f", (5, 10))]
    assert "h_all_l10_net" in {r[1] for r in sq.connect(cli_db).execute("pragma table_info(game_features)")}


def test_backtest_command(cli_db):
    import conftest
    from nba_predict.db import connect, init_schema

    c = connect(cli_db)
    init_schema(c)
    conftest.build_league(c)
    c.close()
    r = CliRunner().invoke(cli.app, ["backtest", "--season", "2025-26", "--n-boot", "20"])
    assert r.exit_code == 0, r.output
    assert "== 2025-26 (test) ==" in r.output and "log_loss" in r.output and "best model" in r.output
    assert "by segment" in r.output
