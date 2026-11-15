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
