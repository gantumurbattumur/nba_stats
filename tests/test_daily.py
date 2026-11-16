import pandas as pd
from typer.testing import CliRunner

import conftest
from nba_predict import cli
from nba_predict.config import get_settings
from nba_predict.db import connect, init_schema


class _NoNba:
    def season_team_logs(self, season, season_type="Regular Season"):
        return pd.DataFrame()

    def slate(self, d):
        return []


class _NoEspn:
    def scoreboard(self, d):
        return {"events": []}

    def events(self, d):
        return []

    def summary(self, e):
        raise AssertionError("no summaries expected")


def test_daily_without_odds_key(tmp_path, monkeypatch):
    db = tmp_path / "daily.db"
    monkeypatch.setenv("NBA_DB_PATH", str(db))
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("NBA_ODDS_API_KEY", raising=False)
    from nba_predict.config import Settings
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / ".env")
    get_settings.cache_clear()
    c = connect(db)
    init_schema(c)
    conftest.build_league(c)
    c.close()
    monkeypatch.setattr(cli, "NbaClient", lambda **k: _NoNba())
    import nba_predict.data.espn_client as ec
    monkeypatch.setattr(ec, "EspnClient", lambda **k: _NoEspn())
    try:
        r = CliRunner().invoke(cli.app, ["daily", "--date", "2025-11-10"])
    finally:
        get_settings.cache_clear()
    assert r.exit_code == 0, r.output
    assert "ODDS_API_KEY" in r.output and "flags skipped" in r.output
    assert "NYK @ BOS" in r.output and "DEN" in r.output
    c = connect(db)
    n = c.execute("select count(*) from predictions p join games g using (game_id) "
                  "where g.game_date = '2025-11-10'").fetchone()[0]
    assert n == 6
    assert c.execute("select count(*) from value_flags").fetchone()[0] == 0
