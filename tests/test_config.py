import pytest

from nba_predict.config import Settings


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for k in ("ODDS_API_KEY", "NBA_ODDS_API_KEY", "NBA_DB_PATH", "NBA_EDGE_THRESHOLD",
              "NBA_WINDOWS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in cwd
    return tmp_path


def test_defaults(clean_env):
    s = Settings()
    assert s.windows == (5, 8, 10, 15) and s.edge_threshold == 0.03
    assert s.current_season == "2026-27"
    assert s.train_seasons == ("2023-24", "2024-25", "2025-26")
    assert (s.nba_delay_s, s.espn_delay_s, s.db_path) == (0.6, 1.0, "data/nba.db")
    assert s.odds_api_key is None


def test_unprefixed_odds_key_and_prefixed_fields(clean_env, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "abc")
    monkeypatch.setenv("NBA_EDGE_THRESHOLD", "0.05")
    monkeypatch.setenv("NBA_WINDOWS", "[3, 7]")
    s = Settings()
    assert s.odds_api_key == "abc" and s.edge_threshold == 0.05 and s.windows == (3, 7)


def test_dotenv_in_cwd(clean_env):
    (clean_env / ".env").write_text("ODDS_API_KEY=fromfile\nNBA_DB_PATH=x/y.db\n")
    s = Settings()
    assert s.odds_api_key == "fromfile" and s.db_path == "x/y.db"


def test_no_staking_settings():
    # analysis-only scope: no bankroll / stake / kelly knobs anywhere in settings
    banned = ("bankroll", "stake", "kelly", "wager", "bet_size", "unit_size")
    assert not [f for f in Settings.model_fields if any(b in f for b in banned)]
