from pathlib import Path

import pytest

from nba_predict.config import Settings, project_root

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for k in ("ODDS_API_KEY", "NBA_ODDS_API_KEY", "NBA_DB_PATH", "NBA_EDGE_THRESHOLD",
              "NBA_WINDOWS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    # point the project-root .env at an empty temp file so a real .env can't leak in
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / ".env")
    return tmp_path


def test_defaults(clean_env):
    s = Settings()
    assert s.windows == (5, 8, 10, 15, 20, 30) and s.edge_threshold == 0.03
    assert s.current_season == "2026-27"
    assert s.train_seasons == ("2025-26",)
    assert s.model_seasons == ("2025-26", "2026-27")
    assert (s.nba_delay_s, s.espn_delay_s) == (0.6, 1.0)
    assert s.db_path == str(REPO / "data" / "nba.db")  # project root, not cwd (tmp_path)
    assert s.odds_api_key is None


def test_unprefixed_odds_key_and_prefixed_fields(clean_env, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "abc")
    monkeypatch.setenv("NBA_EDGE_THRESHOLD", "0.05")
    monkeypatch.setenv("NBA_WINDOWS", "[3, 7]")
    s = Settings()
    assert s.odds_api_key == "abc" and s.edge_threshold == 0.05 and s.windows == (3, 7)


def test_dotenv_values_loaded(clean_env):
    (clean_env / ".env").write_text("ODDS_API_KEY=fromfile\nNBA_DB_PATH=x/y.db\n")
    s = Settings()
    # relative paths resolve against the project root, not the cwd
    assert s.odds_api_key == "fromfile" and s.db_path == str(REPO / "x" / "y.db")


def test_project_root_and_env_file_location():
    assert project_root() == REPO
    # the class default (before any test monkeypatching) is <root>/.env
    from nba_predict import config
    assert config.ENV_FILE == REPO / ".env"


def test_absolute_db_path_override(clean_env, monkeypatch):
    monkeypatch.setenv("NBA_DB_PATH", str(clean_env / "abs.db"))
    assert Settings().db_path == str(clean_env / "abs.db")


def test_cwd_dotenv_ignored(clean_env, monkeypatch):
    sub = clean_env / "elsewhere"
    sub.mkdir()
    (sub / ".env").write_text("ODDS_API_KEY=wrong\n")
    monkeypatch.chdir(sub)
    assert Settings().odds_api_key is None


def test_no_staking_settings():
    # analysis-only scope: no bankroll / stake / kelly knobs anywhere in settings
    banned = ("bankroll", "stake", "kelly", "wager", "bet_size", "unit_size")
    assert not [f for f in Settings.model_fields if any(b in f for b in banned)]
