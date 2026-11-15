import json
from pathlib import Path

import pandas as pd
import pytest

from nba_predict.db import connect, init_schema

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    init_schema(c)
    yield c
    c.close()


@pytest.fixture
def sample_logs() -> pd.DataFrame:
    """Hand-built LeagueGameLog (team mode) rows: 3 games of 2025-26, both teams each.

    0022500001 2025-10-21 NYK @ BOS 106-116; 0022500002 2025-10-22 LAL @ DEN 122-121 (OT);
    0022500003 2025-11-01 LAL @ NYK 102-116.
    """
    return pd.DataFrame(load_fixture("leaguegamelog_sample.json"))


class FakeNbaClient:
    def __init__(self, logs: pd.DataFrame):
        self.logs = logs
        self.slate_rows: list[dict] = []

    def season_team_logs(self, season: str) -> pd.DataFrame:
        return self.logs.copy()

    def slate(self, date) -> list[dict]:
        return list(self.slate_rows)


@pytest.fixture
def fake_client(sample_logs) -> FakeNbaClient:
    return FakeNbaClient(sample_logs)


class FakeEspn:
    """Serves the saved real ESPN responses: scoreboard 2026-11-10; summary 401909960 has a
    DraftKings pickcenter line, every other event returns the no-odds summary."""

    def __init__(self):
        from nba_predict.data.espn_client import parse_scoreboard

        self._events = parse_scoreboard(load_fixture("espn_scoreboard_20261110.json"))
        self._with = load_fixture("espn_summary_with_odds.json")
        self._without = load_fixture("espn_summary_no_odds.json")
        self.summary_calls = 0
        self.events_calls = 0

    def events(self, date):
        self.events_calls += 1
        return [e for e in self._events if e["game_date"] == date.isoformat()]

    def summary(self, event_id):
        self.summary_calls += 1
        return self._with if event_id == "401909960" else self._without


@pytest.fixture
def fake_espn() -> FakeEspn:
    return FakeEspn()


@pytest.fixture
def summary_with_odds() -> dict:
    return load_fixture("espn_summary_with_odds.json")


@pytest.fixture
def summary_no_odds() -> dict:
    return load_fixture("espn_summary_no_odds.json")


@pytest.fixture
def seeded_games(conn):
    """Two final games on 2026-11-10 matching the ESPN scoreboard fixture:
    MIL @ NYK (event 401909960, has odds) and BOS @ PHI (401909961, served without odds)."""
    from nba_predict.db import upsert

    rows = [
        dict(game_id="0022600211", season="2026-27", game_date="2026-11-10",
             home_team_id=1610612752, away_team_id=1610612749, home_pts=117, away_pts=85,
             status="final"),
        dict(game_id="0022600212", season="2026-27", game_date="2026-11-10",
             home_team_id=1610612755, away_team_id=1610612738, home_pts=110, away_pts=108,
             status="final"),
    ]
    upsert(conn, "games", rows, ["game_id"])
    return rows
