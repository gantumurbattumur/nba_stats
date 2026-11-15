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
