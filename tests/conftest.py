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
    """``logs`` is served for "Regular Season"; ``other_logs[season_type]`` (default empty)
    for "PlayIn"/"Playoffs". ``log_calls`` records (season, season_type)."""

    def __init__(self, logs: pd.DataFrame):
        self.logs = logs
        self.other_logs: dict[str, pd.DataFrame] = {}
        self.slate_rows: list[dict] = []
        self.log_calls: list[tuple[str, str]] = []

    def season_team_logs(self, season: str, season_type: str = "Regular Season") -> pd.DataFrame:
        self.log_calls.append((season, season_type))
        if season_type == "Regular Season":
            return self.logs.copy()
        return self.other_logs.get(season_type, pd.DataFrame()).copy()

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


# --- synthetic league (Task 5+) ---------------------------------------------------------------

LEAGUE = {"BOS": (1610612738, 6), "NYK": (1610612752, 0), "LAL": (1610612747, -4),
          "DEN": (1610612743, 3)}  # abbr -> (nba_id, strength)
SEASONS = ("2024-25", "2025-26")
SEASON_START = {"2024-25": "2024-10-22", "2025-26": "2025-10-21"}
DAY_OFFSETS = (0, 1, 3, 5, 6, 8, 10, 12, 13, 15, 17, 19)  # 12 dates; days 0/1 = back-to-back
MATCHINGS = ((("BOS", "NYK"), ("LAL", "DEN")), (("BOS", "LAL"), ("NYK", "DEN")),
             (("BOS", "DEN"), ("NYK", "LAL")))


def _box(i: int, t: int, strength: int) -> dict:
    """Deterministic box score for global game index ``i`` and team slot ``t`` (0 home, 1 away)."""
    fga = 84 + (7 * i + 3 * t) % 9
    fgm = 37 + strength // 2 + (5 * i + 11 * t) % 7
    fg3a = 30 + (i + 4 * t) % 9
    fg3m = 10 + (3 * i + t) % 6
    fta = 17 + (2 * i + 5 * t) % 9
    ftm = fta - 2 - (i + t) % 4
    oreb = 8 + (i + 2 * t) % 6
    dreb = 31 + (3 * i + t) % 7
    tov = 11 + (i + 3 * t) % 6
    return dict(min=240, fgm=fgm, fga=fga, fg3m=fg3m, fg3a=fg3a, ftm=ftm, fta=fta, oreb=oreb,
                dreb=dreb, reb=oreb + dreb, ast=22 + (i + t) % 8, stl=6 + (i + t) % 5,
                blk=4 + (2 * i + t) % 4, tov=tov, pf=18 + (i + 2 * t) % 6)


def build_league(conn) -> None:
    import datetime as dt

    from nba_predict.db import upsert

    games, tgs, snaps = [], [], []
    i = 0
    for si, season in enumerate(SEASONS):
        start = dt.date.fromisoformat(SEASON_START[season])
        for di, off in enumerate(DAY_OFFSETS):
            d = (start + dt.timedelta(days=off)).isoformat()
            for (a, b) in MATCHINGS[di % 3]:
                # each pair meets 4x per season; alternate who hosts
                home, away = (a, b) if (di // 3) % 2 == 0 else (b, a)
                gid = f"00{2 + 0}{24 + si}{i:05d}"
                (hid, hs), (aid, as_) = LEAGUE[home], LEAGUE[away]
                hb, ab = _box(i, 0, hs), _box(i, 1, as_)
                hb["pts"] = 2 * hb["fgm"] + hb["fg3m"] + hb["ftm"]
                ab["pts"] = 2 * ab["fgm"] + ab["fg3m"] + ab["ftm"]
                if hb["pts"] == ab["pts"]:  # no ties
                    hb["ftm"] += 1
                    hb["pts"] += 1
                games.append(dict(game_id=gid, season=season, game_date=d, home_team_id=hid,
                                  away_team_id=aid, home_pts=hb["pts"], away_pts=ab["pts"],
                                  status="final", season_type="regular"))
                for tid, oid, box, obox, is_home in ((hid, aid, hb, ab, 1), (aid, hid, ab, hb, 0)):
                    tgs.append(dict(game_id=gid, team_id=tid, opp_team_id=oid, season=season,
                                    game_date=d, is_home=is_home,
                                    wl="W" if box["pts"] > obox["pts"] else "L",
                                    plus_minus=float(box["pts"] - obox["pts"]),
                                    season_type="regular", **box))
                if si == 1:  # ESPN closing lines on season 2 only
                    spread = -(hs - as_ + 2.5)
                    fav = 110 + int(abs(spread)) * 20
                    dog = 100 + int(abs(spread)) * 17
                    hml, aml = (-fav, dog) if spread < 0 else (dog, -fav)
                    snaps.append(dict(game_id=gid, source="espn", book="Draft Kings",
                                      pulled_at=f"{d}T23:59:59Z", is_closing=1, home_ml=hml,
                                      away_ml=aml, home_spread=spread, home_spread_price=-110,
                                      away_spread_price=-110, total=224.5 + ((i % 5) - 2) * 2,
                                      over_price=-110, under_price=-110))
                i += 1
    last = dt.date.fromisoformat(games[-1]["game_date"]) + dt.timedelta(days=1)
    for k, (h, a) in enumerate((("BOS", "NYK"), ("DEN", "LAL"))):
        games.append(dict(game_id=f"0022599{k:03d}", season=SEASONS[1], game_date=last.isoformat(),
                          home_team_id=LEAGUE[h][0], away_team_id=LEAGUE[a][0], status="scheduled",
                          season_type="regular"))
    upsert(conn, "games", games, ["game_id"])
    upsert(conn, "team_games", tgs, ["game_id", "team_id"])
    upsert(conn, "odds_snapshots", snaps, ["game_id", "source", "book", "pulled_at"])


@pytest.fixture
def fixture_db(conn):
    """Deterministic 4-team league (BOS, NYK, LAL, DEN; real nba ids), seasons 2024-25 and
    2025-26, 24 final games each (12 dates, 2 games per date, every team plays every date,
    each pair meets 4x with home/away alternating), ESPN closing lines on 2025-26 only,
    plus 2 scheduled games (BOS-NYK, DEN-LAL) on 2025-11-10 (last date + 1).

    Box scores come from ``_box(i, slot, strength)``. BOS's first 3 games (2024-25, all home):

    | # | date       | game_id    | opp | pts-opp | fgm fga 3pm ftm fta oreb dreb tov | opp fgm fga 3pm ftm fta oreb dreb tov |
    |---|------------|------------|-----|---------|-----------------------------------|---------------------------------------|
    | 1 | 2024-10-22 | 0022400000 | NYK | 105-112 | 40 84 10 15 17  8 31 11           | 41 87 11 19 22 10 32 14               |
    | 2 | 2024-10-23 | 0022400002 | LAL | 113-93  | 43 89 10 17 21 10 37 13           | 35 92 11 12 17 12 31 16               |
    | 3 | 2024-10-25 | 0022400004 | DEN | 125-111 | 46 85 10 23 25 12 36 15           | 41 88 11 18 21  8 37 12               |

    Game 1 by hand: poss = 84 + 0.44*17 - 8 + 11 = 94.48, opp_poss = 87 + 0.44*22 - 10 + 14 =
    100.68, game_poss = 97.58, ortg = 100*105/97.58 = 107.6040, drtg = 100*112/97.58 =
    114.7776, pace = 97.58, efg = (40 + 5)/84, orb_pct = 8/(8 + 32) = 0.2, ftr = 17/84.
    Game 2 is the 2nd day in a row (rest_days 0, b2b); game 3 has rest_days 1.
    """
    build_league(conn)
    return conn


@pytest.fixture
def enriched(fixture_db):
    from nba_predict.analysis.enrich import build_enriched

    return build_enriched(fixture_db)
