"""Static NBA team map: nba_api id <-> abbreviation <-> Odds API name <-> ESPN abbreviation.

Seeded from ``nba_api.stats.static.teams.get_teams()`` (written out, no runtime dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Team:
    nba_id: int
    abbr: str
    name: str
    odds_name: str
    espn_abbr: str
    conference: Literal["East", "West"]
    division: str


# (nba_id, abbr, full name, espn abbr, conference, division); Odds API name == NBA full name
_RAW = [
    (1610612737, "ATL", "Atlanta Hawks", "ATL", "East", "Southeast"),
    (1610612738, "BOS", "Boston Celtics", "BOS", "East", "Atlantic"),
    (1610612751, "BKN", "Brooklyn Nets", "BKN", "East", "Atlantic"),
    (1610612766, "CHA", "Charlotte Hornets", "CHA", "East", "Southeast"),
    (1610612741, "CHI", "Chicago Bulls", "CHI", "East", "Central"),
    (1610612739, "CLE", "Cleveland Cavaliers", "CLE", "East", "Central"),
    (1610612742, "DAL", "Dallas Mavericks", "DAL", "West", "Southwest"),
    (1610612743, "DEN", "Denver Nuggets", "DEN", "West", "Northwest"),
    (1610612765, "DET", "Detroit Pistons", "DET", "East", "Central"),
    (1610612744, "GSW", "Golden State Warriors", "GS", "West", "Pacific"),
    (1610612745, "HOU", "Houston Rockets", "HOU", "West", "Southwest"),
    (1610612754, "IND", "Indiana Pacers", "IND", "East", "Central"),
    (1610612746, "LAC", "Los Angeles Clippers", "LAC", "West", "Pacific"),
    (1610612747, "LAL", "Los Angeles Lakers", "LAL", "West", "Pacific"),
    (1610612763, "MEM", "Memphis Grizzlies", "MEM", "West", "Southwest"),
    (1610612748, "MIA", "Miami Heat", "MIA", "East", "Southeast"),
    (1610612749, "MIL", "Milwaukee Bucks", "MIL", "East", "Central"),
    (1610612750, "MIN", "Minnesota Timberwolves", "MIN", "West", "Northwest"),
    (1610612740, "NOP", "New Orleans Pelicans", "NO", "West", "Southwest"),
    (1610612752, "NYK", "New York Knicks", "NY", "East", "Atlantic"),
    (1610612760, "OKC", "Oklahoma City Thunder", "OKC", "West", "Northwest"),
    (1610612753, "ORL", "Orlando Magic", "ORL", "East", "Southeast"),
    (1610612755, "PHI", "Philadelphia 76ers", "PHI", "East", "Atlantic"),
    (1610612756, "PHX", "Phoenix Suns", "PHX", "West", "Pacific"),
    (1610612757, "POR", "Portland Trail Blazers", "POR", "West", "Northwest"),
    (1610612758, "SAC", "Sacramento Kings", "SAC", "West", "Pacific"),
    (1610612759, "SAS", "San Antonio Spurs", "SA", "West", "Southwest"),
    (1610612761, "TOR", "Toronto Raptors", "TOR", "East", "Atlantic"),
    (1610612762, "UTA", "Utah Jazz", "UTAH", "West", "Northwest"),
    (1610612764, "WAS", "Washington Wizards", "WSH", "East", "Southeast"),
]

TEAMS: list[Team] = [
    Team(nba_id=i, abbr=a, name=n, odds_name=n, espn_abbr=e, conference=c, division=d)
    for i, a, n, e, c, d in _RAW
]

_BY_ID = {t.nba_id: t for t in TEAMS}
_BY_ABBR = {t.abbr: t for t in TEAMS}
_BY_ODDS = {t.odds_name.lower(): t for t in TEAMS}
_BY_ESPN = {t.espn_abbr: t for t in TEAMS}


def by_nba_id(i: int) -> Team | None:
    return _BY_ID.get(int(i)) if i is not None else None


def by_abbr(s: str) -> Team | None:
    return _BY_ABBR.get(s.upper()) if s else None


def by_odds_name(s: str) -> Team | None:
    return _BY_ODDS.get(s.strip().lower()) if s else None


def by_espn_abbr(s: str) -> Team | None:
    return _BY_ESPN.get(s.upper()) if s else None
