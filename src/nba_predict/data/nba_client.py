"""nba_api wrapper: rate limit (sleep before every call) + retry with exponential backoff."""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
import warnings
from collections.abc import Callable
from typing import TypeVar
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamelog, scoreboardv2

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
T = TypeVar("T")


class IngestError(RuntimeError):
    """Raised when an upstream call keeps failing after all retries."""


def today_et() -> dt.date:
    return dt.datetime.now(ET).date()


def _status(status_id: int | None, text: str | None) -> str:
    t = (text or "").lower()
    if "ppd" in t or "postpon" in t:
        return "postponed"
    return "final" if status_id == 3 else "scheduled"


_ET_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*([ap])m\s*ET\s*$", re.IGNORECASE)


def start_time_from_status(game_date: str, status_text: str | None) -> str | None:
    """ScoreboardV2 shows a future game's tip as GAME_STATUS_TEXT '7:30 pm ET' → ISO UTC."""
    m = _ET_TIME.match(status_text or "")
    if not m:
        return None
    h, mi = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0), int(m.group(2))
    d = dt.date.fromisoformat(game_date[:10])
    t = dt.datetime(d.year, d.month, d.day, h, mi, tzinfo=ET).astimezone(dt.UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class NbaClient:
    def __init__(self, delay_s: float, retries: int = 3, timeout: int = 30):
        self.delay_s = delay_s
        self.retries = retries
        self.timeout = timeout

    def _call(self, what: str, fn: Callable[[], T]) -> T:
        last: Exception | None = None
        for attempt in range(self.retries):
            time.sleep(self.delay_s)
            try:
                return fn()
            except (requests.RequestException, ValueError) as e:  # ValueError: bad/blocked JSON
                last = e
                log.warning("%s failed (attempt %d/%d): %s", what, attempt + 1, self.retries, e)
                if attempt < self.retries - 1:
                    time.sleep(2**attempt)  # 1s, 2s, 4s, ...
        raise IngestError(f"{what} failed after {self.retries} attempts: {last}")

    def season_team_logs(self, season: str, season_type: str = "Regular Season") -> pd.DataFrame:
        """Team game logs; ``season_type`` is nba_api's ``season_type_all_star``
        ("Regular Season", "PlayIn", "Playoffs")."""
        return self._call(
            f"LeagueGameLog {season} {season_type}",
            lambda: leaguegamelog.LeagueGameLog(
                season=season, player_or_team_abbreviation="T",
                season_type_all_star=season_type, timeout=self.timeout,
            ).get_data_frames()[0],
        )

    def slate(self, date: dt.date) -> list[dict]:
        """Games on ``date`` (ET): ``{game_id, game_date, home_team_id, away_team_id, status}``."""
        if date == today_et():
            rows = self._call("live ScoreBoard", self._live_slate)
            # the live board rolls over late morning ET; fall back if it still shows yesterday
            if rows is not None:
                return rows
        return self._call(f"ScoreboardV2 {date}", lambda: self._v2_slate(date))

    def _live_slate(self) -> list[dict] | None:
        board = scoreboard.ScoreBoard(timeout=self.timeout).get_dict()["scoreboard"]
        if board.get("gameDate") != today_et().isoformat():
            return None
        return [
            dict(
                game_id=g["gameId"],
                game_date=board["gameDate"],
                home_team_id=int(g["homeTeam"]["teamId"]),
                away_team_id=int(g["awayTeam"]["teamId"]),
                status=_status(g.get("gameStatus"), g.get("gameStatusText")),
                start_time_utc=(g.get("gameTimeUTC")
                                or start_time_from_status(board["gameDate"], g.get("gameStatusText"))),
            )
            for g in board["games"]
        ]

    def _v2_slate(self, date: dt.date) -> list[dict]:
        with warnings.catch_warnings():
            # V2's deprecation warning concerns LineScore; GameHeader is fine
            warnings.simplefilter("ignore", DeprecationWarning)
            sb = scoreboardv2.ScoreboardV2(game_date=date.isoformat(), timeout=self.timeout)
        gh = sb.game_header.get_data_frame()
        return [
            dict(
                game_id=r.GAME_ID,
                game_date=str(r.GAME_DATE_EST)[:10],
                home_team_id=int(r.HOME_TEAM_ID),
                away_team_id=int(r.VISITOR_TEAM_ID),
                status=_status(int(r.GAME_STATUS_ID), r.GAME_STATUS_TEXT),
                start_time_utc=start_time_from_status(str(r.GAME_DATE_EST), r.GAME_STATUS_TEXT),
            )
            for r in gh.drop_duplicates("GAME_ID").itertuples()
        ]
