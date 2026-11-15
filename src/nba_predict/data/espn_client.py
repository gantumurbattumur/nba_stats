"""ESPN unofficial API: scoreboard (event ids) and summary (``pickcenter`` past-game lines).

ESPN ``spread`` is from the home team's perspective (negative = home favored).
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from zoneinfo import ZoneInfo

import requests

from nba_predict.data.nba_client import IngestError

log = logging.getLogger(__name__)
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ET = ZoneInfo("America/New_York")


def _num(v, cast):
    if v is None or v == "":
        return None
    try:
        return cast(float(v))
    except (TypeError, ValueError):
        return None


def _price(v) -> int | None:
    """American price or ``None``; values strictly between -100 and +100 (e.g. 0) are invalid."""
    p = _num(v, int)
    return None if p is None or -100 < p < 100 else p


def _et_date(iso: str) -> str:
    """ESPN timestamps look like '2026-11-11T00:30Z' (UTC) → Eastern calendar date."""
    t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return t.astimezone(ET).date().isoformat()


def parse_scoreboard(data: dict) -> list[dict]:
    out = []
    for e in data.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c.get("team", {}).get("abbreviation") for c in comp.get("competitors", [])}
        status = (comp.get("status") or e.get("status") or {}).get("type", {})
        out.append(dict(
            espn_event_id=str(e["id"]),
            game_date=_et_date(e.get("date") or comp["date"]),
            home_espn_abbr=sides.get("home"),
            away_espn_abbr=sides.get("away"),
            completed=bool(status.get("completed", False)),
        ))
    return out


def parse_pickcenter(summary: dict) -> dict | None:
    """First ``pickcenter`` entry → snapshot fields; ``None`` if absent/empty/no numbers."""
    pcs = summary.get("pickcenter") or []
    if not pcs:
        return None
    pc = pcs[0] or {}
    home = pc.get("homeTeamOdds") or {}
    away = pc.get("awayTeamOdds") or {}
    out = dict(
        home_ml=_price(home.get("moneyLine")),
        away_ml=_price(away.get("moneyLine")),
        home_spread=_num(pc.get("spread"), float),
        home_spread_price=_price(home.get("spreadOdds")),
        away_spread_price=_price(away.get("spreadOdds")),
        total=_num(pc.get("overUnder"), float),
        over_price=_price(pc.get("overOdds")),
        under_price=_price(pc.get("underOdds")),
    )
    if all(v is None for v in out.values()):
        return None
    return {"book": (pc.get("provider") or {}).get("name") or "espn", **out}


class EspnClient:
    def __init__(self, delay_s: float, retries: int = 3, timeout: int = 20):
        self.delay_s = delay_s
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, params: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries):
            time.sleep(self.delay_s)
            try:
                r = self.session.get(f"{BASE}/{path}", params=params, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                last = e
                log.warning("ESPN %s %s failed (attempt %d/%d): %s", path, params, attempt + 1, self.retries, e)
                if attempt < self.retries - 1:
                    time.sleep(2**attempt)
        raise IngestError(f"ESPN {path} {params} failed after {self.retries} attempts: {last}")

    def events(self, date: dt.date) -> list[dict]:
        return parse_scoreboard(self._get("scoreboard", {"dates": date.strftime("%Y%m%d")}))

    def summary(self, event_id: str) -> dict:
        return self._get("summary", {"event": str(event_id)})
