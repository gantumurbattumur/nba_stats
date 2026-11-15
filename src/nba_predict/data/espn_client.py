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
# core API: per-provider odds with open/close values (home-perspective spread)
CORE_URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event_id}/"
            "competitions/{event_id}/odds")


class EspnNotFound(IngestError):
    """ESPN answered 404/410 for this resource: permanent, not retried."""
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


def _american(v) -> float | None:
    """Core-API american strings: '+2.5', '-120', 'EVEN' → number."""
    if isinstance(v, dict):
        v = v.get("american", v.get("alternateDisplayValue"))
    if isinstance(v, str) and v.strip().upper() in ("EVEN", "EV", "PK", "PICK"):
        return 100.0 if v.strip().upper().startswith("EV") else 0.0
    return _num(v, float)


def _core_side(item: dict, when: str) -> dict:
    """Snapshot fields for ``when`` ∈ {open, close}; close falls back to the top-level values."""
    h = (item.get("homeTeamOdds") or {})
    a = (item.get("awayTeamOdds") or {})
    ho, ao, it = h.get(when) or {}, a.get(when) or {}, item.get(when) or {}
    top = when == "close"
    out = dict(
        home_ml=_price(_american(ho.get("moneyLine")) if ho.get("moneyLine") else (h.get("moneyLine") if top else None)),
        away_ml=_price(_american(ao.get("moneyLine")) if ao.get("moneyLine") else (a.get("moneyLine") if top else None)),
        home_spread=(_american(ho.get("pointSpread")) if ho.get("pointSpread")
                     else (_num(item.get("spread"), float) if top else None)),
        home_spread_price=_price(_american(ho.get("spread")) if ho.get("spread") else (h.get("spreadOdds") if top else None)),
        away_spread_price=_price(_american(ao.get("spread")) if ao.get("spread") else (a.get("spreadOdds") if top else None)),
        total=(_american(it.get("total")) if it.get("total") else (_num(item.get("overUnder"), float) if top else None)),
        over_price=_price(_american(it.get("over")) if it.get("over") else (item.get("overOdds") if top else None)),
        under_price=_price(_american(it.get("under")) if it.get("under") else (item.get("underOdds") if top else None)),
    )
    return out


def parse_core_odds(data: dict) -> dict | None:
    """First non-live provider of the core odds response → ``{book, close, open}`` (each a dict of
    snapshot fields or ``None``); ``None`` if there is no such provider."""
    for item in data.get("items") or []:
        name = (item.get("provider") or {}).get("name") or ""
        if "live" in name.lower():
            continue
        close, open_ = _core_side(item, "close"), _core_side(item, "open")
        return {"book": name or "espn",
                "close": close if any(v is not None for v in close.values()) else None,
                "open": open_ if any(v is not None for v in open_.values()) else None}
    return None


class EspnClient:
    def __init__(self, delay_s: float, retries: int = 3, timeout: int = 20):
        self.delay_s = delay_s
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, params: dict, url: str | None = None) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries):
            time.sleep(self.delay_s)
            try:
                r = self.session.get(url or f"{BASE}/{path}", params=params, timeout=self.timeout)
                if r.status_code in (404, 410):
                    raise EspnNotFound(f"ESPN {path} {params}: HTTP {r.status_code}")
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    # other client errors won't fix themselves on retry
                    raise IngestError(f"ESPN {path} {params}: HTTP {r.status_code}")
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

    def core_odds(self, event_id: str) -> dict:
        return self._get("core-odds", {}, url=CORE_URL.format(event_id=event_id))
