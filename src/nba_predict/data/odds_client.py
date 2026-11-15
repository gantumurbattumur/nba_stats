"""The Odds API v4 — live NBA odds only (the historical endpoint is never called).

One call = ``GET /v4/sports/basketball_nba/odds?regions=us&markets=h2h,spreads,totals&
oddsFormat=american`` (3 credits). Usage comes from the ``x-requests-*`` response headers.
"""

from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)
URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
PARAMS = {"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}


class OddsApiError(RuntimeError):
    """Clean, user-facing Odds API failure (missing/invalid key, quota, HTTP error)."""


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


class OddsApiClient:
    def __init__(self, api_key: str | None, timeout: int = 20):
        self.api_key = api_key
        self.timeout = timeout

    def nba_odds(self) -> tuple[list[dict], dict]:
        """→ (events JSON, ``{"used", "remaining", "last"}``)."""
        if not self.api_key:
            raise OddsApiError("ODDS_API_KEY not set — add it to .env")
        try:
            r = requests.get(URL, params={"apiKey": self.api_key, **PARAMS}, timeout=self.timeout)
        except requests.RequestException as e:
            raise OddsApiError(f"Odds API request failed: {e}") from e
        usage = {"used": _int(r.headers.get("x-requests-used")),
                 "remaining": _int(r.headers.get("x-requests-remaining")),
                 "last": _int(r.headers.get("x-requests-last"))}
        body = r.text.lower()
        if r.status_code == 429 or (r.status_code == 401 and ("out_of_usage" in body or "quota" in body)):
            raise OddsApiError("quota exhausted")
        if r.status_code == 401:
            raise OddsApiError("invalid key")
        if r.status_code != 200:
            raise OddsApiError(f"Odds API HTTP {r.status_code}: {r.text[:200]}")
        if usage["remaining"] == 0:
            log.warning("Odds API quota now exhausted (0 credits remaining); this pull was stored")
        try:
            return r.json(), usage
        except ValueError as e:
            raise OddsApiError(f"Odds API returned invalid JSON: {e}") from e


def parse_events(events: list[dict]) -> list[dict]:
    """One row per (event, bookmaker) in ``odds_snapshots`` shape plus event_id, home_name,
    away_name, commence_time. ``home_spread`` = the home outcome's point."""
    rows = []
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        for bk in ev.get("bookmakers", []):
            row = dict(event_id=ev.get("id"), home_name=home, away_name=away,
                       commence_time=ev.get("commence_time"), book=bk.get("key") or bk.get("title"),
                       home_ml=None, away_ml=None, home_spread=None, home_spread_price=None,
                       away_spread_price=None, total=None, over_price=None, under_price=None)
            for m in bk.get("markets", []):
                outs = {o.get("name"): o for o in m.get("outcomes", [])}
                h, a = outs.get(home, {}), outs.get(away, {})
                if m.get("key") == "h2h":
                    row.update(home_ml=_int(h.get("price")), away_ml=_int(a.get("price")))
                elif m.get("key") == "spreads" and "point" in h:
                    row.update(home_spread=float(h["point"]), home_spread_price=_int(h.get("price")),
                               away_spread_price=_int(a.get("price")))
                elif m.get("key") == "totals":
                    o, u = outs.get("Over", {}), outs.get("Under", {})
                    if "point" in o:
                        row.update(total=float(o["point"]), over_price=_int(o.get("price")),
                                   under_price=_int(u.get("price")))
            rows.append(row)
    return rows
