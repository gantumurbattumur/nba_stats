"""Optional, conservative adapter for externally acquired Sportsbook Review lines.

The public page has no stable documented feed. This module deliberately accepts normalized
records from a caller and persists them; it does not silently scrape a changed page layout.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time

import pandas as pd

from nba_predict.db import upsert

BOOKS = ("FanDuel", "DraftKings", "BetMGM", "Caesars", "BetRivers", "ESPN BET")
_LAST_REQUEST = 0.0


def rate_limit():
    """Enforce at most one source request per second for a future page adapter."""
    global _LAST_REQUEST
    wait = 1.0 - (time.monotonic() - _LAST_REQUEST)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST = time.monotonic()


def ingest_records(conn: sqlite3.Connection, records, *, pulled_at=None) -> int:
    """Persist normalized rows keyed by game and named book; no network access is performed."""
    pulled_at = pulled_at or dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    allowed = {"game_id", "book", "home_ml", "away_ml", "home_spread", "home_spread_price",
               "away_spread_price", "total", "over_price", "under_price"}
    rows = []
    for record in records:
        row = {k: v for k, v in dict(record).items() if k in allowed}
        if not row.get("game_id") or not row.get("book"):
            raise ValueError("each sportsbook line requires game_id and book")
        row.update(source="sbr", pulled_at=pulled_at, is_closing=0)
        rows.append(row)
    if not rows:
        return 0
    return upsert(conn, "odds_snapshots", rows, ["game_id", "source", "book", "pulled_at"])


def book_table(conn: sqlite3.Connection, game_id: str) -> pd.DataFrame:
    """Latest informational line per book for a game."""
    return pd.read_sql_query("""select o.* from odds_snapshots o join
      (select book,max(pulled_at) pulled_at from odds_snapshots where game_id=? and source='sbr' group by book) x
      using(book,pulled_at) where o.game_id=? and o.source='sbr' order by o.book""", conn, params=(game_id, game_id))
