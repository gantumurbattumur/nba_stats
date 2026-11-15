"""SQLite connection, schema and small helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from nba_predict.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season TEXT,
    game_date TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_pts INTEGER,
    away_pts INTEGER,
    status TEXT,
    espn_event_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_games_date ON games(game_date);

CREATE TABLE IF NOT EXISTS team_games (
    game_id TEXT,
    team_id INTEGER,
    opp_team_id INTEGER,
    season TEXT,
    game_date TEXT,
    is_home INTEGER,
    wl TEXT,
    min REAL,
    pts INTEGER, fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER,
    ftm INTEGER, fta INTEGER, oreb INTEGER, dreb INTEGER, reb INTEGER,
    ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER, pf INTEGER,
    plus_minus REAL,
    PRIMARY KEY (game_id, team_id)
);
CREATE INDEX IF NOT EXISTS ix_team_games_team_date ON team_games(team_id, game_date);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    source TEXT,
    book TEXT,
    pulled_at TEXT,
    is_closing INTEGER,
    home_ml INTEGER,
    away_ml INTEGER,
    home_spread REAL,
    home_spread_price INTEGER,
    away_spread_price INTEGER,
    total REAL,
    over_price INTEGER,
    under_price INTEGER,
    UNIQUE (game_id, source, book, pulled_at)
);
CREATE INDEX IF NOT EXISTS ix_odds_game ON odds_snapshots(game_id);

CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT,
    source TEXT,
    credits_used INTEGER,
    credits_remaining INTEGER
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    model TEXT,
    created_at TEXT,
    home_win_prob REAL,
    pred_margin REAL,
    pred_total REAL,
    params_json TEXT,
    UNIQUE (game_id, model)
);

CREATE TABLE IF NOT EXISTS value_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    created_at TEXT,
    model TEXT,
    market TEXT,
    side TEXT,
    line REAL,
    price INTEGER,
    book TEXT,
    model_prob REAL,
    market_prob REAL,
    edge REAL,
    result TEXT,
    units REAL,
    closing_line REAL,
    closing_price INTEGER,
    clv REAL,
    UNIQUE (game_id, market, side)
);

CREATE TABLE IF NOT EXISTS filter_presets (
    name TEXT PRIMARY KEY,
    spec_json TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS custom_features (
    name TEXT PRIMARY KEY,
    expression TEXT,
    use_in_model INTEGER,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS backtest_results (
    run_at TEXT,
    model TEXT,
    season TEXT,
    metric TEXT,
    value REAL
);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the SQLite DB (default ``settings.db_path``), creating its parent dir."""
    p = Path(path if path is not None else get_settings().db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert(conn: sqlite3.Connection, table: str, rows: list[dict], key: list[str]) -> int:
    """INSERT ... ON CONFLICT(key) DO UPDATE for each row. Returns rows written.

    Columns are the union of keys across ``rows``; only columns present in a row
    group are updated, so partial rows never null out existing values.
    """
    if not rows:
        return 0
    n = 0
    # group rows by their column set so partial rows don't overwrite other columns
    groups: dict[tuple[str, ...], list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.keys()), []).append(r)
    for cols, grp in groups.items():
        col_sql = ", ".join(f'"{c}"' for c in cols)
        ph = ", ".join("?" for _ in cols)
        updates = [c for c in cols if c not in key]
        if updates:
            set_sql = ", ".join(f'"{c}" = excluded."{c}"' for c in updates)
            conflict = f"DO UPDATE SET {set_sql}"
        else:
            conflict = "DO NOTHING"
        key_sql = ", ".join(f'"{k}"' for k in key)
        sql = f'INSERT INTO "{table}" ({col_sql}) VALUES ({ph}) ON CONFLICT ({key_sql}) {conflict}'
        conn.executemany(sql, [tuple(r[c] for c in cols) for r in grp])
        n += len(grp)
    conn.commit()
    return n


def read_df(conn: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)
