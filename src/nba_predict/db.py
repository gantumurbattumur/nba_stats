"""SQLite connection, schema and small helpers."""

from __future__ import annotations

import re
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
    espn_event_id TEXT,
    season_type TEXT NOT NULL DEFAULT 'regular'
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
    season_type TEXT NOT NULL DEFAULT 'regular',
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

-- fingerprint of the source rows each derived (pandas-written) table was built from
CREATE TABLE IF NOT EXISTS derived_meta (
    name TEXT PRIMARY KEY,
    fingerprint TEXT,
    built_at TEXT
);

CREATE TABLE IF NOT EXISTS backtest_results (
    run_at TEXT,
    model TEXT,
    season TEXT,
    metric TEXT,
    value REAL,
    segment TEXT NOT NULL DEFAULT 'all',   -- all | early (either team < 15 season games) | rest
    role TEXT NOT NULL DEFAULT 'test'      -- test (walk-forward) | holdout (reported only)
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


# NBA game-id prefix → season_type ('004' playoffs, '005' play-in, else regular)
SEASON_TYPE_SQL = ("CASE substr(game_id, 1, 3) WHEN '004' THEN 'playoffs' "
                   "WHEN '005' THEN 'playin' ELSE 'regular' END")


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations for DBs created by older versions (idempotent)."""
    for table in ("games", "team_games"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "season_type" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN season_type TEXT NOT NULL DEFAULT 'regular'")
            conn.execute(f"UPDATE {table} SET season_type = {SEASON_TYPE_SQL}")
    bt = {r[1] for r in conn.execute("PRAGMA table_info(backtest_results)")}
    if "segment" not in bt:
        conn.execute("ALTER TABLE backtest_results ADD COLUMN segment TEXT NOT NULL DEFAULT 'all'")
    if "role" not in bt:
        conn.execute("ALTER TABLE backtest_results ADD COLUMN role TEXT NOT NULL DEFAULT 'test'")


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_ident(name: str) -> str:
    """Table/column names are interpolated into SQL, so only plain identifiers are allowed."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def upsert(conn: sqlite3.Connection, table: str, rows: list[dict], key: list[str]) -> int:
    """INSERT ... ON CONFLICT(key) DO UPDATE for each row. Returns rows written.

    Columns are the union of keys across ``rows``; only columns present in a row
    group are updated, so partial rows never null out existing values.
    """
    if not rows:
        return 0
    _check_ident(table)
    for k in key:
        _check_ident(k)
    n = 0
    # group rows by their column set so partial rows don't overwrite other columns
    groups: dict[tuple[str, ...], list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.keys()), []).append(r)
    for cols in groups:  # validate everything before writing anything
        for c in cols:
            _check_ident(c)
    try:
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
    except Exception:
        conn.rollback()  # all-or-nothing: don't leave earlier groups pending in the transaction
        raise
    conn.commit()
    return n


def read_df(conn: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


class QueryError(RuntimeError):
    """User SQL rejected (not a single read-only SELECT) or failed."""


_RO_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
               sqlite3.SQLITE_RECURSIVE}
# introspection used by the SQL console sidebar (select ... from pragma_table_info('t'))
_RO_PRAGMAS = {"table_info", "table_xinfo", "table_list", "index_list", "index_info"}
_DENIED_FUNCS = {"load_extension"}


def _ro_authorizer(action, arg1, arg2, _db, _trigger):
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCS:
        return sqlite3.SQLITE_DENY
    if action in _RO_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in _RO_PRAGMAS and arg2 is not None:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


MAX_VALUE_BYTES = 10_000_000  # per string/blob value (SQLITE_LIMIT_LENGTH)


def _row_bytes(row) -> int:
    return sum(len(v) if isinstance(v, (str, bytes)) else 8 for v in row)


def readonly_query(db_path: str | Path, sql: str, params=(), max_rows: int | None = 50_000,
                   timeout_s: float = 30.0, max_bytes: int | None = 256_000_000) -> pd.DataFrame:
    """Run one user-supplied SELECT through a read-only connection (``mode=ro`` + authorizer
    allowing only SELECT/READ/FUNCTION/RECURSIVE). Raises ``QueryError`` on anything else.
    Resource guards: ``timeout_s`` wall clock, ``MAX_VALUE_BYTES`` per value (``randomblob``,
    ``zeroblob``, ``printf`` padding) and ``max_bytes`` for the whole result.
    ``df.attrs["truncated"]`` tells whether more than ``max_rows`` rows existed."""
    import time

    stmt = sql.strip().rstrip(";").strip()
    if not stmt:
        raise QueryError("empty query")
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        raise QueryError(f"cannot open database read-only: {e}") from e
    try:
        conn.set_authorizer(_ro_authorizer)
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = conn.execute(stmt, params)
            cols = [d[0] for d in cur.description or []]
            want = None if max_rows is None else max_rows + 1
            rows: list = []
            size = 0
            while want is None or len(rows) < want:
                chunk = cur.fetchmany(1000 if want is None else min(1000, want - len(rows)))
                if not chunk:
                    break
                rows.extend(chunk)
                if max_bytes is not None:
                    size += sum(_row_bytes(r) for r in chunk)
                    if size > max_bytes:
                        raise QueryError(f"result too large (> {max_bytes:,} bytes); "
                                         "select fewer rows or columns")
        except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError) as e:
            # sqlite3.ProgrammingError "You can only execute one statement at a time." lands here
            raise QueryError(str(e)) from e
    finally:
        conn.close()
    truncated = max_rows is not None and len(rows) > max_rows
    df = pd.DataFrame(rows[:max_rows] if truncated else rows, columns=cols)
    df.attrs["truncated"] = truncated
    return df
