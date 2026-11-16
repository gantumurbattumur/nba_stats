"""Shared data access for the dashboard. Reads go through cached, read-only loads keyed on the
DB file's mtime (so any ingest/rebuild invalidates them); writes use ``rw_conn()``."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DISCLAIMER = "Analysis only — no bets placed. Model-vs-market comparisons for data-engineering study."
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def db_path() -> str:
    """``NBA_DB_PATH`` (tests / other DBs) or the project's settings path."""
    p = os.environ.get("NBA_DB_PATH")
    if p:
        return str(Path(p).expanduser().resolve())
    from nba_predict.config import get_settings

    return get_settings().db_path


def _mtime(path: str) -> float:
    return sum(os.path.getmtime(p) for p in (path, path + "-wal") if os.path.exists(p))


def rw_conn() -> sqlite3.Connection:
    from nba_predict.db import connect, init_schema

    c = connect(db_path())
    init_schema(c)
    return c


@st.cache_data(show_spinner=False, max_entries=64)
def _load(path: str, name: str, mtime: float) -> pd.DataFrame:
    c = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return pd.read_sql_query(f'select * from "{name}"', c)
    finally:
        c.close()


def table_names() -> list[str]:
    c = sqlite3.connect(Path(db_path()).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return [r[0] for r in c.execute("select name from sqlite_master where type='table' "
                                        "and name not like 'sqlite_%' order by name")]
    finally:
        c.close()


def load_table(name: str) -> pd.DataFrame:
    """Whole table (cached until the DB file changes); empty frame if it doesn't exist."""
    if not _IDENT.match(name) or name not in table_names():
        return pd.DataFrame()
    return _load(db_path(), name, _mtime(db_path()))


@st.cache_data(show_spinner=False, max_entries=16)
def _query(path: str, sql: str, params: tuple, mtime: float) -> pd.DataFrame:
    from nba_predict.db import readonly_query

    return readonly_query(path, sql, params, max_rows=None)


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Read-only SQL (our own queries; user SQL goes through readonly_query directly)."""
    return _query(db_path(), sql, tuple(params), _mtime(db_path()))


def enriched() -> pd.DataFrame:
    """team_games_enriched, rebuilt first if its sources changed."""
    from nba_predict.analysis.enrich import load_enriched

    c = rw_conn()
    try:
        load_enriched(c)  # no-op when fresh
    finally:
        c.close()
    return load_table("team_games_enriched")


def features() -> pd.DataFrame:
    """game_features, rebuilt first if stale."""
    from nba_predict.config import get_settings
    from nba_predict.features.build import load_features

    c = rw_conn()
    try:
        load_features(c, get_settings().windows)
    finally:
        c.close()
    return load_table("game_features")


def team_names() -> dict[int, str]:
    from nba_predict.teams import TEAMS

    return {t.nba_id: t.abbr for t in TEAMS}


def header(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)
    st.caption(DISCLAIMER)
