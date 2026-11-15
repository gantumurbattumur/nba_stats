"""User-defined custom features: named SQL expressions over ``game_features``.

Expressions only ever run through ``db.readonly_query`` (read-only connection + authorizer).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd

from nba_predict.db import QueryError, readonly_query

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
log = logging.getLogger(__name__)


def check_expression(expr: str) -> None:
    """Reject expressions that could leave the ``(...)`` they are wrapped in: a ``)`` that
    closes past depth 0, unbalanced parens, ``;`` or a comment (``--``, ``/*``) outside string
    literals / quoted identifiers. Raises ``QueryError``."""
    if not isinstance(expr, str) or not expr.strip():
        raise QueryError("empty expression")
    depth, i, n = 0, 0, len(expr)
    closers = {"'": "'", '"': '"', "`": "`", "[": "]"}
    while i < n:
        ch = expr[i]
        if ch in closers:
            j = expr.find(closers[ch], i + 1)
            while j != -1 and ch != "[" and j + 1 < n and expr[j + 1] == ch:  # '' / "" escapes
                j = expr.find(closers[ch], j + 2)
            if j == -1:
                raise QueryError("unterminated quote in expression")
            i = j + 1
            continue
        if ch == ";" or expr.startswith(("--", "/*"), i):
            raise QueryError("';' and SQL comments are not allowed in a feature expression")
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise QueryError("unbalanced ')' in expression")
        i += 1
    if depth != 0:
        raise QueryError("unbalanced '(' in expression")


def _db_path(conn: sqlite3.Connection) -> str:
    return conn.execute("pragma database_list").fetchone()[2]


def _feature_columns(conn: sqlite3.Connection) -> set[str]:
    return {r[1] for r in conn.execute("pragma table_info(game_features)")}


def add_feature(conn: sqlite3.Connection, name: str, expression: str, use_in_model: bool = False) -> None:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"feature name must match {NAME_RE.pattern}: {name!r}")
    if name in _feature_columns(conn):
        raise ValueError(f"{name!r} collides with a game_features column")
    check_expression(expression)
    # validation run; QueryError propagates (bad column, multiple statements, writes, ...)
    readonly_query(_db_path(conn), f"SELECT game_id, ({expression}) AS v FROM game_features LIMIT 5")
    conn.execute(
        "insert into custom_features (name, expression, use_in_model, created_at) values (?, ?, ?, ?) "
        "on conflict (name) do update set expression = excluded.expression, "
        "use_in_model = excluded.use_in_model",
        (name, expression, int(bool(use_in_model)), dt.datetime.now(dt.UTC).isoformat(timespec="seconds")))
    conn.commit()


def list_features(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "select name, expression, use_in_model, created_at from custom_features order by name", conn)


def delete_feature(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("delete from custom_features where name = ?", (name,))
    conn.commit()


def set_use_in_model(conn: sqlite3.Connection, name: str, use: bool) -> None:
    conn.execute("update custom_features set use_in_model = ? where name = ?", (int(bool(use)), name))
    conn.commit()


def _defs(db_path: str | Path, names: list[str] | None, only_model: bool = False) -> list[tuple[str, str]]:
    df = readonly_query(db_path, "select name, expression, use_in_model from custom_features order by name",
                        max_rows=None)
    if names is not None:
        df = df[df.name.isin(names)]
    if only_model:
        df = df[df.use_in_model == 1]
    return list(zip(df.name, df.expression))


def evaluate(db_path: str | Path, names: list[str] | None = None, only_model: bool = False) -> pd.DataFrame:
    """``game_id`` + one column per custom feature, evaluated over ``game_features``."""
    defs = _defs(db_path, names, only_model)
    if not defs:
        return readonly_query(db_path, "select game_id from game_features", max_rows=None)
    for n, expr in defs:  # validated on insert; re-check before interpolating
        if not NAME_RE.match(n):
            raise QueryError(f"invalid stored feature name {n!r}")
        check_expression(expr)
    cols = ", ".join(f'({expr}) AS "{n}"' for n, expr in defs)
    try:
        return readonly_query(db_path, f"SELECT game_id, {cols} FROM game_features", max_rows=None)
    except QueryError:
        pass  # one stale/broken feature (e.g. a dropped window column): evaluate one by one
    out = readonly_query(db_path, "select game_id from game_features", max_rows=None)
    for n, expr in defs:
        try:
            v = readonly_query(db_path, f'SELECT game_id, ({expr}) AS "{n}" FROM game_features',
                               max_rows=None)
        except QueryError as e:
            log.warning("custom feature %r skipped: %s", n, e)
            continue
        out = out.merge(v, on="game_id", how="left")
    return out


def with_custom(features_df: pd.DataFrame, db_path: str | Path, only_model: bool = False) -> pd.DataFrame:
    """Left-join custom feature columns (all, or only ``use_in_model``) onto ``features_df``."""
    ev = evaluate(db_path, only_model=only_model)
    extra = [c for c in ev.columns if c != "game_id"]
    if not extra:
        return features_df
    return features_df.merge(ev, on="game_id", how="left")
