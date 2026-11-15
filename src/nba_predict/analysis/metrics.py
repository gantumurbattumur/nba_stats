"""Per-game derived columns and a registry of aggregate metrics.

Ratings are sum-based over a set of games (``ortg = 100 * Σpts / Σgame_poss``), not the mean
of per-game ratios.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


def _div(a, b):
    """Elementwise/scalar a / b with 0 or NaN denominators → NaN."""
    if isinstance(b, pd.Series):
        return a / b.where(b != 0)
    return a / b if b not in (0, None) and not pd.isna(b) else np.nan


def add_game_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row poss, game_poss, ortg, drtg, net, pace, efg, tov_pct, orb_pct, ftr, margin, win.

    Needs team box columns plus ``opp_pts, opp_fga, opp_fta, opp_oreb, opp_tov, opp_dreb``
    (``opp_min`` optional; pace uses ``game_min = max(min, opp_min)``).
    """
    df = df.copy()
    df["poss"] = df.fga + 0.44 * df.fta - df.oreb + df.tov
    df["opp_poss"] = df.opp_fga + 0.44 * df.opp_fta - df.opp_oreb + df.opp_tov
    df["game_poss"] = (df.poss + df.opp_poss) / 2
    df["ortg"] = 100 * _div(df.pts, df.game_poss)
    df["drtg"] = 100 * _div(df.opp_pts, df.game_poss)
    df["net"] = df.ortg - df.drtg
    # both teams play the same minutes; max() guards against a bad team MIN upstream
    df["game_min"] = df[["min", "opp_min"]].max(axis=1) if "opp_min" in df else df["min"]
    df["pace"] = 48 * _div(df.game_poss, df["game_min"] / 5)
    df["efg"] = _div(df.fgm + 0.5 * df.fg3m, df.fga)
    df["tov_pct"] = _div(df.tov, df.poss)
    df["orb_pct"] = _div(df.oreb, df.oreb + df.opp_dreb)
    df["ftr"] = _div(df.fta, df.fga)
    df["margin"] = df.pts - df.opp_pts
    df["total_pts"] = df.pts + df.opp_pts
    df["win"] = df.wl == "W"
    return df


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    kind: Literal["rating", "pct", "mean", "record"]
    fn: Callable[[pd.DataFrame], float | str]


def _s(df, c):
    return df[c].sum() if c in df and len(df) else 0.0


def _ortg(d):
    return 100 * _div(_s(d, "pts"), _s(d, "game_poss"))


def _drtg(d):
    return 100 * _div(_s(d, "opp_pts"), _s(d, "game_poss"))


def _mean(c):
    return lambda d: d[c].mean() if c in d and len(d) else np.nan


def _count(d, col, val):
    return int((d[col] == val).sum()) if col in d else 0


def _record(d):
    w = int(d["win"].sum()) if "win" in d and len(d) else 0
    return f"{w}-{len(d) - w}"


def _ats_pct(d):
    c, n = _count(d, "ats", "cover"), _count(d, "ats", "no_cover")
    return _div(c, c + n)


def _over_pct(d):
    o, u = _count(d, "ou", "over"), _count(d, "ou", "under")
    return _div(o, o + u)


_DEFS = [
    ("ortg", "Offensive rating", "rating", _ortg),
    ("drtg", "Defensive rating", "rating", _drtg),
    ("net", "Net rating", "rating", lambda d: _ortg(d) - _drtg(d)),
    ("pace", "Pace", "rating", lambda d: 48 * _div(_s(d, "game_poss"), _s(d, "game_min") / 5)),
    ("efg", "Effective FG%", "pct", lambda d: _div(_s(d, "fgm") + 0.5 * _s(d, "fg3m"), _s(d, "fga"))),
    ("tov_pct", "Turnover rate", "pct", lambda d: _div(_s(d, "tov"), _s(d, "poss"))),
    ("orb_pct", "Off. rebound rate", "pct",
     lambda d: _div(_s(d, "oreb"), _s(d, "oreb") + _s(d, "opp_dreb"))),
    ("ftr", "Free-throw rate", "pct", lambda d: _div(_s(d, "fta"), _s(d, "fga"))),
    ("margin", "Avg margin", "mean", _mean("margin")),
    ("pts", "Points", "mean", _mean("pts")),
    ("opp_pts", "Opp points", "mean", _mean("opp_pts")),
    ("total_pts", "Total points", "mean", _mean("total_pts")),
    ("win_pct", "Win %", "pct", lambda d: d["win"].mean() if len(d) else np.nan),
    ("record", "Record (W-L)", "record", _record),
    ("ats_record", "ATS (W-L-P)", "record",
     lambda d: f"{_count(d, 'ats', 'cover')}-{_count(d, 'ats', 'no_cover')}-{_count(d, 'ats', 'push')}"),
    ("ats_pct", "ATS cover %", "pct", _ats_pct),
    ("ou_record", "O/U (O-U-P)", "record",
     lambda d: f"{_count(d, 'ou', 'over')}-{_count(d, 'ou', 'under')}-{_count(d, 'ou', 'push')}"),
    ("over_pct", "Over %", "pct", _over_pct),
    ("cover_margin", "Avg cover margin", "mean", _mean("cover_margin")),
]
METRICS: dict[str, Metric] = {k: Metric(k, label, kind, fn) for k, label, kind, fn in _DEFS}


def _agg_one(d: pd.DataFrame, keys: list[str], raw_aggs: dict[str, str]) -> dict:
    row: dict = {"gp": len(d)}
    for k in keys:
        v = METRICS[k].fn(d)
        row[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
    for col, how in raw_aggs.items():
        if how not in ("mean", "median", "sum"):
            raise ValueError(f"raw agg must be mean|median|sum, got {how!r}")
        row[f"{col}_{how}"] = getattr(d[col], how)() if len(d) else np.nan
    return row


def aggregate(
    df: pd.DataFrame,
    metric_keys: list[str],
    group_by: list[str] | None = None,
    raw_aggs: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate ``df`` into metrics (always with ``gp``); one row, or one per group."""
    for k in metric_keys:
        if k not in METRICS:
            raise KeyError(f"unknown metric {k!r}")
    raw_aggs = raw_aggs or {}
    cols = ["gp", *metric_keys, *(f"{c}_{h}" for c, h in raw_aggs.items())]
    if not group_by:
        return pd.DataFrame([_agg_one(df, metric_keys, raw_aggs)], columns=cols)
    if df.empty:
        return pd.DataFrame(columns=[*group_by, *cols])
    rows = []
    for key, grp in df.groupby(group_by, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_by, key)), **_agg_one(grp, metric_keys, raw_aggs)})
    return pd.DataFrame(rows, columns=[*group_by, *cols])
