"""Leakage-safe game feature table (``game_features``) and per-call ``team_profile``.

Every feature for a game on date D uses only the team's final games with ``game_date < D``.
Columns: ``{side}_{scope}_{win}_{metric}`` with side ∈ {h, a}, scope ∈ {all, split} (split =
home team's home games / away team's road games), win ∈ {l<N>..., szn}; plus ``_gp`` and
``_adjnet`` (= net + sos), diffs ``d_* = h − a``, schedule (rest, b2b, 3in4, opener), Elo (h_elo, a_elo, elo_prob, elo_margin), market (closing)
and targets.

Window rule: ``szn`` is season-to-date only. ``l<N>`` windows reach back into the previous
season while the team has fewer than ``CROSS_SEASON_GAMES`` prior in-season games in that
scope; after that they use in-season games only. Ratings are sum-based over the window.
Post-season games are part of the same date-ordered sequence as regular-season games.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd

from nba_predict.analysis.enrich import closing_lines, load_enriched, source_fingerprint
from nba_predict.analysis.filters import FilterSpec, apply_filter
from nba_predict.analysis.metrics import aggregate
from nba_predict.config import DEFAULT_WINDOWS
from nba_predict.ingest import season_for_date
from nba_predict.models.elo import EloParams, run_elo
from nba_predict.odds_math import american_to_prob, devig_two_way
from nba_predict.teams import by_abbr

CROSS_SEASON_GAMES = 15
METRICS = ["ortg", "drtg", "net", "pace", "efg", "tov_pct", "orb_pct", "ftr", "margin",
           "win_pct", "ats_pct", "over_pct", "cover_margin", "sos"]
COMP = ["one", "pts", "opp_pts", "game_poss", "game_min", "fgm", "fg3m", "fga", "tov", "poss",
        "oreb", "opp_dreb", "fta", "margin", "win", "cover", "no_cover", "over", "under", "cm",
        "cm_n", "sos", "sos_n"]


def _components(e: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame({
        "team_id": e.team_id.astype(int), "season": e.season, "is_home": e.is_home.astype(int),
        "_date": pd.to_datetime(e.game_date),
        "one": 1.0, "win": e.win.astype(float),
        "cover": (e.ats == "cover").astype(float), "no_cover": (e.ats == "no_cover").astype(float),
        "over": (e.ou == "over").astype(float), "under": (e.ou == "under").astype(float),
        "cm": e.cover_margin.fillna(0.0).astype(float), "cm_n": e.cover_margin.notna().astype(float),
        "sos": e.opp_net_pre.fillna(0.0).astype(float), "sos_n": e.opp_net_pre.notna().astype(float),
    })
    for k in ("pts", "opp_pts", "game_poss", "game_min", "fgm", "fg3m", "fga", "tov", "poss",
              "oreb", "opp_dreb", "fta", "margin"):
        c[k] = e[k].astype(float)
    return c.sort_values(["team_id", "_date"], kind="stable").reset_index(drop=True)


def _ratio(a, b):
    b = b.where(b != 0)
    return a / b


def _metrics_from_sums(s: pd.DataFrame) -> pd.DataFrame:
    """Window component sums (columns = COMP) → metric columns + gp + adjnet."""
    out = pd.DataFrame(index=s.index)
    out["ortg"] = 100 * _ratio(s.pts, s.game_poss)
    out["drtg"] = 100 * _ratio(s.opp_pts, s.game_poss)
    out["net"] = out.ortg - out.drtg
    out["pace"] = 48 * _ratio(s.game_poss, s.game_min / 5)
    out["efg"] = _ratio(s.fgm + 0.5 * s.fg3m, s.fga)
    out["tov_pct"] = _ratio(s.tov, s.poss)
    out["orb_pct"] = _ratio(s.oreb, s.oreb + s.opp_dreb)
    out["ftr"] = _ratio(s.fta, s.fga)
    out["margin"] = _ratio(s.margin, s.one)
    out["win_pct"] = _ratio(s.win, s.one)
    out["ats_pct"] = _ratio(s.cover, s.cover + s.no_cover)
    out["over_pct"] = _ratio(s.over, s.over + s.under)
    out["cover_margin"] = _ratio(s.cm, s.cm_n)
    out["sos"] = _ratio(s.sos, s.sos_n)
    out["gp"] = s.one.fillna(0).astype(int)
    out["adjnet"] = out.net + out.sos
    return out


def _states(seq: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """State *after* each game of a team's scope sequence: cross-season and in-season rolling
    sums per window, season-to-date sums, and the in-season game count."""
    st = seq[["team_id", "season", "_date"]].copy()
    by_team = seq.groupby("team_id")[COMP]
    by_ts = seq.groupby(["team_id", "season"])[COMP]
    parts = [st]
    for n in windows:
        x = by_team.rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
        i = by_ts.rolling(n, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
        parts += [x.add_prefix(f"x{n}_"), i.add_prefix(f"i{n}_")]
    parts.append(by_ts.cumsum().add_prefix("szn_"))
    out = pd.concat(parts, axis=1)
    out["c_in"] = seq.groupby(["team_id", "season"]).cumcount() + 1
    return out.rename(columns={"season": "s_last"})


def _rows_where(cond: pd.Series, a: pd.DataFrame, b) -> pd.DataFrame:
    """Row-wise select: a where cond else b (b: same-shape frame or scalar)."""
    bv = b.to_numpy() if isinstance(b, pd.DataFrame) else b
    return pd.DataFrame(np.where(cond.to_numpy()[:, None], a.to_numpy(dtype=float), bv),
                        index=a.index, columns=a.columns)


def _side_features(targets: pd.DataFrame, states: pd.DataFrame, windows, prefix: str) -> pd.DataFrame:
    """targets: game_id, team_id, season, _date → wide window features with ``prefix``."""
    t = targets.sort_values("_date")
    m = pd.merge_asof(t, states.sort_values("_date", kind="stable"), on="_date", by="team_id",
                      allow_exact_matches=False, direction="backward")
    same = m.s_last == m.season
    c = m.c_in.where(same, 0).fillna(0)
    blocks = [pd.DataFrame({"game_id": m.game_id.values, "_c": c.values}, index=m.index)]
    for n in windows:
        cross = m[[f"x{n}_{k}" for k in COMP]].set_axis(COMP, axis=1)
        inseason = _rows_where(same, m[[f"i{n}_{k}" for k in COMP]].set_axis(COMP, axis=1), np.nan)
        sums = _rows_where(c < CROSS_SEASON_GAMES, cross, inseason)
        blocks.append(_metrics_from_sums(sums.fillna({"one": 0})).add_prefix(f"{prefix}l{n}_"))
    szn = _rows_where(same, m[[f"szn_{k}" for k in COMP]].set_axis(COMP, axis=1), np.nan)
    blocks.append(_metrics_from_sums(szn.fillna({"one": 0})).add_prefix(f"{prefix}szn_"))
    return pd.concat(blocks, axis=1).set_index("game_id")


def _schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team): rest days, b2b, 3-in-4 from all final+scheduled games."""
    g = games[games.status.isin(["final", "scheduled"])]
    long = pd.concat([
        g[["game_id", "season", "_date"]].assign(team_id=g.home_team_id, side="h"),
        g[["game_id", "season", "_date"]].assign(team_id=g.away_team_id, side="a"),
    ]).sort_values(["team_id", "_date"])
    grp = long.groupby(["team_id", "season"])["_date"]
    long["rest"] = ((long._date - grp.shift(1)).dt.days - 1).clip(lower=0, upper=7).fillna(7).astype(int)
    long["b2b"] = (long.rest == 0).astype(int)
    long["opener"] = grp.shift(1).isna().astype(int)  # first game of the team's season (rest=7)
    long["3in4"] = (long._date - grp.shift(2)).dt.days.le(3).fillna(False).astype(int)
    out = {}
    for side in ("h", "a"):
        s = long[long.side == side].set_index("game_id")[["rest", "b2b", "3in4", "opener"]]
        out[side] = s.add_prefix(f"{side}_")
    return out["h"].join(out["a"], how="outer")


def _market(conn: sqlite3.Connection) -> pd.DataFrame:
    lines = closing_lines(conn).set_index("game_id")
    m = pd.DataFrame(index=lines.index)
    m["m_home_spread"] = lines.home_spread.astype(float)
    m["m_total"] = lines.total.astype(float)
    m["m_home_ml"] = lines.home_ml.astype(float)
    m["m_away_ml"] = lines.away_ml.astype(float)

    def novig(r):
        if pd.isna(r.m_home_ml) or pd.isna(r.m_away_ml):
            return np.nan
        return devig_two_way(american_to_prob(r.m_home_ml), american_to_prob(r.m_away_ml))[0]

    m["m_home_novig"] = m.apply(novig, axis=1) if len(m) else pd.Series(dtype=float)
    return m


def build_feature_table(conn: sqlite3.Connection, windows=DEFAULT_WINDOWS) -> pd.DataFrame:
    windows = tuple(int(w) for w in windows)
    e = load_enriched(conn)
    fp = features_fingerprint(conn, windows)  # after enriched is fresh; before reading games
    games = pd.read_sql_query(
        "select game_id, season, season_type, game_date, status, home_team_id, away_team_id, "
        "home_pts, away_pts from games", conn)
    games["_date"] = pd.to_datetime(games.game_date)
    comp = _components(e)
    scopes = {"all": comp, "home": comp[comp.is_home == 1], "away": comp[comp.is_home == 0]}
    states = {k: _states(v.reset_index(drop=True), windows) for k, v in scopes.items()}

    base = games.drop(columns=["_date"]).set_index("game_id")
    parts = [base]
    for side, tcol, split in (("h", "home_team_id", "home"), ("a", "away_team_id", "away")):
        tgt = games[["game_id", "season", "_date"]].assign(team_id=games[tcol].astype(int))
        allf = _side_features(tgt, states["all"], windows, f"{side}_all_")
        allf = allf.rename(columns={"_c": f"{side}_gp_szn"}).astype({f"{side}_gp_szn": int})
        splf = _side_features(tgt, states[split], windows, f"{side}_split_").drop(columns="_c")
        parts += [allf.reindex(base.index), splf.reindex(base.index)]
    parts.append(_schedule(games).reindex(base.index))
    out = pd.concat(parts, axis=1)

    diffs = {}
    for scope in ("all", "split"):
        for w in [f"l{n}" for n in windows] + ["szn"]:
            for mkey in METRICS + ["adjnet"]:
                diffs[f"d_{scope}_{w}_{mkey}"] = out[f"h_{scope}_{w}_{mkey}"] - out[f"a_{scope}_{w}_{mkey}"]
    elo = run_elo(games, EloParams())[["game_id", "h_elo", "a_elo", "elo_prob", "elo_margin"]].set_index("game_id")
    out = pd.concat([out, pd.DataFrame(diffs, index=out.index)], axis=1).join(_market(conn)).join(elo).copy()

    final = out.status == "final"
    out["margin"] = (out.home_pts - out.away_pts).where(final)
    out["total"] = (out.home_pts + out.away_pts).where(final)
    out["home_win"] = (out.home_pts > out.away_pts).astype(float).where(final)
    out = out.drop(columns=["home_pts", "away_pts"]).reset_index().sort_values(
        ["game_date", "game_id"]).reset_index(drop=True)
    out.to_sql("game_features", conn, if_exists="replace", index=False)
    conn.execute(
        "insert into derived_meta (name, fingerprint, built_at) values ('game_features', ?, ?) "
        "on conflict (name) do update set fingerprint = excluded.fingerprint, built_at = excluded.built_at",
        (fp, dt.datetime.now(dt.UTC).isoformat(timespec="seconds")))
    conn.commit()
    return out


def features_fingerprint(conn: sqlite3.Connection, windows) -> str:
    """Signature of what ``game_features`` derives from: the enriched sources, every game row
    (ids, dates, teams, status, scores — so new/rescheduled/postponed/finished games count)
    and the windows."""
    h = hashlib.sha256()
    for r in conn.execute(
            "select game_id, game_date, status, home_team_id, away_team_id, home_pts, away_pts, "
            "season, season_type from games order by game_id"):
        h.update(repr(tuple(r)).encode())
    return json.dumps([source_fingerprint(conn), h.hexdigest(), [int(w) for w in windows],
                       EloParams().model_dump()])  # Elo columns change with the Elo params


def load_features(conn: sqlite3.Connection, windows=DEFAULT_WINDOWS) -> pd.DataFrame:
    """``game_features`` if built from the current sources with these windows, else rebuild
    (which also rebuilds a stale ``team_games_enriched``)."""
    row = conn.execute("select fingerprint from derived_meta where name = 'game_features'").fetchone()
    exists = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = 'game_features'").fetchone()
    if exists and row and row[0] == features_fingerprint(conn, windows):
        return pd.read_sql_query("select * from game_features", conn)
    return build_feature_table(conn, windows)


def team_profile(enriched: pd.DataFrame, team_abbr: str, as_of: dt.date, n: int | None,
                 venue: str = "all", spec: FilterSpec | None = None) -> dict[str, float]:
    """Slow per-call profile of a team's games before ``as_of`` (``n=None`` = season-to-date),
    optionally restricted by ``venue`` and an extra ``FilterSpec``. Same window rule as the
    feature table. Returns metric keys + ``gp``, ``sos``, ``adjnet``."""
    team = by_abbr(team_abbr)
    g = enriched[(enriched.team_id == team.nba_id) & (enriched.game_date < as_of.isoformat())]
    if venue != "all":
        g = g[g.is_home == (1 if venue == "home" else 0)]
    if spec is not None:
        g = apply_filter(g, spec)
    g = g.sort_values("game_date")
    season = season_for_date(as_of)
    in_season = g[g.season == season]
    if n is None:
        sel = in_season
    elif len(in_season) < CROSS_SEASON_GAMES:
        sel = g.tail(n)
    else:
        sel = in_season.tail(n)
    keys = [k for k in METRICS if k != "sos"]
    row = aggregate(sel, keys).iloc[0].to_dict()
    sos = sel.opp_net_pre.dropna()
    row["sos"] = float(sos.mean()) if len(sos) else np.nan
    row["adjnet"] = row["net"] + row["sos"]
    row["gp"] = int(row["gp"])
    return {k: (float(v) if k != "gp" else v) for k, v in row.items()}
