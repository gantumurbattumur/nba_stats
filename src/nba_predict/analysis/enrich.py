"""``team_games_enriched``: one row per final team-game with opponent box, derived metrics,
schedule, closing odds from the team's side, and opponent strength as of the game."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3

import numpy as np
import pandas as pd

from nba_predict.analysis.metrics import add_game_metrics
from nba_predict.odds_math import american_to_decimal, decimal_to_american
from nba_predict.teams import TEAMS

BOX = ["min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast",
       "stl", "blk", "tov", "pf", "plus_minus"]
OPP_MIN_GP = 5  # opponent-strength columns are NULL before the opponent has 5 games


def _median_ml(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    return decimal_to_american(float(np.median([american_to_decimal(x) for x in s])))


def closing_lines(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per game: ``home_spread, total, home_ml, away_ml, odds_source`` from closing snapshots.
    Median across Odds API books (latest closing pull per book) if any, else the ESPN row."""
    snaps = pd.read_sql_query(
        "select game_id, source, book, pulled_at, home_spread, total, home_ml, away_ml "
        "from odds_snapshots where is_closing = 1", conn)
    cols = ["game_id", "home_spread", "total", "home_ml", "away_ml", "odds_source"]
    if snaps.empty:
        return pd.DataFrame(columns=cols)
    snaps = snaps.sort_values("pulled_at").drop_duplicates(["game_id", "source", "book"], keep="last")
    api = snaps[snaps.source == "oddsapi"]
    api = api.groupby("game_id").agg(
        home_spread=("home_spread", "median"), total=("total", "median"),
        home_ml=("home_ml", _median_ml), away_ml=("away_ml", _median_ml),
    ).reset_index().assign(odds_source="oddsapi")
    espn = snaps[(snaps.source == "espn") & ~snaps.game_id.isin(api.game_id)]
    espn = espn.drop_duplicates("game_id", keep="last").assign(odds_source="espn")
    return pd.concat([api, espn[cols]], ignore_index=True)[cols]


def _schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["team_id", "game_date"]).copy()
    d = df["_date"]
    g = df.groupby(["team_id", "season"])["_date"]
    prev = g.shift(1)
    rest = (d - prev).dt.days - 1
    df["rest_days"] = rest.clip(upper=7).fillna(7).astype(int)
    df["b2b"] = df["rest_days"] == 0
    df["three_in_four"] = (d - g.shift(2)).dt.days.le(3).fillna(False).astype(bool)
    return df


def _opp_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date record/net/rank of every team *before* each game date."""
    s = df.sort_values("_date", kind="stable")[["season", "team_id", "_date", "win", "pts", "opp_pts", "game_poss"]].copy()
    grp = s.groupby(["season", "team_id"])
    s["c_gp"] = grp.cumcount() + 1
    for c in ("win", "pts", "opp_pts", "game_poss"):
        s[f"c_{c}"] = grp[c].cumsum()
    state = s[["season", "team_id", "_date", "c_gp", "c_win", "c_pts", "c_opp_pts", "c_game_poss"]]
    grid = df[["season", "_date"]].drop_duplicates().merge(
        df[["season", "team_id"]].drop_duplicates(), on="season")
    grid = pd.merge_asof(grid.sort_values("_date"), state.sort_values("_date", kind="stable"), on="_date",
                         by=["season", "team_id"], allow_exact_matches=False, direction="backward")
    ok = grid["c_gp"] >= OPP_MIN_GP
    grid["win_pct_pre"] = (grid.c_win / grid.c_gp).where(ok)
    grid["net_pre"] = (100 * (grid.c_pts - grid.c_opp_pts) / grid.c_game_poss).where(ok)
    grid["net_rank_pre"] = grid.groupby(["season", "_date"])["net_pre"].rank(
        ascending=False, method="min")
    return grid.rename(columns={"team_id": "opp_team_id", "win_pct_pre": "opp_win_pct_pre",
                                "net_pre": "opp_net_pre", "net_rank_pre": "opp_net_rank_pre"})[
        ["season", "_date", "opp_team_id", "opp_win_pct_pre", "opp_net_pre", "opp_net_rank_pre"]]


def source_fingerprint(conn: sqlite3.Connection) -> str:
    """Cheap signature of everything ``team_games_enriched`` is derived from: final team-game
    count, latest date and box-score sums, plus closing-snapshot count/ids/line sums. Any
    ingest, box-score correction or new/changed closing line changes it."""
    tg = conn.execute(
        "select count(*), max(t.game_date), total(t.pts), total(t.fgm), total(t.fga), "
        "total(t.fg3m), total(t.fta), total(t.oreb), total(t.dreb), total(t.tov), total(t.min), "
        "total(t.is_home), group_concat(distinct t.season_type) "
        "from team_games t join games g on g.game_id = t.game_id where g.status = 'final'"
    ).fetchone()
    od = conn.execute(
        "select count(*), max(id), total(home_spread), total(total), total(home_ml), "
        "total(away_ml), max(pulled_at) from odds_snapshots where is_closing = 1"
    ).fetchone()
    return json.dumps([list(tg), list(od)])


def load_enriched(conn: sqlite3.Connection) -> pd.DataFrame:
    """``team_games_enriched`` if it was built from the current source rows, else rebuild."""
    fp = source_fingerprint(conn)
    row = conn.execute(
        "select fingerprint from derived_meta where name = 'team_games_enriched'").fetchone()
    exists = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = 'team_games_enriched'"
    ).fetchone()
    if exists and row and row[0] == fp:
        return pd.read_sql_query("select * from team_games_enriched", conn)
    return build_enriched(conn)


def build_enriched(conn: sqlite3.Connection) -> pd.DataFrame:
    fp = source_fingerprint(conn)  # taken before reading, so a concurrent write forces a rebuild
    tg = pd.read_sql_query(
        "select t.* from team_games t join games g on g.game_id = t.game_id where g.status = 'final'",
        conn)
    opp = tg[["game_id", "team_id", *BOX]].rename(
        columns={"team_id": "opp_team_id", **{c: f"opp_{c}" for c in BOX}})
    df = tg.merge(opp, on=["game_id", "opp_team_id"], how="inner")
    info = pd.DataFrame([dict(team_id=t.nba_id, team_abbr=t.abbr, conference=t.conference,
                              division=t.division) for t in TEAMS])
    df = df.merge(info, on="team_id", how="left").merge(
        info.rename(columns={"team_id": "opp_team_id", "team_abbr": "opp_abbr",
                             "conference": "opp_conference", "division": "opp_division"}),
        on="opp_team_id", how="left")
    df = add_game_metrics(df)
    df["_date"] = pd.to_datetime(df["game_date"])
    df = _schedule(df)

    lines = closing_lines(conn)
    df = df.merge(lines, on="game_id", how="left")
    home = df["is_home"] == 1
    df["team_spread"] = np.where(home, df["home_spread"], -df["home_spread"]).astype(float)
    df["team_ml"] = np.where(home, df["home_ml"], df["away_ml"]).astype(float)
    df["opp_ml"] = np.where(home, df["away_ml"], df["home_ml"]).astype(float)
    df["total_line"] = df["total"].astype(float)
    fav = pd.Series(pd.NA, index=df.index, dtype="Int64")
    fav[df.team_spread < 0] = 1
    fav[df.team_spread > 0] = 0
    by_ml = fav.isna() & df.team_ml.notna() & df.opp_ml.notna()
    fav[by_ml] = (df.team_ml[by_ml] < df.opp_ml[by_ml]).astype(int)
    df["is_favorite"] = fav
    df["cover_margin"] = df["margin"] + df["team_spread"]
    df["ats"] = np.select([df.cover_margin > 0, df.cover_margin < 0, df.cover_margin == 0],
                          ["cover", "no_cover", "push"], default=None)
    diff = df["total_pts"] - df["total_line"]
    df["ou"] = np.select([diff > 0, diff < 0, diff == 0], ["over", "under", "push"], default=None)
    df = df.drop(columns=["home_spread", "total", "home_ml", "away_ml"])

    df = df.merge(_opp_strength(df), on=["season", "_date", "opp_team_id"], how="left")
    df["month"] = df["_date"].dt.month
    df["weekday"] = df["_date"].dt.day_of_week
    df = df.drop(columns=["_date"]).sort_values(["game_date", "game_id", "is_home"]).reset_index(drop=True)
    df.to_sql("team_games_enriched", conn, if_exists="replace", index=False)
    conn.execute(
        "insert into derived_meta (name, fingerprint, built_at) values ('team_games_enriched', ?, ?) "
        "on conflict (name) do update set fingerprint = excluded.fingerprint, built_at = excluded.built_at",
        (fp, dt.datetime.now(dt.UTC).isoformat(timespec="seconds")))
    conn.commit()
    return df
