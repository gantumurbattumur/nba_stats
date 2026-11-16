"""Open → close line movement for games with an ESPN opening snapshot (is_closing = 0)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd


def line_moves(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per final game: earliest open/latest close from the same ESPN provider.
    If multiple providers have pairs, use the most recently closed pair. Missing lines stay unknown.
    """
    d = pd.read_sql_query(
        """with snapshots as (
               select *, row_number() over (
                   partition by game_id, book, is_closing
                   order by case when is_closing = 0 then pulled_at end asc,
                            case when is_closing = 1 then pulled_at end desc, id desc
               ) as snapshot_rank
               from odds_snapshots where source = 'espn'
           ), pairs as (
           select g.game_id, g.season, g.game_date, g.home_team_id, g.away_team_id, g.home_pts, g.away_pts,
                  o.book, o.home_spread open_spread, o.total open_total, c.home_spread close_spread,
                  c.total close_total,
                  row_number() over (partition by g.game_id order by c.pulled_at desc, c.id desc) as pair_rank
           from games g
           join snapshots o on o.game_id = g.game_id and o.is_closing = 0 and o.snapshot_rank = 1
           join snapshots c on c.game_id = g.game_id and c.is_closing = 1 and c.snapshot_rank = 1
                            and c.book = o.book and c.pulled_at >= o.pulled_at
           where g.status = 'final'
           ) select * from pairs where pair_rank = 1""", conn).drop(columns="pair_rank")
    if d.empty:
        return d
    for col in ("open_spread", "close_spread", "open_total", "close_total"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["margin"] = d.home_pts - d.away_pts
    d["total_pts"] = d.home_pts + d.away_pts
    d["spread_move"] = d.close_spread - d.open_spread  # < 0: home became more favored
    d["total_move"] = d.close_total - d.open_total
    d["moved_toward"] = np.select([d.spread_move < 0, d.spread_move > 0, d.spread_move == 0],
                                  ["home", "away", "no move"], "unknown")
    sign = np.where(d.moved_toward == "home", 1, -1)

    def covered(line):  # did the side the line moved toward cover `line` (home perspective)?
        diff = sign * (d.margin + line)
        return np.where(line.notna(), np.select([diff > 0, diff < 0], ["cover", "no_cover"], "push"), None)

    directed = d.moved_toward.isin(["home", "away"])
    d["cover_close"] = np.where(directed, covered(d.close_spread), None)
    d["cover_open"] = np.where(directed, covered(d.open_spread), None)
    d["total_dir"] = np.select([d.total_move > 0, d.total_move < 0, d.total_move == 0],
                               ["up", "down", "no move"], "unknown")
    d["ou_close"] = np.where(d.close_total.notna(), np.select(
        [d.total_pts > d.close_total, d.total_pts < d.close_total], ["over", "under"], "push"), None)
    return d


def summarize_moves(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(spread table by direction, total table by direction)."""
    def rate(s, val):
        dec = s[s.isin([val, "no_cover" if val == "cover" else "under"])]
        return (dec == val).mean() if len(dec) else np.nan

    sp = d.groupby("moved_toward").agg(
        games=("game_id", "size"), mean_move=("spread_move", "mean"),
        cover_rate_close=("cover_close", lambda s: rate(s, "cover")),
        cover_rate_open=("cover_open", lambda s: rate(s, "cover"))).reset_index()
    tt = d.groupby("total_dir").agg(games=("game_id", "size"), mean_move=("total_move", "mean"),
                                    over_rate_close=("ou_close", lambda s: rate(s, "over"))).reset_index()
    return sp, tt
