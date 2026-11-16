"""Open → close line movement for games with an ESPN opening snapshot (is_closing = 0)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd


def line_moves(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per final game with ESPN open and close lines: moves (home perspective), which side
    the spread moved toward, and whether that side covered the closing and the opening spread."""
    d = pd.read_sql_query(
        """select g.game_id, g.season, g.game_date, g.home_team_id, g.away_team_id, g.home_pts, g.away_pts,
                  o.book, o.home_spread open_spread, o.total open_total, c.home_spread close_spread,
                  c.total close_total
           from games g
           join odds_snapshots o on o.game_id = g.game_id and o.source = 'espn' and o.is_closing = 0
           join odds_snapshots c on c.game_id = g.game_id and c.source = 'espn' and c.is_closing = 1
           where g.status = 'final'""", conn)
    if d.empty:
        return d
    d["margin"] = d.home_pts - d.away_pts
    d["total_pts"] = d.home_pts + d.away_pts
    d["spread_move"] = d.close_spread - d.open_spread  # < 0: home became more favored
    d["total_move"] = d.close_total - d.open_total
    d["moved_toward"] = np.select([d.spread_move < 0, d.spread_move > 0], ["home", "away"], "no move")
    sign = np.where(d.moved_toward == "home", 1, -1)

    def covered(line):  # did the side the line moved toward cover `line` (home perspective)?
        diff = sign * (d.margin + line)
        return np.select([diff > 0, diff < 0], ["cover", "no_cover"], "push")

    d["cover_close"] = np.where(d.moved_toward == "no move", None, covered(d.close_spread))
    d["cover_open"] = np.where(d.moved_toward == "no move", None, covered(d.open_spread))
    d["total_dir"] = np.select([d.total_move > 0, d.total_move < 0], ["up", "down"], "no move")
    d["ou_close"] = np.select([d.total_pts > d.close_total, d.total_pts < d.close_total], ["over", "under"], "push")
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
