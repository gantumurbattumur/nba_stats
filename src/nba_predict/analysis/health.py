"""Data-health checks shown on the dashboard (all read-only)."""

from __future__ import annotations

import sqlite3

import pandas as pd


def line_coverage_by_month(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """select g.season, substr(g.game_date, 1, 7) month, count(*) final_games,
                  sum(exists(select 1 from odds_snapshots o where o.game_id = g.game_id and o.is_closing = 1))
                      with_closing_line,
                  sum(exists(select 1 from odds_snapshots o where o.game_id = g.game_id and o.is_closing = 0
                             and o.source = 'espn')) with_open_line
           from games g where g.status = 'final' group by 1, 2 order by 2""", conn)


def fav_disagree_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    """Closing ESPN snapshots whose moneyline favorite is not the spread favorite."""
    return pd.read_sql_query(
        """select g.game_id, g.game_date, o.book, o.home_spread, o.home_ml, o.away_ml, g.home_pts, g.away_pts
           from odds_snapshots o join games g using (game_id)
           where o.source = 'espn' and o.is_closing = 1 and o.home_spread != 0
             and o.home_ml is not null and o.away_ml is not null
             and ((o.home_spread < 0) != (o.home_ml < o.away_ml)) order by g.game_date""", conn)


def bad_minutes_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    """Team rows whose MIN is not 240 + 25k (k overtimes) or differs from the opponent's."""
    return pd.read_sql_query(
        """select t.game_id, t.game_date, t.team_id, t.min, o.min opp_min
           from team_games t join team_games o on o.game_id = t.game_id and o.team_id = t.opp_team_id
           where t.min < 240 or cast(t.min - 240 as integer) % 25 != 0 or t.min != o.min
           order by t.game_date""", conn)


def derived_freshness(conn: sqlite3.Connection) -> pd.DataFrame:
    """Each derived table's build time and whether its fingerprint still matches the sources."""
    from nba_predict.analysis.enrich import source_fingerprint
    from nba_predict.config import get_settings
    from nba_predict.features.build import features_fingerprint

    meta = {r[0]: (r[1], r[2]) for r in conn.execute("select name, fingerprint, built_at from derived_meta")}
    current = {"team_games_enriched": source_fingerprint(conn),
               "game_features": features_fingerprint(conn, get_settings().windows)}
    rows = []
    for name, fp in current.items():
        stored, built = meta.get(name, (None, None))
        rows.append(dict(table=name, built_at=built, fresh=stored == fp,
                         note="up to date" if stored == fp else "stale — rebuilt on next use / `nba features`"))
    return pd.DataFrame(rows)
