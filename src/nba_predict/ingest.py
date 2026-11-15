"""Ingest stages: nba_api team logs + slate (Task 3), ESPN lines (Task 4)."""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3

import pandas as pd

from nba_predict.db import upsert

log = logging.getLogger(__name__)

BOX_COLS = ["min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb",
            "ast", "stl", "blk", "tov", "pf"]


def season_for_date(d: dt.date) -> str:
    """NBA season label for a date: Aug–Dec → 'YYYY-(YY+1)', Jan–Jul → previous year's season."""
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _int(v):
    return None if pd.isna(v) else int(v)


def _ambiguous_games(df: pd.DataFrame) -> pd.DataFrame:
    """game_id/GAME_DATE of games whose MATCHUP doesn't identify exactly one home team
    (neutral-site games, e.g. Mexico City, list both teams as '@')."""
    n_home = df["MATCHUP"].str.contains(" vs. ", regex=False).groupby(df["GAME_ID"]).sum()
    ids = n_home[n_home != 1].index
    return df[df["GAME_ID"].isin(ids)].drop_duplicates("GAME_ID")[["GAME_ID", "GAME_DATE"]]


def parse_team_logs(
    df: pd.DataFrame, season: str, home_ids: dict[str, int] | None = None
) -> tuple[list[dict], list[dict]]:
    """LeagueGameLog team rows → (games rows, team_games rows).

    Home team = row whose MATCHUP contains " vs. "; for neutral-site games (no single " vs.")
    ``home_ids[game_id]`` decides, else the game is skipped with a warning. Only games with
    exactly two team rows and a W/L on both are emitted (in-progress games have WL = NaN).
    """
    home_ids = home_ids or {}
    games: list[dict] = []
    tgs: list[dict] = []
    for gid, grp in df.groupby("GAME_ID", sort=False):
        if len(grp) != 2 or grp["WL"].isna().any():
            log.warning("skipping game %s: %d team rows / missing WL", gid, len(grp))
            continue
        home_mask = grp["MATCHUP"].str.contains(" vs. ", regex=False)
        if home_mask.sum() != 1 and str(gid) in home_ids:
            home_mask = grp["TEAM_ID"].astype(int) == int(home_ids[str(gid)])
        if home_mask.sum() != 1:
            log.warning("skipping game %s: cannot determine home team", gid)
            continue
        home = grp[home_mask].iloc[0]
        away = grp[~home_mask].iloc[0]
        game_date = str(home["GAME_DATE"])[:10]
        games.append(dict(
            game_id=str(gid), season=season, game_date=game_date,
            home_team_id=int(home["TEAM_ID"]), away_team_id=int(away["TEAM_ID"]),
            home_pts=int(home["PTS"]), away_pts=int(away["PTS"]), status="final",
        ))
        for row, opp, is_home in ((home, away, 1), (away, home, 0)):
            tg = dict(
                game_id=str(gid), team_id=int(row["TEAM_ID"]), opp_team_id=int(opp["TEAM_ID"]),
                season=season, game_date=game_date, is_home=is_home, wl=row["WL"],
            )
            tg.update({c: _int(row[c.upper()]) for c in BOX_COLS})
            pm = row["PLUS_MINUS"]
            tg["plus_minus"] = None if pd.isna(pm) else float(pm)
            tgs.append(tg)
    return games, tgs


def ingest_season(conn: sqlite3.Connection, client, season: str) -> int:
    """Fetch and upsert a season's team game logs. Returns team_games rows written.

    Neutral-site games get their designated home team from ``games`` (if already known) or
    from the scoreboard for that date (one extra call per such date).
    """
    df = client.season_team_logs(season)
    home_ids: dict[str, int] = {}
    amb = _ambiguous_games(df)
    for gid, gdate in amb.itertuples(index=False):
        row = conn.execute("select home_team_id from games where game_id=?", (str(gid),)).fetchone()
        if row and row[0] is not None:
            home_ids[str(gid)] = int(row[0])
    for gdate in sorted({str(d)[:10] for g, d in amb.itertuples(index=False) if str(g) not in home_ids}):
        try:
            for r in client.slate(dt.date.fromisoformat(gdate)):
                home_ids[str(r["game_id"])] = int(r["home_team_id"])
        except Exception as e:  # noqa: BLE001 - lookup is best effort; game is skipped
            log.warning("home-team lookup for %s failed: %s", gdate, e)
    games, tgs = parse_team_logs(df, season, home_ids)
    upsert(conn, "games", games, ["game_id"])
    return upsert(conn, "team_games", tgs, ["game_id", "team_id"])


def ingest_slate(conn: sqlite3.Connection, client, date: dt.date) -> int:
    """Upsert the day's games. Never touches a row already ``final``.

    Final status and scores come only from ``ingest_season`` (which has the box scores), so a
    slate game reported final but not yet in the game log is stored as ``scheduled``.
    """
    rows = client.slate(date)
    sql = """
        INSERT INTO games (game_id, season, game_date, home_team_id, away_team_id, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (game_id) DO UPDATE SET
            game_date = excluded.game_date, home_team_id = excluded.home_team_id,
            away_team_id = excluded.away_team_id, status = excluded.status
        WHERE games.status IS NOT 'final'
    """
    params = []
    for r in rows:
        status = "scheduled" if r["status"] == "final" else r["status"]
        gdate = r["game_date"]
        params.append((r["game_id"], season_for_date(dt.date.fromisoformat(gdate)), gdate,
                       r["home_team_id"], r["away_team_id"], status))
    conn.executemany(sql, params)
    conn.commit()
    return len(rows)
