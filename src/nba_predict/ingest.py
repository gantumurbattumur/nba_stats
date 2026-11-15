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


# nba_api season_type_all_star → our season_type label
SEASON_TYPES = {"Regular Season": "regular", "PlayIn": "playin", "Playoffs": "playoffs"}
_PREFIX_TYPE = {"002": "regular", "004": "playoffs", "005": "playin"}


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
    df: pd.DataFrame, season: str, home_ids: dict[str, int] | None = None,
    season_type: str = "regular",
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
            season_type=season_type,
        ))
        for row, opp, is_home in ((home, away, 1), (away, home, 0)):
            tg = dict(
                game_id=str(gid), team_id=int(row["TEAM_ID"]), opp_team_id=int(opp["TEAM_ID"]),
                season=season, game_date=game_date, is_home=is_home, wl=row["WL"],
                season_type=season_type,
            )
            tg.update({c: _int(row[c.upper()]) for c in BOX_COLS})
            pm = row["PLUS_MINUS"]
            tg["plus_minus"] = None if pd.isna(pm) else float(pm)
            tgs.append(tg)
    return games, tgs


def ingest_season(
    conn: sqlite3.Connection, client, season: str,
    season_types: tuple[str, ...] = tuple(SEASON_TYPES),
) -> int:
    """Fetch and upsert a season's team game logs for each nba_api season type (default:
    regular season, play-in, playoffs). Returns team_games rows written.

    Neutral-site games get their designated home team from ``games`` (if already known) or
    from the scoreboard for that date (one extra call per such date).
    """
    n = 0
    for st in season_types:
        df = client.season_team_logs(season, season_type=st)
        if df is None or df.empty:
            continue
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
        games, tgs = parse_team_logs(df, season, home_ids, season_type=SEASON_TYPES[st])
        upsert(conn, "games", games, ["game_id"])
        n += upsert(conn, "team_games", tgs, ["game_id", "team_id"])
    return n


# NBA game-id prefixes: 001 preseason, 002 regular season, 003 All-Star, 004 playoffs,
# 005 play-in, 006 NBA Cup final. Only 002/004/005 are ingested with box scores; the Cup final
# (only in the 'IST' season type, not counted in standings) is skipped like preseason/All-Star.
_NON_SCORED_PREFIXES = ("001", "003", "006")


def ingest_slate(conn: sqlite3.Connection, client, date: dt.date) -> int:
    """Upsert the day's games. Never touches a row already ``final``.

    Final status and scores come only from ``ingest_season`` (which has the box scores), so a
    slate game reported final but not yet in the game log is stored as ``scheduled``.
    """
    rows = client.slate(date)
    sql = """
        INSERT INTO games (game_id, season, game_date, home_team_id, away_team_id, status,
                           season_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (game_id) DO UPDATE SET
            game_date = excluded.game_date, home_team_id = excluded.home_team_id,
            away_team_id = excluded.away_team_id, status = excluded.status
        WHERE games.status IS NOT 'final'
    """
    params = []
    for r in rows:
        if str(r["game_id"])[:3] in _NON_SCORED_PREFIXES:
            log.info("slate: skipping preseason/All-Star/Cup-final game %s", r["game_id"])
            continue
        status = "scheduled" if r["status"] == "final" else r["status"]
        gdate = r["game_date"]
        params.append((r["game_id"], season_for_date(dt.date.fromisoformat(gdate)), gdate,
                       r["home_team_id"], r["away_team_id"], status,
                       _PREFIX_TYPE.get(str(r["game_id"])[:3], "regular")))
    conn.executemany(sql, params)
    conn.commit()
    return len(params)


def ingest_espn_lines(
    conn: sqlite3.Connection, client, date_from: dt.date, date_to: dt.date, only_missing: bool = True,
    with_open: bool = False,
) -> int:
    """Attach ESPN event ids and lines to ``games`` for each date in the range.

    Matches events by (ET game_date, home team via ``by_espn_abbr``). Only completed events
    are fetched. The summary ``pickcenter`` line is stored as the closing line. When it is
    empty, the ESPN core odds API is queried (first non-live provider): its close is stored as
    the closing line (``is_closing=1``, ``<date>T23:59:59Z``) and its open as an extra snapshot
    (``is_closing=0``, ``<date>T12:00:00Z``). ``with_open`` also fetches the core API for games
    that already have a pickcenter close, to add their open line. ``games.espn_core_checked``
    marks games whose core odds were queried, so resumes never repeat a call.

    With ``only_missing``: unlinked games go through scoreboard → summary (→ core); linked games
    still lacking a close (or an open, with ``with_open``) and not yet core-checked go straight
    to the core API; everything else — and dates with nothing to do — makes no calls.
    Returns snapshots written (closes + opens).
    """
    from nba_predict.data.espn_client import EspnNotFound, parse_core_odds, parse_pickcenter
    from nba_predict.teams import by_espn_abbr

    has = ("exists (select 1 from odds_snapshots o where o.game_id = games.game_id and o.source = 'espn'"
           " and o.is_closing = {})")
    need_sql = "select game_id, home_team_id, away_team_id from games where game_date = ?"
    if only_missing:
        need_sql += """ and espn_event_id is null and not exists (
            select 1 from odds_snapshots o where o.game_id = games.game_id and o.source = 'espn')"""
    core_sql = ("select game_id, espn_event_id from games where game_date = ? and status = 'final' "
                "and espn_event_id is not null and espn_core_checked = 0 and (not " + has.format(1)
                + (" or not " + has.format(0) if with_open else "") + ")")

    def has_espn(gid: str, closing: int) -> bool:
        return conn.execute("select 1 from odds_snapshots where game_id = ? and source = 'espn' "
                            "and is_closing = ? limit 1", (gid, closing)).fetchone() is not None

    def store(gid: str, ds: str, book: str, fields: dict | None, closing: bool) -> int:
        if not fields:
            return 0
        upsert(conn, "odds_snapshots", [dict(
            game_id=gid, source="espn", book=book, pulled_at=f"{ds}T{'23:59:59' if closing else '12:00:00'}Z",
            is_closing=int(closing), **fields)], ["game_id", "source", "book", "pulled_at"])
        return 1

    def from_core(gid: str, event_id: str, ds: str, need_close: bool) -> int:
        try:
            core = parse_core_odds(client.core_odds(event_id))
        except EspnNotFound as e:
            log.warning("ESPN core odds for event %s (game %s) not found: %s", event_id, gid, e)
            core = None
        conn.execute("update games set espn_core_checked = 1 where game_id = ?", (gid,))
        if core is None:
            return 0
        n = store(gid, ds, core["book"], core["close"], True) if need_close else 0
        return n + (0 if has_espn(gid, 0) else store(gid, ds, core["book"], core["open"], False))

    written = 0
    d = date_from
    while d <= date_to:
        ds = d.isoformat()
        d += dt.timedelta(days=1)
        if only_missing:  # linked games: core API only, no scoreboard/summary calls
            for r in conn.execute(core_sql, (ds,)).fetchall():
                written += from_core(r["game_id"], r["espn_event_id"], ds, need_close=not has_espn(r["game_id"], 1))
                conn.commit()
        rows = conn.execute(need_sql, (ds,)).fetchall()
        need = {int(r["home_team_id"]): r["game_id"] for r in rows}
        # (espn home, espn away) as NBA (away, home): ESPN disagrees on home/away for this game
        flipped = {(int(r["away_team_id"]), int(r["home_team_id"])): r["game_id"] for r in rows}
        if not need:
            continue
        for ev in client.events(dt.date.fromisoformat(ds)):
            team = by_espn_abbr(ev["home_espn_abbr"])
            if ev["game_date"] != ds or team is None:
                if team is None:
                    log.warning("ESPN event %s: unknown home abbr %r", ev["espn_event_id"], ev["home_espn_abbr"])
                continue
            gid = need.get(team.nba_id)
            if gid is None:
                away = by_espn_abbr(ev["away_espn_abbr"])
                fgid = flipped.get((team.nba_id, away.nba_id)) if away else None
                if fgid is not None:
                    # attaching this line would invert home_spread / ML sides; leave it unlinked
                    log.warning("ESPN event %s: home/away reversed vs NBA game %s; skipped",
                                ev["espn_event_id"], fgid)
                continue
            if not ev["completed"]:
                continue
            try:
                line = parse_pickcenter(client.summary(ev["espn_event_id"]))
            except EspnNotFound as e:
                # permanent gap at the source: link the event (so resume skips it), no line
                log.warning("ESPN event %s (game %s): summary not found, no line: %s",
                            ev["espn_event_id"], gid, e)
                line = None
            conn.execute("update games set espn_event_id = ? where game_id = ?", (ev["espn_event_id"], gid))
            if line is not None:
                book = line.pop("book")
                written += store(gid, ds, book, line, True)
            if line is None or with_open:
                written += from_core(gid, ev["espn_event_id"], ds, need_close=line is None)
            conn.commit()
    return written


def coverage(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per (season, season_type) data coverage for ``nba status``.

    ``fav_disagree`` counts ESPN snapshots whose moneyline favorite (lower ML) is not the
    spread favorite (negative home_spread = home) — a data-quality signal; pick'ems excluded.
    """
    return pd.read_sql_query(
        """
        select g.season, g.season_type,
               sum(g.status = 'final') as final_games,
               sum(g.status = 'scheduled') as scheduled,
               sum(g.status = 'postponed') as postponed,
               (select count(*) from team_games t
                 where t.season = g.season and t.season_type = g.season_type) as team_games,
               count(g.espn_event_id) as espn_linked,
               count(e.game_id) as espn_lines,
               coalesce(sum(e.home_spread != 0 and e.home_ml is not null and e.away_ml is not null
                   and ((e.home_spread < 0) != (e.home_ml < e.away_ml))), 0) as fav_disagree
        from games g
        left join (select game_id, max(home_spread) as home_spread, max(home_ml) as home_ml,
                          max(away_ml) as away_ml
                   from odds_snapshots where source = 'espn' group by game_id) e
               on e.game_id = g.game_id
        group by g.season, g.season_type
        order by g.season, g.season_type
        """,
        conn,
    )


def ingest_odds(conn: sqlite3.Connection, client, closing: bool = False) -> int:
    """One Odds API pull → ``odds_snapshots`` (source 'oddsapi', shared ``pulled_at``) plus an
    ``api_usage`` row. Events match ``games`` by the **Eastern** date of ``commence_time`` and
    the home team; unmapped team names / unmatched games are logged and skipped. Returns
    snapshots written."""
    from zoneinfo import ZoneInfo

    from nba_predict.data.odds_client import parse_events
    from nba_predict.teams import by_odds_name

    events, usage = client.nba_odds()
    pulled_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    et = ZoneInfo("America/New_York")
    snaps = []
    warned: set[str] = set()
    for r in parse_events(events):
        home, away = by_odds_name(r["home_name"]), by_odds_name(r["away_name"])
        if home is None or away is None:
            if r["event_id"] not in warned:
                log.warning("Odds API event %s: unmapped team name(s) %r / %r; skipped",
                            r["event_id"], r["home_name"], r["away_name"])
                warned.add(r["event_id"])
            continue
        tip = dt.datetime.fromisoformat(r["commence_time"].replace("Z", "+00:00"))
        gdate = tip.astimezone(et).date().isoformat()
        g = conn.execute("select game_id from games where game_date = ? and home_team_id = ?",
                         (gdate, home.nba_id)).fetchone()
        if g is None:
            if r["event_id"] not in warned:
                log.warning("Odds API event %s (%s @ %s, %s ET): no game in DB; run `nba slate`",
                            r["event_id"], away.abbr, home.abbr, gdate)
                warned.add(r["event_id"])
            continue
        snaps.append(dict(game_id=g[0], source="oddsapi", book=r["book"], pulled_at=pulled_at,
                          is_closing=int(closing),
                          **{k: r[k] for k in ("home_ml", "away_ml", "home_spread", "home_spread_price",
                                               "away_spread_price", "total", "over_price", "under_price")}))
    n = upsert(conn, "odds_snapshots", snaps, ["game_id", "source", "book", "pulled_at"])
    conn.execute("insert into api_usage (at, source, credits_used, credits_remaining) values (?, ?, ?, ?)",
                 (pulled_at, "oddsapi", usage.get("last"), usage.get("remaining")))
    conn.commit()
    return n
