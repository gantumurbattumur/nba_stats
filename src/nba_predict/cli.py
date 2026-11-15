"""``nba`` command line."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Annotated

import typer

from nba_predict.config import get_settings
from nba_predict.data.nba_client import IngestError, NbaClient, today_et
from nba_predict.db import connect, init_schema
from nba_predict import ingest as ingest_mod

app = typer.Typer(help="NBA stats & odds analysis (analysis only).", no_args_is_help=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _parse_date(s: str | None, default: dt.date) -> dt.date:
    return dt.date.fromisoformat(s) if s else default


def _conn():
    conn = connect()
    init_schema(conn)
    return conn


@app.command("init-db")
def init_db() -> None:
    """Create the SQLite schema (idempotent)."""
    s = get_settings()
    conn = _conn()
    n = conn.execute("select count(*) from sqlite_master where type='table'").fetchone()[0]
    typer.echo(f"schema ready at {s.db_path} ({n} tables)")


@app.command()
def ingest(
    season: Annotated[list[str] | None, typer.Option("--season", help="e.g. 2026-27; repeatable")] = None,
) -> None:
    """Ingest team game logs for one or more seasons (default: current season)."""
    s = get_settings()
    seasons = season or [s.current_season]
    conn = _conn()
    client = NbaClient(delay_s=s.nba_delay_s)
    for sz in seasons:
        try:
            n = ingest_mod.ingest_season(conn, client, sz)
        except IngestError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        total = conn.execute("select count(*) from team_games where season=?", (sz,)).fetchone()[0]
        games = conn.execute("select count(*) from games where season=? and status='final'", (sz,)).fetchone()[0]
        typer.echo(f"{sz}: wrote {n} team_games rows; {total} team_games / {games} final games in DB")


@app.command()
def slate(date: Annotated[str | None, typer.Option(help="YYYY-MM-DD (ET), default today")] = None) -> None:
    """Ingest the day's schedule into ``games``."""
    s = get_settings()
    d = _parse_date(date, today_et())
    conn = _conn()
    try:
        n = ingest_mod.ingest_slate(conn, NbaClient(delay_s=s.nba_delay_s), d)
    except IngestError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    by = conn.execute(
        "select status, count(*) from games where game_date=? group by status", (d.isoformat(),)
    ).fetchall()
    typer.echo(f"{d}: {n} games on slate; in DB: " + ", ".join(f"{r[0]}={r[1]}" for r in by))


@app.command()
def espn(
    date_from: Annotated[str | None, typer.Option("--from", help="YYYY-MM-DD, default yesterday (ET)")] = None,
    date_to: Annotated[str | None, typer.Option("--to", help="YYYY-MM-DD, default --from")] = None,
    refetch: Annotated[bool, typer.Option(help="re-fetch games already fetched")] = False,
) -> None:
    """Backfill ESPN past-game closing lines (resumable; ~1 request/s)."""
    from nba_predict.data.espn_client import EspnClient

    s = get_settings()
    d0 = _parse_date(date_from, today_et() - dt.timedelta(days=1))
    d1 = _parse_date(date_to, d0)
    conn = _conn()
    try:
        n = ingest_mod.ingest_espn_lines(conn, EspnClient(delay_s=s.espn_delay_s), d0, d1,
                                         only_missing=not refetch)
    except IngestError as e:
        typer.echo(f"error: {e} (re-run to resume)", err=True)
        raise typer.Exit(1)
    r = conn.execute(
        """select count(*) as games, count(g.espn_event_id) as linked,
                  sum(exists(select 1 from odds_snapshots o where o.game_id=g.game_id and o.source='espn')) as lined
           from games g where g.status='final' and g.game_date between ? and ?""",
        (d0.isoformat(), d1.isoformat()),
    ).fetchone()
    typer.echo(f"{d0}..{d1}: wrote {n} ESPN snapshots; final games {r['games']}, "
               f"linked {r['linked']}, with lines {r['lined'] or 0}")
