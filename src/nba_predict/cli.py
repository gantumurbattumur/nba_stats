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
    season_type: Annotated[
        list[str] | None,
        typer.Option("--season-type", help="regular | playin | playoffs; repeatable (default: all)"),
    ] = None,
) -> None:
    """Ingest team game logs for one or more seasons (default: current season, all season types)."""
    s = get_settings()
    seasons = season or [s.current_season]
    label_to_api = {v: k for k, v in ingest_mod.SEASON_TYPES.items()}
    bad = [t for t in (season_type or []) if t not in label_to_api]
    if bad:
        typer.echo(f"error: unknown --season-type {bad}; use regular, playin, playoffs", err=True)
        raise typer.Exit(2)
    api_types = tuple(label_to_api[t] for t in season_type) if season_type else tuple(ingest_mod.SEASON_TYPES)
    conn = _conn()
    client = NbaClient(delay_s=s.nba_delay_s)
    for sz in seasons:
        try:
            n = ingest_mod.ingest_season(conn, client, sz, season_types=api_types)
        except IngestError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        by = conn.execute(
            "select season_type, count(*) from games where season=? and status='final' group by 1 order by 1",
            (sz,),
        ).fetchall()
        total = conn.execute("select count(*) from team_games where season=?", (sz,)).fetchone()[0]
        typer.echo(f"{sz}: wrote {n} team_games rows; {total} team_games in DB; final games "
                   + ", ".join(f"{r[0]}={r[1]}" for r in by))


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
    with_open: Annotated[bool, typer.Option("--with-open", help="also store ESPN core-API opening lines")] = False,
) -> None:
    """Backfill ESPN past-game closing lines (resumable; ~1 request/s)."""
    from nba_predict.data.espn_client import EspnClient

    s = get_settings()
    d0 = _parse_date(date_from, today_et() - dt.timedelta(days=1))
    d1 = _parse_date(date_to, d0)
    conn = _conn()
    try:
        n = ingest_mod.ingest_espn_lines(conn, EspnClient(delay_s=s.espn_delay_s), d0, d1,
                                         only_missing=not refetch, with_open=with_open)
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


@app.command()
def status() -> None:
    """Per-season coverage: games, team_games, ESPN links/lines, ML-vs-spread favorite mismatches."""
    conn = _conn()
    typer.echo(f"db: {get_settings().db_path}")
    df = ingest_mod.coverage(conn)
    typer.echo(df.to_string(index=False) if not df.empty else "(no games yet: run `nba ingest`)")
    if df.empty:
        typer.echo("columns: final_games scheduled team_games espn_linked espn_lines fav_disagree")


@app.command()
def explore(
    preset: Annotated[str | None, typer.Option(help="saved FilterSpec preset name")] = None,
    team: Annotated[list[str] | None, typer.Option("--team", help="abbr; repeatable")] = None,
    venue: Annotated[str | None, typer.Option(help="all | home | away")] = None,
    last_n: Annotated[int | None, typer.Option("--last-n", help="most recent N per team")] = None,
    season: Annotated[list[str] | None, typer.Option("--season", help="repeatable")] = None,
    metrics: Annotated[str, typer.Option(help="comma-separated metric keys")] = "record,net,ortg,drtg,pace",
    group_by: Annotated[str | None, typer.Option("--group-by", help="comma-separated columns")] = None,
    csv: Annotated[str | None, typer.Option(help="write the table to this CSV path")] = None,
) -> None:
    """Filter team_games_enriched and print aggregate metrics."""
    from nba_predict.analysis.enrich import load_enriched
    from nba_predict.analysis.filters import FilterSpec, apply_filter, load_preset
    from nba_predict.analysis.metrics import aggregate

    conn = _conn()
    spec = load_preset(conn, preset) if preset else FilterSpec()
    overrides = {k: v for k, v in dict(teams=team, venue=venue, last_n=last_n, seasons=season).items() if v}
    spec = spec.model_copy(update=overrides)
    spec = FilterSpec.model_validate(spec.model_dump())  # re-validate overrides
    df = load_enriched(conn)  # rebuilt when team_games / closing lines changed since last build
    out = aggregate(apply_filter(df, spec), [m.strip() for m in metrics.split(",") if m.strip()],
                    group_by=[g.strip() for g in group_by.split(",")] if group_by else None)
    typer.echo(spec.describe())
    typer.echo(out.to_string(index=False))
    if csv:
        out.to_csv(csv, index=False)


@app.command()
def features(
    windows: Annotated[str | None, typer.Option(help="comma-separated window sizes, default settings.windows")] = None,
) -> None:
    """Rebuild team_games_enriched and game_features (always, regardless of fingerprints)."""
    import time

    from nba_predict.analysis.enrich import build_enriched
    from nba_predict.features.build import build_feature_table

    s = get_settings()
    wins = tuple(int(w) for w in windows.split(",")) if windows else s.windows
    conn = _conn()
    t = time.monotonic()
    e = build_enriched(conn)
    f = build_feature_table(conn, wins)
    typer.echo(f"team_games_enriched: {len(e)} rows; game_features: {len(f)} rows x {f.shape[1]} cols "
               f"(windows {','.join(map(str, wins))}) in {time.monotonic() - t:.1f}s")


@app.command("backtest")
def backtest_cmd(
    season: Annotated[list[str] | None, typer.Option("--season", help="test season; repeatable")] = None,
    holdout: Annotated[list[str] | None, typer.Option("--holdout", help="reported-only season; repeatable")] = None,
    n_boot: Annotated[int, typer.Option("--n-boot", help="bootstrap resamples for CIs")] = 1000,
    all_seasons: Annotated[bool, typer.Option("--all-seasons", help="Use all history with monthly refits")] = False,
) -> None:
    """Walk-forward backtest of all models; prints metric × model per season (evaluation only)."""
    import pandas as pd

    from nba_predict import backtest

    conn = _conn()
    settings = get_settings()
    tests = tuple(season) if season else (("2024-25", "2025-26") if all_seasons else (settings.current_season,))
    typer.echo("Seasons used: " + ("all available; monthly refits" if all_seasons else
               f"models {', '.join(settings.model_seasons)}; fit {', '.join(settings.train_seasons)} (frozen)")
               + f"; test {', '.join(tests) or 'none'}; holdout {', '.join(holdout or ()) or 'none'}")
    res = backtest.run(conn, test_seasons=tests,
                       holdout=tuple(holdout or ()), n_boot=n_boot, all_seasons=all_seasons)
    if res.empty:
        typer.echo("no results (no final games in the requested seasons?)")
        raise typer.Exit(1)
    pd.set_option("display.width", 250)
    order = list(dict.fromkeys(res.metric))
    for (role, sz), g in res[res.segment == "all"].groupby(["role", "season"], sort=False):
        typer.echo(f"\n== {sz} ({role}) ==")
        t = g.pivot_table(index="metric", columns="model", values="value", aggfunc="first")
        t = t.reindex([m for m in order if m in t.index])
        typer.echo(t.to_string(float_format=lambda v: f"{v:.4f}"))
    seg = res[(res.segment != "all") & res.metric.isin(["n", "log_loss", "margin_mae", "mae_gap"])]
    if not seg.empty:
        typer.echo("\n== by segment (early = either team < 15 season games) ==")
        t = seg.pivot_table(index=["role", "season", "segment", "metric"], columns="model", values="value",
                            aggfunc="first")
        typer.echo(t.to_string(float_format=lambda v: f"{v:.4f}"))
    typer.echo(f"\nbest model (lowest mean test log_loss): {backtest.best_model(conn)}")


@app.command()
def odds(closing: Annotated[bool, typer.Option(help="mark this pull as the closing line (~30 min before tip)")] = False) -> None:
    """Pull current odds: Odds API when configured, otherwise the free ESPN scoreboard."""
    from nba_predict.data.odds_client import OddsApiClient, OddsApiError

    conn = _conn()
    if not get_settings().odds_api_key:
        try:
            n = ingest_mod.ingest_espn_scoreboard_odds(conn, today_et())
        except IngestError as e:
            typer.echo(f"odds: {e}", err=True)
            raise typer.Exit(1)
        typer.echo(f"odds: {n} ESPN pregame snapshots (no ODDS_API_KEY)")
        if closing:
            typer.echo("ESPN scoreboard snapshots remain current observations; closing lines come from finished-game ingest.")
        return
    try:
        n = ingest_mod.ingest_odds(conn, OddsApiClient(get_settings().odds_api_key), closing=closing)
    except OddsApiError as e:
        typer.echo(f"odds: {e}", err=True)
        raise typer.Exit(1)
    u = conn.execute("select at, credits_used, credits_remaining from api_usage order by id desc limit 1").fetchone()
    games = conn.execute("select count(distinct game_id) from odds_snapshots where source='oddsapi' and pulled_at=?",
                         (u["at"],)).fetchone()[0]
    typer.echo(f"odds: {n} snapshots for {games} games; credits used {u['credits_used']}, "
               f"remaining {u['credits_remaining']}")


def _today_or(date: str | None) -> dt.date:
    return _parse_date(date, today_et())


@app.command()
def predict(date: Annotated[str | None, typer.Option(help="YYYY-MM-DD (ET), default today")] = None,
            all_seasons: Annotated[bool, typer.Option("--all-seasons", help="Use all history for analysis")] = False) -> None:
    """Fit elo/window/ml on games before the date and write predictions for that date's games."""
    from nba_predict.predict import predict_date

    d = _today_or(date)
    out = predict_date(_conn(), d, all_seasons=all_seasons)
    typer.echo(f"{d}: {out.game_id.nunique()} games x {out.model.nunique()} models predicted")
    if not out.empty:
        typer.echo(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@app.command("flags")
def flags_cmd(date: Annotated[str | None, typer.Option(help="YYYY-MM-DD (ET), default today")] = None,
              model: Annotated[str | None, typer.Option(help="default: best backtest model")] = None) -> None:
    """Record model-vs-market disagreements (edge >= edge_threshold) for the date's games."""
    from nba_predict.flags import flag_values

    d = _today_or(date)
    out = flag_values(_conn(), d, model)
    typer.echo(f"{d}: {len(out)} value flags (analysis only)")
    if not out.empty:
        typer.echo(out.drop(columns=["created_at"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@app.command("grade")
def grade_cmd(date: Annotated[str | None, typer.Option(help="only this date; default all unsettled")] = None) -> None:
    """Settle flags of finished games (result, 1-unit return for evaluation, closing line value)."""
    from nba_predict.flags import grade

    n = grade(_conn(), dt.date.fromisoformat(date) if date else None)
    typer.echo(f"graded {n} flags")


def _daily_table(conn, d: dt.date, model: str) -> str:
    import pandas as pd

    from nba_predict.flags import market_view
    from nba_predict.teams import by_nba_id

    rows = conn.execute(
        "select g.game_id, g.home_team_id, g.away_team_id, p.home_win_prob, p.pred_margin, p.pred_total, "
        "p.params_json from games g left join predictions p on p.game_id = g.game_id and p.model = ? "
        "where g.game_date = ? and g.status != 'postponed' order by g.game_id", (model, d.isoformat())).fetchall()
    feats = {r["game_id"]: r for r in conn.execute(
        "select game_id, h_b2b, a_b2b, h_rest, a_rest, h_gp_szn, a_gp_szn from game_features where game_date = ?",
        (d.isoformat(),))} if conn.execute(
        "select 1 from sqlite_master where name = 'game_features'").fetchone() else {}
    nflags = dict(conn.execute("select game_id, group_concat(market || ':' || side) from value_flags "
                               "where result is null group by game_id").fetchall())
    out = []
    for r in rows:
        h, a = by_nba_id(r["home_team_id"]), by_nba_id(r["away_team_id"])
        mv = market_view(conn, r["game_id"]) or {}
        mk = mv.get("ml_novig")
        p = r["home_win_prob"]
        notes = []
        f = feats.get(r["game_id"])
        if f is not None:
            if f["h_b2b"]:
                notes.append(f"{h.abbr} B2B")
            if f["a_b2b"]:
                notes.append(f"{a.abbr} B2B")
            if f["h_rest"] is not None and f["a_rest"] is not None and abs(f["h_rest"] - f["a_rest"]) >= 2:
                notes.append(f"rest {f['h_rest']}-{f['a_rest']}")
            if min(f["h_gp_szn"] or 0, f["a_gp_szn"] or 0) < 5:
                notes.append("low gp")
        if not mv:
            notes.append("no odds")
        if r["game_id"] in nflags:
            notes.append("FLAG " + nflags[r["game_id"]])
        out.append({
            "game": f"{a.abbr} @ {h.abbr}",
            "model win% (home)": f"{100 * p:.1f}" if p is not None else "-",
            "market win%": f"{100 * mk[0]:.1f}" if mk else "-",
            "edge": f"{100 * (p - mk[0]):+.1f}" if mk and p is not None else "-",
            "proj spread / line": (f"{-r['pred_margin']:+.1f} / "
                                   + (f"{mv['home_spread']:+.1f}" if mv.get("home_spread") is not None else "-"))
            if r["pred_margin"] is not None else "-",
            "proj total / line": (f"{r['pred_total']:.1f} / " + (f"{mv['total']:.1f}" if mv.get("total") else "-"))
            if r["pred_total"] is not None else "-",
            "notes": ", ".join(notes),
        })
    return pd.DataFrame(out).to_string(index=False) if out else "(no games on this date)"


@app.command()
def daily(date: Annotated[str | None, typer.Option(help="YYYY-MM-DD (ET), default today")] = None) -> None:
    """ingest → slate → espn (yesterday) → odds → features → predict → flags → table.
    Any failing data step prints a message and the rest still runs. Analysis only."""
    from nba_predict.analysis.enrich import build_enriched
    from nba_predict.backtest import best_model
    from nba_predict.data.espn_client import EspnClient
    from nba_predict.data.odds_client import OddsApiClient, OddsApiError
    from nba_predict.features.build import build_feature_table
    from nba_predict.flags import flag_values
    from nba_predict.predict import predict_date

    s = get_settings()
    d = _today_or(date)
    conn = _conn()
    nba = NbaClient(delay_s=s.nba_delay_s)
    steps = [
        ("ingest", lambda: f"{ingest_mod.ingest_season(conn, nba, s.current_season)} team_games rows"),
        ("slate", lambda: f"{ingest_mod.ingest_slate(conn, nba, d)} games"),
        ("espn", lambda: f"{ingest_mod.ingest_espn_lines(conn, EspnClient(delay_s=s.espn_delay_s), d - dt.timedelta(days=1), d - dt.timedelta(days=1))} lines"),
    ]
    for name, fn in steps:
        try:
            typer.echo(f"[{name}] {fn()}")
        except IngestError as e:
            typer.echo(f"[{name}] skipped: {e}")
    have_odds = True
    try:
        if s.odds_api_key:
            n = ingest_mod.ingest_odds(conn, OddsApiClient(s.odds_api_key))
            typer.echo(f"[odds] {n} Odds API snapshots")
        else:
            n = ingest_mod.ingest_espn_scoreboard_odds(conn, d)
            typer.echo(f"[odds] {n} ESPN pregame snapshots (no ODDS_API_KEY)")
        have_odds = n > 0
    except (OddsApiError, IngestError) as e:
        have_odds = False
        typer.echo(f"[odds] skipped: {e}")
    build_enriched(conn)
    f = build_feature_table(conn, s.windows)
    typer.echo(f"[features] {len(f)} games")
    pred = predict_date(conn, d)
    typer.echo(f"[predict] {pred.game_id.nunique() if not pred.empty else 0} games")
    model = best_model(conn)
    if have_odds:
        typer.echo(f"[flags] {len(flag_values(conn, d, model))} value flags ({model})")
    else:
        typer.echo("[flags] flags skipped: no live odds")
    typer.echo(f"\n{d} — model: {model} (injuries not included; analysis only)")
    typer.echo(_daily_table(conn, d, model))
