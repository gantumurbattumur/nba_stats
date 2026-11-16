"""Predicted vs closing line vs actual, with explicit prediction provenance.

The main workflow uses the window model (margin, total and win probability together).
Retro predictions are computed in memory, once per date, using only eligible games before
that date. They never replace recorded predictions. Prior-season in-sample estimates are
available only for the separately labelled honesty table.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from nba_predict.analysis.enrich import closing_lines, load_enriched
from nba_predict.analysis.filters import FilterSpec, apply_filter
from nba_predict.analysis.trends import lean, team_cover_trends
from nba_predict.config import get_settings, model_scope
from nba_predict.features.build import load_features
from nba_predict.models import window_rating as wr
from nba_predict.models.ml import eligible
from nba_predict.teams import by_nba_id

MODEL = "window"
PRED_COLS = ["game_id", "home_win_prob", "pred_margin", "pred_total", "prediction_origin",
             "retro", "made_before_tip", "created_at", "train_games", "train_max_date"]


def _timestamp(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def workflow_predictions(conn, games: pd.DataFrame, *, features=None, prediction_mode="pregame",
                         now=None) -> pd.DataFrame:
    """Recorded policy-compatible pre-tip rows, else pre-date window reconstructions/previews.

    Unknown timing, legacy season scope, and late records cannot become a pre-tip observation.
    ``in_sample`` is explicit and is never used by the Results Review table.
    """
    if prediction_mode not in ("pregame", "in_sample"):
        raise ValueError("prediction_mode must be pregame or in_sample")
    games = model_scope(games)
    if games.empty:
        return pd.DataFrame(columns=PRED_COLS)
    now = _timestamp(now or dt.datetime.now(dt.UTC))
    f = load_features(conn, get_settings().windows) if features is None else model_scope(features)
    trainable = f[(f.status == "final") & f.margin.notna()]
    trainable = trainable[eligible(trainable)]
    saved = pd.read_sql_query("select * from predictions where model=?", conn, params=(MODEL,))
    saved = saved.set_index("game_id")
    out, pending = [], []
    for game in games.to_dict("records"):
        p = saved.loc[game["game_id"]].to_dict() if game["game_id"] in saved.index else None
        valid = False
        if p and prediction_mode == "pregame":
            try:
                info = json.loads(p.get("params_json") or "{}")
                valid = (info.get("workflow_version") == 1
                         and info.get("model_seasons") == list(get_settings().model_seasons)
                         and bool(info.get("train_seasons"))
                         and set(info["train_seasons"]) <= set(get_settings().model_seasons)
                         and isinstance(info.get("train_games"), int)
                         and info["train_games"] >= 10
                         and bool(info.get("train_max_date"))
                         and info["train_max_date"] < game["game_date"]
                         and p.get("made_after_tip") == 0
                         and all(np.isfinite(p.get(k)) for k in ("home_win_prob", "pred_margin", "pred_total"))
                         and 0 <= p["home_win_prob"] <= 1
                         and _timestamp(p.get("created_at")) < _timestamp(game.get("start_time_utc")))
            except (ValueError, TypeError, AttributeError):
                valid = False
        if valid:
            out.append(dict(game_id=game["game_id"], **{k: p[k] for k in ("home_win_prob", "pred_margin", "pred_total", "created_at")},
                            prediction_origin="recorded pre-tip", retro=False, made_before_tip=True,
                            train_games=info["train_games"], train_max_date=info["train_max_date"]))
        else:
            pending.append(game)
    if not pending:
        return pd.DataFrame(out, columns=PRED_COLS)
    todo = pd.DataFrame(pending)
    fit_cache = {}
    for day, group in todo.groupby("game_date", sort=True):
        if prediction_mode == "in_sample":
            season = group.season.iloc[0]
            train = trainable[trainable.season == season]
            cache_key = season
        else:
            train = trainable[trainable.game_date < day]
            cache_key = day
        target = f[f.game_id.isin(group.game_id)]
        pred = pd.DataFrame(columns=["game_id", "home_win_prob", "pred_margin", "pred_total"])
        if len(train) >= 10 and not target.empty:
            if cache_key not in fit_cache:
                fit_cache[cache_key] = wr.fit(train)
            pred = wr.predict_rows(target, fit_cache[cache_key])
        pred = pred.set_index("game_id")
        for game in group.to_dict("records"):
            gid = game["game_id"]
            before = game["status"] == "scheduled" and now < _timestamp(game.get("start_time_utc"))
            origin = "in-sample" if prediction_mode == "in_sample" else "preview" if before else "retro"
            p = pred.loc[gid].to_dict() if gid in pred.index else {}
            out.append(dict(game_id=gid, **{k: p.get(k, np.nan) for k in ("home_win_prob", "pred_margin", "pred_total")},
                            prediction_origin=origin if p else "unavailable", retro=origin == "retro",
                            made_before_tip=bool(before) if prediction_mode == "pregame" else False,
                            created_at=now.isoformat(), train_games=len(train),
                            train_max_date=train.game_date.max() if len(train) else None))
    return pd.DataFrame(out, columns=PRED_COLS)


def market_lines(conn, game_row) -> dict:
    """Current line for scheduled games; stored closing line for finals.

    Display lines even without prices; do not invent prices to enable a comparison.
    Opening/current comparisons use the same source and provider.
    """
    gid = game_row["game_id"]
    if game_row["status"] == "final":
        close = closing_lines(conn)
        close = close[close.game_id == gid]
        if close.empty:
            return {}
        row = close.iloc[0].to_dict()
        return {**row, "source": row["odds_source"], "line_kind": "closing"}
    snaps = pd.read_sql_query("select * from odds_snapshots where game_id=? order by pulled_at,id",
                             conn, params=(gid,))
    for source in ("oddsapi", "espn"):
        rows = snaps[snaps.source == source]
        if rows.empty:
            continue
        latest = rows[rows.pulled_at == rows.pulled_at.max()]
        # One named provider keeps opening/current and the ML pair coherent.
        chosen = latest.sort_values("book", key=lambda s: s.str.lower().ne("draftkings")).iloc[0]
        same = rows[rows.book == chosen.book]
        result = chosen.to_dict()
        result.update(line_kind="current", first_observed=same.iloc[0].to_dict() if len(same) > 1 else None)
        return result
    return {}


def _hit(direction, difference):
    if direction is None or pd.isna(difference) or difference == 0:
        return np.nan
    return float((difference > 0) == (direction in ("home", "over")))


def build_results(conn, spec: FilterSpec | None = None, *, prediction_mode="pregame") -> pd.DataFrame:
    """One row per finished game, FilterSpec matches either team's enriched row.

    Unavailable predictions/lines retain the game with missing outcomes. No pushes are wins
    or losses. spread_hit/total_hit grade labelled tendencies; model_*_hit grade the model
    direction even below a label threshold.
    """
    spec = spec or FilterSpec(seasons=[get_settings().current_season])
    history = model_scope(load_enriched(conn))
    ids = apply_filter(history, spec).game_id.unique()
    games = model_scope(pd.read_sql_query("select * from games where status='final'", conn))
    games = games[games.game_id.isin(ids)].sort_values(["game_date", "game_id"])
    if games.empty:
        return pd.DataFrame()
    preds = workflow_predictions(conn, games, prediction_mode=prediction_mode)
    lines = closing_lines(conn).rename(columns={"total": "total_line"})
    combined = games.merge(preds, on="game_id", how="left").merge(lines, on="game_id", how="left")
    trend_cache, rows = {}, []
    for row in combined.to_dict("records"):
        home, away = by_nba_id(row["home_team_id"]), by_nba_id(row["away_team_id"])
        day = row["game_date"]
        for team in (home, away):
            key = (team.nba_id, day)
            if key not in trend_cache:
                trend_cache[key] = team_cover_trends(history, team.nba_id, day, windows=(10,),
                                                     splits=("all", "home", "away"))
        trends = pd.concat([trend_cache[(t.nba_id, day)] for t in (home, away)], ignore_index=True)
        line = {"home_spread": row["home_spread"], "total": row["total_line"],
                "home_ml": row["home_ml"], "away_ml": row["away_ml"]}
        evidence = lean(row, trends, row, line)
        margin = row["home_pts"] - row["away_pts"]
        total = row["home_pts"] + row["away_pts"]
        cover_diff = margin + row["home_spread"]
        total_diff = total - row["total_line"]
        row.update(home_team=home.abbr, away_team=away.abbr, matchup=f"{away.abbr} @ {home.abbr}",
                   final_score=f"{away.abbr} {row['away_pts']} – {home.abbr} {row['home_pts']}",
                   actual_margin=margin, actual_total=total,
                   covered=None if pd.isna(cover_diff) else "Push" if cover_diff == 0 else home.abbr if cover_diff > 0 else away.abbr,
                   over_under=None if pd.isna(total_diff) else "Push" if total_diff == 0 else "Over" if total_diff > 0 else "Under")
        for market, diff in (("spread", cover_diff), ("total", total_diff), ("moneyline", margin)):
            ev = evidence[market]
            row[f"{market}_label"] = ev["label"]
            row[f"{market}_direction"] = ev["direction"]
            row[f"model_{market}_hit"] = _hit(ev["direction"], diff)
            row[f"{market}_hit"] = row[f"model_{market}_hit"] if ev["label"] != "None" else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def result_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Labelled tendency hit rates, always separated by recorded/retro/in-sample origin."""
    rows = []
    if not results.empty:
        for origin, group in results.groupby("prediction_origin"):
            for market in ("spread", "total", "moneyline"):
                hits = group[f"{market}_hit"].dropna()
                rows.append(dict(origin=origin, market=market, n=len(hits),
                                 hit_rate=float(hits.mean()) if len(hits) else np.nan,
                                 small_sample=len(hits) < 30))
    return pd.DataFrame(rows)


def team_last_n(enriched: pd.DataFrame, team: str, as_of, n=5) -> dict:
    """Last N actual games (not last N lined games), retaining missing-line/push counts."""
    row = team_cover_trends(enriched, team, as_of, windows=(n,), splits=("all",)).iloc[0]
    return dict(team=team, games=int(row.games), covers=int(row.covers), decided=int(row.ats_n),
                pushes=int(row.ats_pushes), no_line=int(row.games - row.ats_n - row.ats_pushes))
