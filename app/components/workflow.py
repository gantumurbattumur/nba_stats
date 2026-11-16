"""Shared presentation and cached, in-memory analysis for the game workflow."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from components.data import _mtime, db_path, enriched, features, rw_conn
from nba_predict.analysis.filters import FilterSpec
from nba_predict.analysis.results import build_results, workflow_predictions
from nba_predict.analysis.trends import lean_history
from nba_predict.config import get_settings, model_scope
from nba_predict.db import connect

ORIGINS = {"retro": "Retro · reconstructed", "recorded pre-tip": "Recorded before tip",
           "preview": "Preview · not saved", "in-sample": "In-sample · fitted on this season",
           "unavailable": "Unavailable"}
MARKETS = {"spread": "Spread", "total": "Total", "moneyline": "Moneyline"}
SPLITS = {"all": "All games", "home": "At home", "away": "On the road",
          "fav": "As favourite", "dog": "As underdog"}
WINDOWS = {"last5": "Last 5", "last10": "Last 10", "last15": "Last 15", "season": "This season"}


def number(value, fmt=".1f", missing="—"):
    return missing if value is None or pd.isna(value) else format(float(value), fmt)


def percent(value):
    return number(value, ".1%")


def history():
    return model_scope(enriched())


def policy_key():
    s = get_settings()
    return (s.model_seasons, s.train_seasons, s.windows)


@st.cache_data(show_spinner=False, max_entries=12)
def _results(path, stamp, season, mode, policy):
    c = connect(path)
    try:
        return build_results(c, FilterSpec(seasons=[season]), prediction_mode=mode)
    finally:
        c.close()


def season_results(season, mode="pregame"):
    # Refresh derived data before taking the cache key; reconstruction saves no predictions.
    features()
    return _results(db_path(), _mtime(db_path()), season, mode, policy_key())


@st.cache_data(show_spinner=False, max_entries=32)
def _predictions(path, stamp, games, policy):
    c = connect(path)
    try:
        return workflow_predictions(c, games)
    finally:
        c.close()


def slate_predictions(games):
    features()
    out = _predictions(db_path(), _mtime(db_path()), games, policy_key())
    # Cached forecasts remain useful after tip, but the timing claim must not go stale.
    now = pd.Timestamp(dt.datetime.now(dt.UTC))
    tips = pd.to_datetime(games.set_index("game_id").start_time_utc, utc=True, errors="coerce")
    before = out.game_id.map(tips).gt(now)
    expired = out.prediction_origin.eq("preview") & ~before
    out.loc[expired, ["prediction_origin", "retro", "made_before_tip"]] = ["retro", True, False]
    return out


def performance_table(frame):
    """Readable rates with denominators and provenance on every row."""
    if frame.empty:
        st.info("No finished games to summarize yet.")
        return
    view = pd.DataFrame({
        "Prediction": frame.origin.map(ORIGINS), "Market": frame.market.map(MARKETS),
        **({"Tendency": frame.label} if "label" in frame else {}),
        "Hit rate": frame.hit_rate.map(percent), "Decided games": frame.n,
        "Sample": frame.n.map(lambda n: "No decided games" if n == 0 else "Small sample" if n < 30 else "30+ games"),
    })
    st.dataframe(view, hide_index=True, width="stretch")
    if frame.n.lt(30).any():
        st.warning("Small sample: rates with fewer than 30 decided games can change sharply. "
                   "Pushes, missing lines and unavailable predictions do not count in the hit rate.")


def honesty_table(season, results=None):
    c = rw_conn()
    try:
        data = lean_history(c, season, results=results)
    finally:
        c.close()
    performance_table(data)


def recent_table(rows):
    view = pd.DataFrame({
        "Date": rows.game_date, "Team": rows.team_abbr, "Opponent": rows.opp_abbr,
        "Venue": rows.is_home.map({1: "Home", 0: "Away"}), "Result": rows.wl,
        "Points": rows.pts, "Opponent points": rows.opp_pts,
        "Team spread": rows.team_spread.map(lambda x: number(x, "+.1f", "No line")),
        "Margin": rows.margin.map(lambda x: number(x, "+.0f")),
        "Covered?": rows.ats.map({"cover": "Yes", "no_cover": "No", "push": "Push"}).fillna("No line"),
        "Total line": rows.total_line.map(lambda x: number(x, missing="No line")),
        "Total points": rows.total_pts,
        "Over / under": rows.ou.map({"over": "Over", "under": "Under", "push": "Push"}).fillna("No line"),
    })
    st.dataframe(view, hide_index=True, width="stretch")


def trend_table(trends, market):
    rows = []
    for r in trends.itertuples():
        spread = market == "spread"
        n = r.ats_n if spread else r.ou_n
        rows.append({"Team": r.team, "Window": WINDOWS[r.window], "Games": r.games,
                     "Record": f"{r.covers}–{r.no_covers}–{r.ats_pushes}" if spread else f"{r.overs}–{r.unders}–{r.ou_pushes}",
                     **({"Cover rate": percent(r.ats_pct)} if spread else
                        {"Over rate": percent(r.over_pct), "Under rate": percent(r.under_pct)}),
                     "Decided games": n, "Missing lines": r.games - n - (r.ats_pushes if spread else r.ou_pushes)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def evidence_table(evidence, home, away):
    direction = {"home": home, "away": away, "over": "Over", "under": "Under"}.get(evidence["direction"])
    label = evidence["label"]
    st.markdown(f"**Tendency: {label}" + (f" · {direction}**" if label != "None" and direction else "**"))
    st.dataframe(pd.DataFrame([{"Check": f["status"], "Evidence": f["factor"], "Detail": f["detail"]}
                               for f in evidence["factors"]]), hide_index=True, width="stretch")
    if any(f["n"] is not None and f["n"] < 5 for f in evidence["factors"]):
        st.warning("Too few decided games for a trend check: each team needs at least 5.")
