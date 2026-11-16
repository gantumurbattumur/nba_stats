"""Predicted versus closing line versus actual, with explicit forecast provenance."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components.data import header  # noqa: E402
from components.workflow import (ORIGINS, history, honesty_table, number, percent,  # noqa: E402
                                 performance_table, season_results)
from nba_predict.analysis.filters import FilterSpec, apply_filter  # noqa: E402
from nba_predict.analysis.results import result_summary, team_last_n  # noqa: E402
from nba_predict.config import get_settings  # noqa: E402

st.set_page_config(page_title="Results Review", layout="wide")
header("Results Review", "What the model projected, where the line closed, and how the game ended.")
hist = history()
s = get_settings()
if hist.empty:
    st.info("No finished team history yet. Run `uv run nba daily` to refresh the local data.")
    st.stop()
seasons = [season for season in s.model_seasons if season in set(hist.season)]
if not seasons:
    st.info("No finished games in the configured model seasons.")
    st.stop()
season = st.selectbox("Season", seasons, index=seasons.index(s.current_season) if s.current_season in seasons else len(seasons)-1,
                      key="rr_season")
season_hist = hist[hist.season == season]
dates = st.date_input("Finished game dates", (pd.Timestamp(season_hist.game_date.min()).date(),
                                            pd.Timestamp(season_hist.game_date.max()).date()), key=f"rr_dates_{season}")
teams = st.multiselect("Teams", sorted(season_hist.team_abbr.unique()), key="rr_teams",
                       help="Include a game when either team matches. Each game appears once.")
if len(dates) != 2:
    st.info("Choose both the start and end date.")
    st.stop()
spec = FilterSpec(seasons=[season], date_from=dates[0], date_to=dates[1], teams=teams or None)
ids = apply_filter(hist, spec).game_id.unique()
with st.spinner("Reconstructing forecasts using only earlier games…"):
    all_results = season_results(season)
results = all_results[all_results.game_id.isin(ids)].copy() if not all_results.empty else all_results
if results.empty:
    st.info("No finished games match these filters.")
    st.stop()
origins = sorted(results.prediction_origin.unique())
chosen_origins = st.multiselect("Prediction types", origins, default=origins, format_func=ORIGINS.get, key="rr_origins")
results = results[results.prediction_origin.isin(chosen_origins)]
if results.empty:
    st.info("Select a prediction type to review.")
    st.stop()
st.caption("Retro = reconstructed now from games strictly before the game date. Recorded before tip = a saved forecast "
           "with verified timing and season scope. They are never pooled in hit-rate summaries. Injuries are not included.")
st.subheader("Tendency outcomes")
performance_table(result_summary(results))
st.caption("Hit rates grade only Strong, Lean and Slight tendencies. None, pushes and missing outcomes are excluded.")

st.subheader("A team's recent cover record")
c1, c2 = st.columns(2)
team_options = sorted(season_hist.team_abbr.unique())
default_team = teams[0] if teams else team_options[0]
team = c1.selectbox("Team to review", team_options, index=team_options.index(default_team), key="rr_team")
n = c2.selectbox("Recent games", [5, 10, 15], key="rr_n")
record = team_last_n(hist, team, dates[1] + dt.timedelta(days=1), n=n)
st.write(f"**{team} covered {record['covers']} of its last {record['games']} games** through {dates[1]}.")
st.caption(f"{record['decided']} decided · {record['pushes']} pushes · {record['no_line']} missing lines. "
           "Uses actual games through the end date, regardless of the range start or prediction-type filter; may reach into last season.")
st.warning("Small sample: a last-5, last-10 or last-15 record is descriptive, not reliable evidence of future accuracy.")

st.subheader("Predicted vs line vs final")
results = results.sort_values(["game_date", "game_id"], ascending=False)


def outcome(value):
    return "—" if pd.isna(value) else "Yes" if value == 1 else "No"


view = pd.DataFrame({
    "Date": results.game_date, "Game": results.matchup,
    "Prediction": results.prediction_origin.map(ORIGINS),
    "Home win probability": results.home_win_prob.map(percent),
    "Projected home margin": results.pred_margin.map(lambda x: number(x, "+.1f")),
    "Projected total": results.pred_total.map(number),
    "Closing home spread": results.home_spread.map(lambda x: number(x, "+.1f", "No line")),
    "Closing total": results.total_line.map(lambda x: number(x, missing="No line")),
    "Closing home moneyline": results.home_ml.map(lambda x: number(x, "+.0f", "No price")),
    "Closing away moneyline": results.away_ml.map(lambda x: number(x, "+.0f", "No price")),
    "Line source": results.odds_source.map({"espn": "ESPN", "oddsapi": "The Odds API"}).fillna("No line"),
    "Final score": results.final_score, "Final home margin": results.actual_margin,
    "Final total": results.actual_total, "Covered?": results.covered.fillna("No line"),
    "Over / under": results.over_under.fillna("No line"),
    "Spread tendency": results.spread_label, "Total tendency": results.total_label,
    "Model margin side covered?": results.model_spread_hit.map(outcome),
    "Model total side hit?": results.model_total_hit.map(outcome),
    "Latest training date": results.train_max_date,
})
st.caption(f"{len(view)} games · Home margin is positive for a home win; a negative home spread means home favoured. "
           "Model-side columns grade the model direction even below a tendency threshold; — means ungraded, including pushes.")
st.dataframe(view, hide_index=True, width="stretch")
st.download_button("Download results CSV", view.to_csv(index=False), "results-review.csv", "text/csv")
with st.expander("Hit rate by tendency label"):
    st.caption("These are observed hit rates, not calibrated probabilities. Filters above apply.")
    honesty_table(season, results)
