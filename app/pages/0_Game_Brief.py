"""A game-first view of market lines, model estimates and the evidence behind tendencies."""

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components.data import header, load_table, rw_conn  # noqa: E402
from components.workflow import (ORIGINS, SPLITS, WINDOWS, evidence_table, history, honesty_table,  # noqa: E402
                                 number, percent, recent_table, season_results, slate_predictions, trend_table)
from nba_predict.analysis.results import market_lines  # noqa: E402
from nba_predict.analysis.trends import RULE, lean, team_cover_trends  # noqa: E402
from nba_predict.config import get_settings, model_scope  # noqa: E402
from nba_predict.odds_math import american_to_prob, devig_two_way  # noqa: E402
from nba_predict.teams import by_nba_id  # noqa: E402

st.set_page_config(page_title="Game Brief", layout="wide")
header("Game Brief", "Compare the line, the model and each team's recent record. Make your own assessment.")
s = get_settings()
day = st.date_input("Game date (Eastern time)", dt.datetime.now(ZoneInfo("America/New_York")).date(), key="gb_date")
games = load_table("games")
if games.empty:
    st.info("No games yet. Run `uv run nba daily` to refresh the local data.")
    st.stop()
games = model_scope(games)
games = games[(games.game_date == day.isoformat()) & games.status.ne("postponed")].copy()
if games.empty:
    st.info("No games on this date in the configured model seasons: " + ", ".join(s.model_seasons) + ".")
    st.stop()
hist = history()
if hist.empty:
    st.info("Team history is not available yet. Run `uv run nba ingest` then `uv run nba features`.")
    st.stop()
with st.spinner("Preparing the game comparisons…"):
    preds = slate_predictions(games).set_index("game_id")
c = rw_conn()
try:
    lines = {g["game_id"]: market_lines(c, g) for g in games.to_dict("records")}
finally:
    c.close()


def tip(value):
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return "Tip time unknown" if pd.isna(stamp) else stamp.tz_convert("America/New_York").strftime("%I:%M %p ET")


def moneyline(line, home, away):
    return f"{away} {number(line.get('away_ml'), '+.0f', 'no price')} / {home} {number(line.get('home_ml'), '+.0f', 'no price')}"


slate, labels = [], {}
for g in games.sort_values(["start_time_utc", "game_id"]).to_dict("records"):
    gid = g["game_id"]
    h, a = by_nba_id(g["home_team_id"]).abbr, by_nba_id(g["away_team_id"]).abbr
    labels[gid] = f"{a} @ {h}"
    line, pred = lines[gid], preds.loc[gid]
    slate.append({"Game": labels[gid], "Tip": tip(g["start_time_utc"]), "Status": g["status"].title(),
                  "Home spread": number(line.get("home_spread"), "+.1f", "No line yet"),
                  "Total": number(line.get("total"), missing="No line yet"), "Moneyline": moneyline(line, h, a),
                  "Model home win": percent(pred.home_win_prob), "Prediction": ORIGINS[pred.prediction_origin]})
st.caption(f"{len(games)} games · Spread and model margin are shown from the home team's perspective.")
st.dataframe(pd.DataFrame(slate), hide_index=True, width="stretch")
gid = st.selectbox("Choose a game", list(labels), format_func=labels.get, key="gb_game")
g = games.set_index("game_id").loc[gid].to_dict() | {"game_id": gid}
h, a = by_nba_id(g["home_team_id"]), by_nba_id(g["away_team_id"])
line, pred = lines[gid], preds.loc[gid]
trends = pd.concat([team_cover_trends(hist, t.nba_id, day) for t in (h, a)], ignore_index=True)
evidence = lean(g, trends, pred, line)
prior = hist[hist.game_date < day.isoformat()].sort_values(["game_date", "game_id"], ascending=False)
st.header(f"{a.name} at {h.name}")
if g["status"] == "final":
    st.info(f"Final: {a.abbr} {g['away_pts']} – {h.abbr} {g['home_pts']}. Trends below use only earlier games.")

st.subheader("1. Market")
c1, c2, c3 = st.columns(3)
c1.metric(f"{h.abbr} spread", number(line.get("home_spread"), "+.1f", "No line yet"),
          help="Negative means the home team is favoured. A push means the final margin exactly matches the spread.")
c2.metric("Total points line", number(line.get("total"), missing="No line yet"))
c3.metric("Moneyline", moneyline(line, h.abbr, a.abbr), help="Prices for each team to win outright.")
if line:
    source = {"espn": "ESPN", "oddsapi": "The Odds API"}.get(line.get("source"), "Unknown source")
    st.caption(f"{source} · {line['line_kind'].title()} line · Provider: {line.get('book') or 'stored closing summary'}")
    if line.get("pulled_at"):
        st.caption(f"Observed: {line['pulled_at']} · Stored snapshot; use `uv run nba odds` to refresh.")
    first = line.get("first_observed")
    if first:
        st.write(f"First observed → current: {h.abbr} spread {number(first.get('home_spread'), '+.1f')} → "
                 f"{number(line.get('home_spread'), '+.1f')}; total {number(first.get('total'))} → {number(line.get('total'))}.")
        st.caption(f"First observed {first['pulled_at']}; same source and provider. This may not be the true opening line.")
    else:
        st.caption("No comparable opening/current pair is stored.")
else:
    st.info("No line yet for this game. Refresh with `uv run nba odds`; ESPN may not have published a line.")

st.subheader("2. Model")
c1, c2, c3 = st.columns(3)
c1.metric(f"{h.abbr} win probability", percent(pred.home_win_prob))
c2.metric(f"Projected {h.abbr} margin", number(pred.pred_margin, "+.1f"),
          help="Positive means the home team is projected to win by this many points.")
c3.metric("Projected total points", number(pred.pred_total))
st.caption(f"Window model · Seasons: {', '.join(s.model_seasons)} · Injuries and expected lineups are not included.")
timing = "Yes" if pred.made_before_tip else "No"
if pd.isna(pd.to_datetime(g["start_time_utc"], utc=True, errors="coerce")):
    timing = "Not verified · tip time unknown"
elif pred.prediction_origin == "unavailable":
    timing = "Not verified"
st.write(f"**{ORIGINS[pred.prediction_origin]}** · Made before tip: {timing}")
st.caption(f"Training games: {int(pred.train_games):,} · Latest training date: {pred.train_max_date or 'unavailable'} "
           f"· Prediction time: {pred.created_at}")
if pred.prediction_origin == "retro":
    st.info("Retro: reconstructed now using only data before the game date. It is not a saved pre-tip forecast.")
elif pred.prediction_origin == "preview":
    st.caption("Preview computed before the known tip time, but not saved. Use `uv run nba predict` to record forecasts.")
st.write(f"Model minus line: spread {number(evidence['spread']['disagreement'], '+.1f')} points; "
         f"total {number(evidence['total']['disagreement'], '+.1f')} points.")

with st.expander("How tendencies and samples work"):
    st.write(RULE)
    st.write("Tendencies describe statistical evidence, not calibrated confidence. A dash means evidence is missing. "
             "Cover rate excludes pushes and missing lines. Last-N windows may reach into last season; "
             "the season window uses only this season. All inputs are strictly before the selected date.")
split = st.selectbox("Trend split", list(SPLITS), format_func=SPLITS.get, key="gb_split",
                     help="Choose the subset first, then take its last 5, 10 or 15 games. Evidence always uses overall and upcoming-venue last 10.")
shown = trends[trends.split == split]
st.warning("Small samples: these short windows have fewer than 30 games. Read the decided-game counts beside each rate.")
st.subheader("3. Spread")
evidence_table(evidence["spread"], h.abbr, a.abbr)
trend_table(shown, "spread")
st.caption("Record = covered – did not cover – pushes. Each spread is from that team's perspective.")

st.subheader("4. Total")
evidence_table(evidence["total"], h.abbr, a.abbr)
trend_table(shown, "total")
st.caption("Record = overs – unders – pushes.")
pace = shown.pivot(index="window", columns="team", values="pace").reindex(WINDOWS)
pace_view = pd.DataFrame({"Window": [WINDOWS[w] for w in pace.index],
                          f"{h.abbr} pace": pace[h.abbr].map(number).to_list(),
                          f"{a.abbr} pace": pace[a.abbr].map(number).to_list(),
                          "Combined pace": pace[[h.abbr, a.abbr]].mean(axis=1, skipna=False).map(number).to_list()})
st.dataframe(pace_view, hide_index=True, width="stretch")
st.caption("Pace = estimated possessions per 48 minutes. Combined pace is the mean of both teams; game counts are in the table above.")

st.subheader("5. Moneyline")
evidence_table(evidence["moneyline"], h.abbr, a.abbr)
hm, am = line.get("home_ml"), line.get("away_ml")
if hm is not None and am is not None and pd.notna(hm) and pd.notna(am):
    hp, ap = devig_two_way(american_to_prob(hm), american_to_prob(am))
    st.write(f"Market win probability with the price margin removed: {h.abbr} **{hp:.1%}**, {a.abbr} **{ap:.1%}**. "
             f"Model: {h.abbr} **{percent(pred.home_win_prob)}**, {a.abbr} **{percent(1 - pred.home_win_prob)}**.")
    st.caption("The tendency follows the model's difference from the market probability; it need not identify the most likely winner.")
else:
    st.info("Both moneyline prices are needed to compare market and model win probabilities.")

st.subheader("6. Recent games")
for team in (h, a):
    st.markdown(f"**{team.name} · last 10**")
    recent_table(prior[prior.team_id == team.nba_id].head(10))

st.subheader("7. Head-to-head")
meetings = prior[(prior.team_id == h.nba_id) & (prior.opp_team_id == a.nba_id)].head(3)
if meetings.empty:
    st.info("No earlier meetings in this season or last season.")
else:
    recent_table(meetings)
    st.caption("Last 3 meetings at most, shown once each from the selected home team's perspective.")

with st.expander("Past performance by tendency · honesty check"):
    st.write(f"{s.current_season}: through the latest stored final. Recorded and retro predictions stay separate. "
             "None means no labelled tendency, so its hit-rate denominator is zero.")
    with st.spinner("Summarizing this season…"):
        honesty_table(s.current_season, season_results(s.current_season))
    if st.checkbox("Show last season's in-sample comparison", key="gb_in_sample"):
        for season in s.train_seasons:
            st.markdown(f"**{season} · In-sample: fitted on the same season being scored.**")
            st.caption("This includes later training outcomes and is not an independent estimate of future performance.")
            with st.spinner("Preparing the in-sample comparison…"):
                honesty_table(season, season_results(season, "in_sample"))
