"""Flag log: recorded model-vs-market disagreements and how they graded. Evaluation only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from components.charts import NEUTRAL, color_map, show, style  # noqa: E402
from components.data import header, query  # noqa: E402

st.set_page_config(page_title="Flag log", layout="wide")
header("Flag log", "Every recorded value flag (model vs market disagreement ≥ the edge threshold), graded "
       "after the game. ROI is the return of a hypothetical 1 unit per flag — 1-unit, model evaluation.")

f = query("select f.*, g.game_date, g.home_team_id, g.away_team_id, g.home_pts, g.away_pts "
          "from value_flags f join games g using (game_id) order by g.game_date")
if f.empty:
    st.info("No value flags yet. They are created by `uv run nba daily` / `uv run nba flags` when live odds exist "
            "(needs `ODDS_API_KEY`), then graded by `uv run nba grade`. See the Backtest page for historical "
            "flag evaluation.")
    st.stop()

c1, c2, c3 = st.columns(3)
d = pd.to_datetime(f.game_date)
lo, hi = c1.date_input("Dates", (d.min().date(), d.max().date()), key="fl_dates")
markets = c2.multiselect("Market", ["ml", "spread", "total"], default=["ml", "spread", "total"], key="fl_mk")
models = c3.multiselect("Model", sorted(f.model.unique()), default=sorted(f.model.unique()), key="fl_models")
f = f[(d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi)) & f.market.isin(markets) & f.model.isin(models)]
g = f[f.result.notna()]
w, lo_, p = (g.result == "win").sum(), (g.result == "loss").sum(), (g.result == "push").sum()
k = st.columns(5)
k[0].metric("Flags", f"{len(f)}", f"{len(f) - len(g)} pending", delta_color="off")
k[1].metric("Record (W-L-P)", f"{w}-{lo_}-{p}")
k[2].metric("Hit rate", f"{w / (w + lo_):.1%}" if w + lo_ else "—")
k[3].metric("ROI (1-unit, model evaluation)", f"{g.units.mean():+.1%}" if g.units.notna().any() else "—")
k[4].metric("Avg CLV (ML prob pts / spread-total pts)",
            f"{100 * g[g.market == 'ml'].clv.mean():+.1f} / {g[g.market != 'ml'].clv.mean():+.2f}"
            if g.clv.notna().any() else "—")

if len(g):
    cmap = color_map(models)
    fig = go.Figure()
    for m in models:
        gm = g[g.model == m].sort_values("game_date")
        if len(gm):
            fig.add_trace(go.Scatter(x=gm.game_date, y=gm.units.cumsum(), mode="lines", name=m,
                                     line=dict(color=cmap[m]), hovertemplate=f"{m}: %{{y:+.2f}} units<br>%{{x}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    show(style(fig, y_title="cumulative 1-unit return (evaluation)", x_title="game date"))

    dec = g[g.result.isin(["win", "loss"])]
    q = dec.groupby(pd.cut(dec.model_prob, np.linspace(0, 1, 11)), observed=True).agg(
        p=("model_prob", "mean"), hit=("result", lambda s: (s == "win").mean()), n=("result", "size"))
    fig = go.Figure([go.Scatter(x=q.p, y=q.hit, mode="lines+markers", name="flags", customdata=q.n,
                                hovertemplate="model %{x:.2f} → hit %{y:.2f} (n %{customdata})<extra></extra>"),
                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                                line=dict(color=NEUTRAL, dash="dash", width=1), hoverinfo="skip")])
    show(style(fig, y_title="hit rate", x_title="model probability (decile)"))

st.dataframe(f.drop(columns=["id"], errors="ignore").sort_values("game_date", ascending=False),
             width="stretch", hide_index=True)
st.download_button("Download CSV", f.to_csv(index=False), "value_flags.csv", "text/csv")
