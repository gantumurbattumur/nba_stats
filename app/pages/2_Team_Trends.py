"""Team trends: rolling net rating by window, splits table, and which window predicts best."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from components.charts import series_colors, show, style  # noqa: E402
from components.data import enriched, header  # noqa: E402
from nba_predict.analysis.metrics import aggregate  # noqa: E402
from nba_predict.analysis.windows import window_strength  # noqa: E402

st.set_page_config(page_title="Team trends", layout="wide")
header("Team trends", "Rolling form by window, home/away splits, and how predictive each window is.")

df = enriched()
if df.empty:
    st.warning("No team-games yet. Run `uv run nba ingest`.")
    st.stop()


def parse_ns(text: str) -> list[int]:
    out = []
    for p in text.replace(" ", "").split(","):
        if p.isdigit() and int(p) >= 1:
            out.append(int(p))
    return sorted(set(out))


c1, c2, c3 = st.columns([1, 1, 2])
seasons = sorted(df.season.unique())
season = c1.selectbox("Season", seasons, index=len(seasons) - 1, key="tt_season")
teams = sorted(df[df.season == season].team_abbr.unique())
team = c2.selectbox("Team", teams, index=teams.index("BOS") if "BOS" in teams else 0, key="tt_team")
ns = parse_ns(c3.text_input("Rolling windows N (any, comma-separated)", "5,10,15", key="tt_ns")) or [10]

t = df[(df.team_abbr == team) & (df.season == season)].sort_values("game_date").reset_index(drop=True)
st.markdown(f"**{team} {season}** — {len(t)} games, record {aggregate(t, ['record']).iloc[0]['record']}")

fig = go.Figure()
cols = series_colors()
for i, n in enumerate(ns[:8]):
    r = t[["pts", "opp_pts", "game_poss"]].rolling(n, min_periods=1).sum()
    net = 100 * (r.pts - r.opp_pts) / r.game_poss
    fig.add_trace(go.Scatter(x=t.game_date, y=net, mode="lines", name=f"L{n}", line=dict(color=cols[i]),
                             hovertemplate=f"L{n} net %{{y:.1f}}<br>%{{x}}<extra></extra>"))
fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.5)", width=1))
show(style(fig, y_title="net rating (pts / 100 poss)", x_title="game date"))
st.caption("Rolling sum-based net rating through each game (includes that game). Early points use fewer games.")

st.subheader("Splits by window")
rows = []
for venue in ("all", "home", "away"):
    v = t if venue == "all" else t[t.is_home == (1 if venue == "home" else 0)]
    for n in [5, 8, 10, 15, None]:
        sel = v if n is None else v.tail(n)
        a = aggregate(sel, ["record", "net", "ortg", "drtg", "pace", "efg", "ats_record", "ou_record",
                            "cover_margin"]).iloc[0].to_dict()
        rows.append(dict(venue=venue, window="season" if n is None else f"L{n}", **a))
tbl = pd.DataFrame(rows)
st.dataframe(tbl.style.format({c: "{:.1f}" for c in ("net", "ortg", "drtg", "pace", "cover_margin")} | {"efg": "{:.3f}"}),
             width="stretch", hide_index=True)
st.caption("ATS (W-L-P) and O/U (O-U-P) need closing lines: complete for 2025-26 and 2026-27 only.")

st.subheader("Window strength: which last-N predicts the next game best?")
c1, c2 = st.columns([2, 2])
wins = parse_ns(c1.text_input("Windows to compare (any N)", "3,5,8,10,15,20,30", key="tt_ws")) or [10]
ws_seasons = c2.multiselect("Seasons", seasons, default=seasons[-2:], key="tt_ws_seasons")
ws = window_strength(df, wins, ws_seasons or None)
fig = go.Figure(go.Bar(x=[f"L{w}" for w in ws.window], y=ws["corr"], marker_color=cols[0],
                       customdata=ws.n, hovertemplate="%{x}: r = %{y:.3f} (n = %{customdata})<extra></extra>"))
show(style(fig, y_title="correlation with home margin", x_title="window"))
st.caption("Correlation of (home prior-N net − away prior-N net) with the home margin; only games where both "
           "teams have N prior games (windows cross seasons). Higher = more predictive.")
st.dataframe(ws, width="stretch", hide_index=True)
