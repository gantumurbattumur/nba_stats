"""Explorer: filter any set of team-games and aggregate any metric, grouped any way."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from components.charts import series_colors, show, style  # noqa: E402
from components.data import db_path, enriched, header  # noqa: E402
from components.sidebar import filter_sidebar  # noqa: E402
from nba_predict.analysis.filters import apply_filter  # noqa: E402
from nba_predict.analysis.metrics import METRICS, aggregate  # noqa: E402

st.set_page_config(page_title="Explorer", layout="wide")
header("Explorer", "Filters in the sidebar → aggregate table, chart and matching games. Ratings are sum-based.")

df = enriched()
if df.empty:
    st.warning("No team-games yet. Run `uv run nba ingest` then `uv run nba features`.")
    st.stop()
try:  # custom features (game-level) joined onto every team-game row
    from nba_predict.analysis.custom import evaluate

    cf = evaluate(db_path())
    if cf.shape[1] > 1:
        df = df.merge(cf, on="game_id", how="left")
except Exception as e:  # noqa: BLE001 - no game_features yet / broken feature: explorer still works
    st.caption(f"Custom features unavailable: {e}")

spec = filter_sidebar(df, "ex")
out = apply_filter(df, spec)

GROUPS = ["team_abbr", "opp_abbr", "is_home", "rest_days", "month", "weekday", "is_favorite", "season",
          "season_type", "b2b", "game_date"]
c1, c2, c3 = st.columns([3, 2, 2])
metric_keys = c1.multiselect("Metrics", list(METRICS), default=["record", "net", "ortg", "drtg", "pace"],
                             format_func=lambda k: f"{METRICS[k].label} ({k})", key="ex_metrics")
group_by = c2.multiselect("Group by", GROUPS, default=["team_abbr"], key="ex_group")
numeric = sorted(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in METRICS)
raw_cols = c3.multiselect("Raw columns", numeric, key="ex_raw")
raw_how = c3.selectbox("Raw aggregation", ["mean", "median", "sum"], key="ex_raw_how")

st.markdown(f"**{spec.describe()}** — {len(out):,} team-games")
if out.empty:
    st.info("No games match these filters")
    st.stop()

agg = aggregate(out, metric_keys, group_by=group_by or None, raw_aggs={c: raw_how for c in raw_cols})
st.dataframe(agg, width="stretch", hide_index=True)
st.download_button("Download aggregate CSV", agg.to_csv(index=False), "aggregate.csv", "text/csv")

plottable = [c for c in agg.columns if c not in (group_by or []) and pd.api.types.is_numeric_dtype(agg[c])]
if group_by and plottable:
    y = st.selectbox("Chart metric", plottable, index=min(1, len(plottable) - 1), key="ex_chart_y")
    color = series_colors()[0]
    if "game_date" in group_by:
        d = agg.sort_values("game_date")
        fig = px.line(d, x="game_date", y=y, color=group_by[1] if len(group_by) > 1 and group_by[0] == "game_date" else None,
                      markers=True)
        if len(group_by) == 1:
            fig.update_traces(line_color=color, marker_color=color)
    else:
        d = agg.copy()
        d["group"] = d[group_by].astype(str).agg(" · ".join, axis=1)
        d = d.sort_values(y, ascending=False)
        fig = px.bar(d, x="group", y=y, hover_data=["gp"])
        fig.update_traces(marker_color=color)
    show(style(fig, y_title=y, x_title=" · ".join(group_by)))

with st.expander(f"Matching team-games ({len(out):,})"):
    cols = ["game_date", "season", "team_abbr", "opp_abbr", "is_home", "wl", "pts", "opp_pts", "net", "pace",
            "rest_days", "team_spread", "total_line", "ats", "ou"]
    view = out.sort_values("game_date", ascending=False)
    st.dataframe(view[[c for c in cols if c in view]], width="stretch", hide_index=True)
    st.download_button("Download games CSV", view.to_csv(index=False), "games.csv", "text/csv")
