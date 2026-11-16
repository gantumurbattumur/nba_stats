"""Data health: coverage, line coverage by month, suspicious rows and derived-table freshness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from components.charts import series_colors, show, style  # noqa: E402
from components.data import db_path, header  # noqa: E402
from nba_predict.analysis import health  # noqa: E402
from nba_predict.ingest import coverage  # noqa: E402

st.set_page_config(page_title="Data health", layout="wide")
header("Data health", "Same numbers as `uv run nba status`, plus row-level checks.")

c = sqlite3.connect(Path(db_path()).resolve().as_uri() + "?mode=ro", uri=True)
try:
    cov = coverage(c)
    by_month = health.line_coverage_by_month(c)
    fav = health.fav_disagree_rows(c)
    mins = health.bad_minutes_rows(c)
    try:
        fresh = health.derived_freshness(c)
    except Exception as e:  # noqa: BLE001 - derived tables not built yet
        fresh = None
        st.caption(f"Freshness unavailable: {e}")
finally:
    c.close()

st.subheader("Coverage by season and season type")
st.dataframe(cov, width="stretch", hide_index=True)

st.subheader("Closing-line coverage by month")
if len(by_month):
    by_month["share"] = by_month.with_closing_line / by_month.final_games
    cols = series_colors()
    fig = go.Figure(go.Bar(x=by_month.month, y=by_month.share, marker_color=cols[0],
                           customdata=by_month[["with_closing_line", "final_games"]].values,
                           hovertemplate="%{x}: %{y:.0%} (%{customdata[0]} of %{customdata[1]})<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    show(style(fig, height=300, y_title="final games with a closing line", x_title="month"))
    st.caption("2023-24 and 2024-25 have no ESPN pickcenter lines; the ESPN core API does have 2024-25 lines "
               "(not yet backfilled).")
    with st.expander("Table"):
        st.dataframe(by_month, width="stretch", hide_index=True)

c1, c2 = st.columns(2)
c1.subheader(f"Moneyline vs spread favorite disagree ({len(fav)})")
c1.dataframe(fav, width="stretch", hide_index=True) if len(fav) else c1.success("None")
c2.subheader(f"Bad team minutes ({len(mins)})")
c2.caption("Team MIN not 240 + 25k, or different from the opponent's (pace uses max of the two).")
c2.dataframe(mins, width="stretch", hide_index=True) if len(mins) else c2.success("None")

st.subheader("Derived tables")
if fresh is not None:
    st.dataframe(fresh, width="stretch", hide_index=True)
