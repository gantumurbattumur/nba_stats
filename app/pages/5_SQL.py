"""SQL console: any single SELECT through a read-only connection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from components.charts import series_colors, show, style  # noqa: E402
from components.data import db_path, header, table_names  # noqa: E402
from nba_predict.db import QueryError, readonly_query  # noqa: E402

st.set_page_config(page_title="SQL", layout="wide")
header("SQL console", "Read-only: one SELECT per run (writes, ATTACH and PRAGMAs are rejected).")

with st.sidebar:
    st.subheader("Tables")
    for t in table_names():
        with st.expander(t):
            cols = readonly_query(db_path(), f"select name, type from pragma_table_info('{t}')", max_rows=None)
            st.dataframe(cols, width="stretch", hide_index=True, height=min(35 * (len(cols) + 1), 300))

EXAMPLE = """-- MIL road games vs >= .500 opponents, 2026-27
select game_date, opp_abbr, wl, pts, opp_pts, round(net, 1) net
from team_games_enriched
where team_abbr = 'MIL' and is_home = 0 and season = '2026-27' and opp_win_pct_pre >= 0.5
order by game_date desc"""
sql = st.text_area("SQL", EXAMPLE, height=180, key="sql_text")
if st.button("Run", key="sql_run", type="primary"):
    st.session_state["sql_last"] = sql
q = st.session_state.get("sql_last")
if q:
    try:
        df = readonly_query(db_path(), q)
    except QueryError as e:
        st.error(f"Query rejected or failed: {e}")
        st.stop()
    st.markdown(f"**{len(df):,} rows**" + (" (truncated)" if df.attrs.get("truncated") else ""))
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "query.csv", "text/csv")
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(df) and num:
        with st.expander("Chart"):
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("x", list(df.columns), key="sql_x")
            y = c2.selectbox("y", num, key="sql_y")
            kind = c3.radio("Type", ["bar", "line", "scatter"], horizontal=True, key="sql_kind")
            fig = {"bar": px.bar, "line": px.line, "scatter": px.scatter}[kind](df, x=x, y=y)
            fig.update_traces(marker_color=series_colors()[0]) if kind != "line" else fig.update_traces(
                line_color=series_colors()[0])
            show(style(fig, y_title=y, x_title=x))
