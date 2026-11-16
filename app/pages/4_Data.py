"""Data: browse every pipeline table with filters, column chooser, row count and CSV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components.data import enriched, header, load_table, table_names  # noqa: E402
from components.sidebar import filter_sidebar  # noqa: E402
from nba_predict.analysis.filters import apply_filter  # noqa: E402

st.set_page_config(page_title="Data", layout="wide")
header("Data", "Every stage of the pipeline, as stored.")

ORDER = ["games", "team_games", "team_games_enriched", "game_features", "odds_snapshots", "predictions",
         "value_flags", "api_usage", "backtest_results", "backtest_predictions", "backtest_flags",
         "custom_features", "filter_presets", "derived_meta"]
names = [n for n in ORDER if n in table_names()] + [n for n in table_names() if n not in ORDER]
name = st.selectbox("Table", names, index=names.index("team_games_enriched") if "team_games_enriched" in names else 0,
                    key="data_table")
df = enriched() if name == "team_games_enriched" else load_table(name)
total = len(df)

if name == "team_games_enriched" and total:
    df = apply_filter(df, filter_sidebar(df, "data"))
else:
    sb = st.sidebar
    if "season" in df and total:
        ss = sb.multiselect("Seasons", sorted(df.season.dropna().unique()), key="data_seasons")
        if ss:
            df = df[df.season.isin(ss)]
    if "game_date" in df and total:
        d = pd.to_datetime(df.game_date, errors="coerce")
        if d.notna().any():
            dates = sb.date_input("Game dates", (d.min().date(), d.max().date()), key="data_dates")
            if len(dates) == 2:
                lo, hi = dates
                df = df[(d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))]
            else:
                sb.info("Choose both a start and an end date to apply the date filter.")
    text = sb.text_input("Contains (any column)", key="data_text")
    if text:
        df = df[df.astype(str).apply(lambda c: c.str.contains(text, case=False, regex=False)).any(axis=1)]

cols = st.multiselect("Columns (empty = all)", list(df.columns), key=f"data_cols_{name}")
view = df[cols] if cols else df
st.markdown(f"**{len(view):,}** of {total:,} rows · {view.shape[1]} columns")
if view.empty:
    st.info("No rows match these filters")
else:
    st.dataframe(view.head(5000), width="stretch", hide_index=True)
    if len(view) > 5000:
        st.caption("Showing the first 5,000 rows; the CSV has all of them.")
    st.download_button("Download CSV", view.to_csv(index=False), f"{name}.csv", "text/csv")
