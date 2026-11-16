"""Manage named, read-only SQL expressions over the game feature table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from components.charts import series_colors, show, style  # noqa: E402
from components.data import db_path, features, header, rw_conn  # noqa: E402
from nba_predict.analysis import custom  # noqa: E402
from nba_predict.db import QueryError, readonly_query  # noqa: E402

st.set_page_config(page_title="Custom features", layout="wide")
header("Custom features", "Named SQL expressions over game_features, available in Explorer.")
st.info("ML uses enabled features after the next backtest/predict. Existing results stay as recorded.")
st.caption("Use pre-game columns only; outcome columns or queries across future games can leak results.")

frame = features()
if frame.empty:
    st.info("No game features yet. Run `uv run nba ingest` then `uv run nba features`.")
    st.stop()

notice = st.session_state.pop("custom_notice", None)
if notice:
    st.success(notice)

conn = rw_conn()
try:
    definitions = custom.list_features(conn)
    st.subheader("Saved features")
    if definitions.empty:
        st.info("No custom features saved yet.")
    for feature in definitions.itertuples(index=False):
        with st.container(border=True):
            description, enabled, delete = st.columns([5, 2, 1])
            description.markdown(f"**{feature.name}**")
            description.code(feature.expression, language="sql")
            use = enabled.toggle("Use in model", value=bool(feature.use_in_model),
                                 key=f"custom_use_{feature.name}")
            if use != bool(feature.use_in_model):
                custom.set_use_in_model(conn, feature.name, use)
            if delete.button("Delete", key=f"custom_delete_{feature.name}"):
                custom.delete_feature(conn, feature.name)
                st.session_state.pop(f"custom_use_{feature.name}", None)
                st.session_state.custom_notice = f"Deleted {feature.name}."
                st.rerun()
finally:
    conn.close()

with st.expander("Available columns"):
    st.dataframe(pd.DataFrame({"column": frame.columns, "dtype": frame.dtypes.astype(str).values}),
                 hide_index=True, width="stretch")

st.subheader("Define a feature")
st.caption("Names use lowercase letters, digits and underscores, starting with a letter (max 41 characters). "
           "Saving an existing name replaces its expression and model setting.")
with st.form("custom_form"):
    name = st.text_input("Name", key="custom_name").strip()
    expression = st.text_area("Expression", "h_all_l5_net - a_all_l5_net", key="custom_expression").strip()
    use_in_model = st.checkbox("Enable for ML", key="custom_enable")
    preview = st.form_submit_button("Preview")
    save = st.form_submit_button("Save")

if preview or save:
    try:
        if save:
            conn = rw_conn()
            try:
                custom.add_feature(conn, name, expression, use_in_model=use_in_model)
            finally:
                conn.close()
            st.session_state.pop(f"custom_use_{name}", None)
            st.session_state.custom_notice = f"Saved {name}."
            st.rerun()
        # Preview executes only a guarded read; Save validates through add_feature.
        if not custom.NAME_RE.fullmatch(name):
            raise ValueError("Enter a valid feature name before previewing.")
        if name in frame.columns:
            raise ValueError(f"{name!r} collides with a game_features column")
        custom.check_expression(expression)
        values = readonly_query(
            db_path(), f"SELECT game_id, ({expression}) AS value FROM game_features "
            "ORDER BY game_date, game_id", max_rows=None)
        st.subheader("Preview — first 20 values")
        st.dataframe(values.head(20), hide_index=True, width="stretch")
        st.caption(f"{len(values):,} rows · {values.value.isna().sum():,} missing values. "
                   "Distribution uses all rows.")
        numeric = pd.to_numeric(values.value, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not numeric.empty:
            fig = px.histogram(x=numeric, nbins=30, color_discrete_sequence=series_colors())
            show(style(fig, x_title=name, y_title="games"))
            st.dataframe(numeric.describe().to_frame(name), width="stretch")
        elif values.value.notna().any():
            counts = values.value.dropna().astype(str).value_counts().head(20)
            st.bar_chart(counts)
        else:
            st.info("The expression has no non-missing values in this data.")
    except (QueryError, ValueError) as exc:
        st.error(str(exc))
