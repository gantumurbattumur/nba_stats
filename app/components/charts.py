"""Plotly styling shared by every page: fixed categorical order (validated palette, light and
dark steps), 2px lines, 8px markers, recessive grid, hover on every mark, legend for >= 2 series."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
NEUTRAL = "#8a8985"


def _dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - older Streamlit / no context (AppTest)
        return False


def series_colors() -> list[str]:
    return DARK if _dark() else LIGHT


def color_map(keys) -> dict:
    """Color follows the entity: assign in a stable (sorted) order, never by rank."""
    cols = series_colors()
    return {k: cols[i % len(cols)] for i, k in enumerate(sorted(map(str, keys)))}


def style(fig: go.Figure, height: int = 360, y_title: str | None = None, x_title: str | None = None) -> go.Figure:
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line=dict(width=2))
    fig.update_traces(selector=dict(type="scatter"), marker=dict(size=8))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      bargap=0.25)
    fig.update_xaxes(showgrid=False, title=x_title)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False, title=y_title)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit")
