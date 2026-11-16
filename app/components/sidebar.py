"""``filter_sidebar``: one widget per FilterSpec field, plus preset load/save/delete."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from components.data import rw_conn
from nba_predict.analysis.filters import FilterSpec, delete_preset, list_presets, load_preset, save_preset

ANY = "any"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _opts(df: pd.DataFrame, col: str) -> list:
    return sorted(df[col].dropna().unique().tolist()) if col in df else []


def _apply_preset_to_state(spec: FilterSpec, k: str) -> None:
    s = st.session_state
    s[f"{k}_seasons"] = spec.seasons or []
    s[f"{k}_stypes"] = spec.season_types or []
    s[f"{k}_teams"] = spec.teams or []
    s[f"{k}_opps"] = spec.opponents or []
    s[f"{k}_venue"] = spec.venue
    s[f"{k}_last_n"] = spec.last_n or 0
    s[f"{k}_use_dates"] = bool(spec.date_from or spec.date_to)
    if spec.date_from:
        s[f"{k}_dfrom"] = spec.date_from
    if spec.date_to:
        s[f"{k}_dto"] = spec.date_to
    s[f"{k}_rest"] = (spec.rest_min if spec.rest_min is not None else 0,
                      spec.rest_max if spec.rest_max is not None else 7)
    s[f"{k}_b2b"] = ANY if spec.b2b is None else ("yes" if spec.b2b else "no")
    s[f"{k}_role"] = spec.role
    s[f"{k}_result"] = spec.result
    s[f"{k}_ats"] = spec.ats
    s[f"{k}_ou"] = spec.ou
    s[f"{k}_use_spread"] = spec.spread_min is not None or spec.spread_max is not None
    s[f"{k}_spread"] = (spec.spread_min if spec.spread_min is not None else -20.0,
                        spec.spread_max if spec.spread_max is not None else 20.0)
    s[f"{k}_use_total"] = spec.total_min is not None or spec.total_max is not None
    s[f"{k}_total"] = (spec.total_min if spec.total_min is not None else 200.0,
                       spec.total_max if spec.total_max is not None else 250.0)
    s[f"{k}_opp_wp"] = spec.opp_min_win_pct or 0.0
    s[f"{k}_opp_rank"] = spec.opp_net_rank_max or 0
    s[f"{k}_confs"] = spec.conferences or []
    s[f"{k}_oconfs"] = spec.opp_conferences or []
    s[f"{k}_divs"] = spec.divisions or []
    s[f"{k}_months"] = spec.months or []
    s[f"{k}_wdays"] = [WEEKDAYS[i] for i in (spec.weekdays or [])]


def filter_sidebar(df: pd.DataFrame, key: str, container=None, presets: bool = True) -> FilterSpec:
    """Render filters for ``df`` (team_games_enriched-shaped) in ``container`` (default sidebar)."""
    box = container or st.sidebar
    k = key
    if presets:
        c = rw_conn()
        try:
            names = list_presets(c)
            with box.expander("Presets", expanded=False):
                pick = st.selectbox("Preset", ["—"] + names, key=f"{k}_preset")
                c1, c2 = st.columns(2)
                if c1.button("Load", key=f"{k}_load", disabled=pick == "—"):
                    _apply_preset_to_state(load_preset(c, pick), k)
                    st.rerun()
                if c2.button("Delete", key=f"{k}_del", disabled=pick == "—"):
                    delete_preset(c, pick)
                    st.rerun()
                new = st.text_input("Save current filters as", key=f"{k}_newname")
                save_clicked = st.button("Save preset", key=f"{k}_save", disabled=not new)
        finally:
            c.close()
    else:
        save_clicked, new = False, ""

    seasons = box.multiselect("Seasons", _opts(df, "season"), key=f"{k}_seasons")
    stypes = box.multiselect("Season types", ["regular", "playin", "playoffs"], key=f"{k}_stypes")
    teams = box.multiselect("Teams", _opts(df, "team_abbr"), key=f"{k}_teams")
    opps = box.multiselect("Opponents", _opts(df, "opp_abbr"), key=f"{k}_opps")
    venue = box.radio("Venue", ["all", "home", "away"], horizontal=True, key=f"{k}_venue")
    last_n = box.number_input("Last N games per team (0 = all)", min_value=0, max_value=500, step=1,
                              key=f"{k}_last_n")
    with box.expander("More filters", expanded=False):
        use_dates = st.checkbox("Date range", key=f"{k}_use_dates")
        dmin = pd.to_datetime(df.game_date).min().date() if len(df) else dt.date(2023, 10, 1)
        dmax = pd.to_datetime(df.game_date).max().date() if len(df) else dt.date.today()
        dfrom = st.date_input("From", value=dmin, key=f"{k}_dfrom") if use_dates else None
        dto = st.date_input("To", value=dmax, key=f"{k}_dto") if use_dates else None
        rest = st.slider("Rest days", 0, 7, (0, 7), key=f"{k}_rest")
        b2b = st.selectbox("Back-to-back", [ANY, "yes", "no"], key=f"{k}_b2b")
        role = st.selectbox("Role (closing line)", ["any", "favorite", "underdog"], key=f"{k}_role")
        use_spread = st.checkbox("Spread range (team perspective)", key=f"{k}_use_spread")
        spread = st.slider("Spread", -25.0, 25.0, (-20.0, 20.0), 0.5, key=f"{k}_spread") if use_spread else None
        use_total = st.checkbox("Total range", key=f"{k}_use_total")
        total = st.slider("Total line", 180.0, 270.0, (200.0, 250.0), 0.5, key=f"{k}_total") if use_total else None
        result = st.selectbox("Result", ["any", "win", "loss"], key=f"{k}_result")
        ats = st.selectbox("ATS", ["any", "cover", "no_cover", "push"], key=f"{k}_ats")
        ou = st.selectbox("Over/under", ["any", "over", "under", "push"], key=f"{k}_ou")
        opp_wp = st.slider("Opponent win% at least (0 = off)", 0.0, 1.0, 0.0, 0.05, key=f"{k}_opp_wp")
        opp_rank = st.number_input("Opponent net-rating rank at most (0 = off)", 0, 30, 0, key=f"{k}_opp_rank")
        confs = st.multiselect("Conference", ["East", "West"], key=f"{k}_confs")
        oconfs = st.multiselect("Opponent conference", ["East", "West"], key=f"{k}_oconfs")
        divs = st.multiselect("Division", _opts(df, "division"), key=f"{k}_divs")
        months = st.multiselect("Months", list(range(1, 13)), key=f"{k}_months")
        wdays = st.multiselect("Weekdays", WEEKDAYS, key=f"{k}_wdays")

    spec = FilterSpec(
        seasons=seasons or None, season_types=stypes or None, teams=teams or None, opponents=opps or None,
        venue=venue, last_n=int(last_n) or None, date_from=dfrom, date_to=dto,
        rest_min=rest[0] if rest[0] > 0 else None, rest_max=rest[1] if rest[1] < 7 else None,
        b2b=None if b2b == ANY else b2b == "yes", role=role,
        spread_min=spread[0] if spread else None, spread_max=spread[1] if spread else None,
        total_min=total[0] if total else None, total_max=total[1] if total else None,
        result=result, ats=ats, ou=ou, opp_min_win_pct=opp_wp or None, opp_net_rank_max=int(opp_rank) or None,
        conferences=confs or None, opp_conferences=oconfs or None, divisions=divs or None,
        months=months or None, weekdays=[WEEKDAYS.index(w) for w in wdays] or None,
    )
    if save_clicked and new:
        c = rw_conn()
        try:
            save_preset(c, new, spec)
        finally:
            c.close()
        box.success(f"Saved preset {new!r}")
    box.caption(f"Filter: {spec.describe()}")
    return spec
