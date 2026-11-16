"""Today's slate: each model's prediction vs the market line, value flags and context."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt  # noqa: E402
import sqlite3  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components.data import db_path, enriched, header, load_table, query, team_names  # noqa: E402
from components.sidebar import filter_sidebar  # noqa: E402
from nba_predict.config import get_settings  # noqa: E402
from nba_predict.features.build import team_profile, team_profile_games  # noqa: E402
from nba_predict.flags import evaluate_markets, market_view  # noqa: E402
from nba_predict.models.window_rating import WindowParams, predict_profiles  # noqa: E402

ET = ZoneInfo("America/New_York")
st.set_page_config(page_title="Today", layout="wide")
header("Today", "Model win probability, projected spread and total vs the market. Injuries are not included.")

day = st.date_input("Date (ET)", value=dt.datetime.now(ET).date(), key="today_date")
if not get_settings().odds_api_key:
    st.info("No `ODDS_API_KEY` set: live odds are unavailable, so today's games show **no odds** unless an "
            "ESPN line exists (ESPN lines arrive after games finish). Predictions still work.")

games = query("select * from games where game_date = ? and status != 'postponed' order by start_time_utc, game_id",
              (day.isoformat(),))
if games.empty:
    st.info("No games on this date. Run `uv run nba slate --date YYYY-MM-DD`.")
    st.stop()
preds = query("select p.* from predictions p join games g using (game_id) where g.game_date = ?", (day.isoformat(),))
flags = query("select f.* from value_flags f join games g using (game_id) where g.game_date = ?", (day.isoformat(),))
gf = load_table("game_features")
gf = gf[gf.game_id.isin(games.game_id)].set_index("game_id") if not gf.empty else gf
abbr = team_names()
history = enriched()
schedule = load_table("games")
best = query("select model from backtest_results where role='test' and segment='all' and metric='log_loss' "
             "and model in ('elo','window','ml') and run_at=(select max(run_at) from backtest_results "
             "where role='test') group by model order by avg(value) limit 1")
best_model = best.model.iloc[0] if not best.empty else "window"
st.caption(f"Edges use **{best_model}** (lowest backtest log loss). {len(games)} games.")

ro = sqlite3.connect(Path(db_path()).resolve().as_uri() + "?mode=ro", uri=True)
ro.row_factory = sqlite3.Row


def tip(g) -> str:
    if not g.start_time_utc:
        return "tip time unknown"
    t = dt.datetime.fromisoformat(str(g.start_time_utc).replace("Z", "+00:00")).astimezone(ET)
    return t.strftime("%-I:%M %p ET")


def playground(g, h: str, a: str, market: dict, key: str) -> None:
    """Recompute a window-model scenario without persisting predictions or flags."""
    if history.empty:
        st.info("No input games yet. Run `uv run nba ingest` then `uv run nba features`.")
        return
    st.caption("Live window-model scenario. Inputs use only games before this date. "
               "Injuries are not included. Changes are local to this playground.")
    base = WindowParams()
    options_key = f"{key}_window_options"
    if options_key not in st.session_state:
        st.session_state[options_key] = [f"l{n}" for n in get_settings().windows] + ["szn"]
    selection_key = f"{key}_windows"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = [w for w in base.weights if w in st.session_state[options_key]]
    add_col, button_col = st.columns([3, 1])
    n = add_col.number_input("Add N", min_value=1, value=7, step=1, key=f"{key}_add_n")
    if button_col.button("Add window", key=f"{key}_add_window"):
        window = f"l{n}"
        if window not in st.session_state[options_key]:
            st.session_state[options_key] = st.session_state[options_key] + [window]
        chosen = st.session_state[selection_key]
        st.session_state[selection_key] = list(dict.fromkeys([*chosen, window]))
    wins = st.multiselect("Windows", st.session_state[options_key], key=selection_key)
    weight_cols = st.columns(3)
    weights = {w: weight_cols[i % 3].slider(f"Weight {w}", 0.0, 1.0, base.weights.get(w, 0.1), 0.05,
                                           key=f"{key}_weight_{w}") for i, w in enumerate(wins)}
    c1, c2, c3 = st.columns(3)
    split_weight = c1.slider("Split weight", 0.0, 1.0, base.split_weight, 0.05, key=f"{key}_split")
    hca = c2.slider("Home-court advantage (points)", -5.0, 10.0, base.hca, 0.1, key=f"{key}_hca")
    use_sos = c3.toggle("Adjust for strength of schedule (SOS)", base.use_sos, key=f"{key}_sos")
    b2b = c1.slider("Back-to-back penalty (points)", 0.0, 5.0, base.b2b_pts, 0.1, key=f"{key}_b2b")
    rest = c2.slider("Rest advantage (points per day)", 0.0, 2.0, base.rest_pts, 0.1, key=f"{key}_rest")
    st.caption("Weights are normalized. Rest difference is capped at 3 days. "
               "This scenario uses no Elo prior; margin/total uncertainty is 12.5/18 points.")
    if not weights or sum(weights.values()) <= 0:
        st.info("Choose at least one window with a positive weight.")
        return
    params = WindowParams(weights=weights, split_weight=split_weight, hca=hca,
                          b2b_pts=b2b, rest_pts=rest, use_sos=use_sos)
    prior = history[history.game_date < day.isoformat()]
    home_box, away_box = st.columns(2)
    profiles, schedules, input_tables, profile_rows = {}, {}, [], []
    for side, team, team_id, venue, box in (
        ("h", h, g.home_team_id, "home", home_box),
        ("a", a, g.away_team_id, "away", away_box),
    ):
        team_history = prior[prior.team_id == team_id]
        box.markdown(f"**{team} input filters**")
        spec = filter_sidebar(team_history, f"{key}_{side}", container=box, presets=False)
        profiles[side] = {}
        for w in wins:
            n = None if w == "szn" else int(w[1:])
            for scope, v in (("all", "all"), ("split", venue)):
                profile = team_profile(prior, team, day, n, venue=v, spec=spec)
                profiles[side][f"{scope}_{w}"] = profile
                profile_rows.append(dict(team=team, scope=scope, window=w,
                                         weight=params.weights[w], **profile))
                inputs = team_profile_games(prior, team, day, n, venue=v, spec=spec)
                input_tables.append(inputs.assign(profile_team=team, scope=scope, window=w,
                                                 window_weight=params.weights[w]))
        # Rest comes from the actual schedule, independent of history filters.
        played = schedule[(schedule.game_date < day.isoformat()) & (schedule.season == g.season)
                          & schedule.status.isin(["final", "scheduled"])
                          & ((schedule.home_team_id == team_id) | (schedule.away_team_id == team_id))]
        rest_days = min(7, max(0, (day - dt.date.fromisoformat(played.game_date.max())).days - 1)) if len(played) else 7
        schedules[side] = dict(rest=rest_days, b2b=int(rest_days == 0))
    pred = predict_profiles(profiles["h"], profiles["a"], schedules["h"], schedules["a"], params)
    c1, c2, c3 = st.columns(3)
    c1.metric("Playground home win", f"{100 * pred['home_win_prob']:.1f}%")
    c2.metric("Playground home margin", f"{pred['pred_margin']:+.2f}")
    c3.metric("Playground total", f"{pred['pred_total']:.2f}" if pd.notna(pred["pred_total"]) else "unavailable")
    st.caption(f"Schedule: {h} rest {schedules['h']['rest']} days, B2B {schedules['h']['b2b']}; "
               f"{a} rest {schedules['a']['rest']} days, B2B {schedules['a']['b2b']}.")
    summary = pd.DataFrame(profile_rows)
    if (summary.gp < 5).any():
        st.warning("Some profiles have fewer than 5 input games. Missing ratings use model fallbacks; "
                   "a total needs both teams' scoring profiles.")
    evaluations = evaluate_markets(pred, market, params.sigma, params.sigma_total)
    if evaluations:
        st.markdown("**Playground model vs market**")
        st.dataframe(pd.DataFrame(evaluations), width="stretch", hide_index=True)
    else:
        st.info("No odds available for this scenario.")
    st.markdown("**Profiles used**")
    st.dataframe(summary, width="stretch", hide_index=True)
    st.markdown("**Input games by window and scope**")
    st.caption("A game appears once per profile it contributes to. Zero-weight windows are shown for inspection. "
               "Season-to-date uses this season; short last-N histories can include the previous season.")
    inputs = pd.concat(input_tables, ignore_index=True)
    if inputs.empty:
        st.info("No games match these input filters.")
    cols = ["profile_team", "scope", "window", "window_weight", "game_id", "game_date", "season",
            "team_abbr", "opp_abbr", "is_home", "pts", "opp_pts", "game_poss", "net", "pace", "opp_net_pre"]
    st.dataframe(inputs[[c for c in cols if c in inputs]], width="stretch", hide_index=True)
    st.download_button("Download input games CSV", inputs.to_csv(index=False), f"inputs_{g.game_id}.csv",
                       "text/csv", key=f"{key}_inputs_csv")


def game_card(g, game_predictions: pd.DataFrame, group: str) -> dict:
    h, a = abbr.get(g.home_team_id, "?"), abbr.get(g.away_team_id, "?")
    mv = market_view(ro, g.game_id) or {}
    gp = game_predictions.set_index("model")
    with st.container(border=True):
        score = f" — final {g.away_pts}-{g.home_pts}" if g.status == "final" else ""
        st.subheader(f"{a} @ {h}")
        st.caption(f"{tip(g)} · {g.status}{score} · {g.season_type}")
        c_model, c_market = st.columns([3, 2])
        if gp.empty:
            c_model.write("No prediction yet — run `uv run nba predict`.")
        else:
            tbl = pd.DataFrame({
                "home win %": (100 * gp.home_win_prob).round(1),
                "proj spread (home)": (-gp.pred_margin).round(1),
                "proj total": gp.pred_total.round(1),
                "made after tip": gp.made_after_tip.map({1: "yes", 0: "no"}).fillna("unknown")
                if "made_after_tip" in gp else "unknown",
            })
            c_model.dataframe(tbl, width="stretch")
            if "made_after_tip" in gp and (gp.made_after_tip == 1).any():
                c_model.warning("Prediction made after tip-off: excluded from the live flag record.")
        if not mv:
            c_market.markdown("**Market:** no odds")
        else:
            nv = mv.get("ml_novig")
            lines = [f"**Market** ({mv['source']}, {str(mv['pulled_at'])[:16]})"]
            if nv:
                lines.append(f"no-vig home win: **{100 * nv[0]:.1f}%** · best ML {mv['ml_price'][0]:+d} "
                             f"({mv['ml_book'][0]}) / {mv['ml_price'][1]:+d} ({mv['ml_book'][1]})")
            if mv.get("home_spread") is not None:
                lines.append(f"spread (home): **{mv['home_spread']:+.1f}**")
            if mv.get("total") is not None:
                lines.append(f"total: **{mv['total']:.1f}**")
            if best_model in gp.index and nv:
                lines.append(f"edge ({best_model}, home ML): **{100 * (gp.loc[best_model, 'home_win_prob'] - nv[0]):+.1f} pts**")
            c_market.markdown("  \n".join(lines))
        notes = []
        if len(gf) and g.game_id in gf.index:
            f = gf.loc[g.game_id]
            if f.get("h_b2b"):
                notes.append(f"{h} on a back-to-back")
            if f.get("a_b2b"):
                notes.append(f"{a} on a back-to-back")
            if pd.notna(f.get("h_rest")) and abs(f.h_rest - f.a_rest) >= 2:
                notes.append(f"rest {h} {int(f.h_rest)} vs {a} {int(f.a_rest)}")
            if min(f.get("h_gp_szn", 99), f.get("a_gp_szn", 99)) < 5:
                notes.append("low games played this season")
        gfl = flags[flags.game_id == g.game_id]
        for fl in gfl.itertuples(index=False):
            line = "" if pd.isna(fl.line) else f" {fl.line:+.1f}"
            res = f" → {fl.result}" if isinstance(fl.result, str) else ""
            notes.append(f"FLAG {fl.market} {fl.side}{line} @ {int(fl.price):+d} edge {100 * fl.edge:.1f} pts{res}")
        if notes:
            st.markdown(" · ".join(notes))
        with st.expander(f"Playground — {a} @ {h}"):
            playground(g, h, a, mv, f"pg_{g.game_id}_{group}")
        return dict(game=f"{a} @ {h}", group=group, status=g.status,
                    **{f"{m} home win %": round(100 * gp.loc[m, "home_win_prob"], 1)
                       for m in gp.index},
                    market=(round(100 * mv["ml_novig"][0], 1) if mv.get("ml_novig") else None))


active = games[games.status != "final"]
active_preds = preds[preds.game_id.isin(active.game_id)]
timing = active_preds.made_after_tip
groups = [
    ("Pre-tip predictions", active_preds[timing == 0]),
    ("Late predictions — made after tip", active_preds[timing == 1]),
    ("Prediction timing unknown", active_preds[timing.isna()]),
]
rows = []
for title, group_preds in groups:
    group_games = active[active.game_id.isin(group_preds.game_id)]
    if group_games.empty:
        continue
    st.header(title)
    if title.startswith("Late"):
        st.caption("These predictions are excluded from the live flag record.")
    for g in group_games.itertuples(index=False):
        rows.append(game_card(g, group_preds[group_preds.game_id == g.game_id], title))

for title, group_games in [
    ("Awaiting predictions", active[~active.game_id.isin(active_preds.game_id)]),
    ("Final games", games[games.status == "final"]),
]:
    if group_games.empty:
        continue
    st.header(title)
    for g in group_games.itertuples(index=False):
        rows.append(game_card(g, preds[preds.game_id == g.game_id], title))
ro.close()

with st.expander("Slate table / CSV"):
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False), f"slate_{day}.csv", "text/csv")
