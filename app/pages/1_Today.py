"""Today's slate: each model's prediction vs the market line, value flags and context."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt  # noqa: E402
import json  # noqa: E402
import sqlite3  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components.data import db_path, header, load_table, query, team_names  # noqa: E402
from nba_predict.config import get_settings  # noqa: E402
from nba_predict.flags import market_view  # noqa: E402

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


rows = []
for g in games.itertuples(index=False):
    h, a = abbr.get(g.home_team_id, "?"), abbr.get(g.away_team_id, "?")
    mv = market_view(ro, g.game_id) or {}
    gp = preds[preds.game_id == g.game_id].set_index("model")
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
        rows.append(dict(game=f"{a} @ {h}", **{f"{m} home win %": round(100 * gp.loc[m, "home_win_prob"], 1)
                                                for m in gp.index},
                         market=(round(100 * mv["ml_novig"][0], 1) if mv.get("ml_novig") else None)))
ro.close()

with st.expander("Slate table / CSV"):
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False), f"slate_{day}.csv", "text/csv")
