"""Line movement: open → close spread and total moves, and results by move direction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from components.charts import series_colors, show, style  # noqa: E402
from components.data import db_path, header, team_names  # noqa: E402
from nba_predict.analysis.line_movement import line_moves, summarize_moves  # noqa: E402

st.set_page_config(page_title="Line movement", layout="wide")
header("Line movement", "Opening vs closing ESPN lines (home-perspective spread).")

c = sqlite3.connect(Path(db_path()).resolve().as_uri() + "?mode=ro", uri=True)
try:
    d = line_moves(c)
finally:
    c.close()
if d.empty:
    st.info("No games with both an opening and a closing ESPN line yet.")
    st.stop()

st.markdown(f"**{len(d)} games** · {d.book.nunique()} book(s): {', '.join(sorted(d.book.unique()))}")
st.caption(f"Opening/closing pairs cover {d.game_date.min()} through {d.game_date.max()}. "
           "Coverage depends on the available ESPN core data. `nba espn --with-open` collects opening lines.")
sp, tt = summarize_moves(d)
c1, c2 = st.columns(2)
c1.subheader("Spread: did the side the line moved toward cover?")
c1.dataframe(sp.style.format({"mean_move": "{:+.2f}", "cover_rate_close": "{:.1%}", "cover_rate_open": "{:.1%}"}),
             width="stretch", hide_index=True)
c1.caption("cover_rate_close: vs the closing spread · cover_rate_open: vs the opening spread (pushes excluded).")
c2.subheader("Total: over rate by move direction")
c2.dataframe(tt.style.format({"mean_move": "{:+.2f}", "over_rate_close": "{:.1%}"}), width="stretch", hide_index=True)

col = series_colors()
for title, x, xt in (("Spread move (close − open, home perspective)", "spread_move", "points"),
                     ("Total move (close − open)", "total_move", "points")):
    vc = d[x].round(1).value_counts().sort_index()
    fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=col[0],
                           hovertemplate="move %{x:+.1f}: %{y} games<extra></extra>"))
    fig.update_layout(title=title)
    show(style(fig, height=300, y_title="games", x_title=xt))

ab = team_names()
view = d.assign(game=[f"{ab.get(a, '?')} @ {ab.get(h, '?')}" for h, a in zip(d.home_team_id, d.away_team_id)])
st.dataframe(view[["game_date", "game", "open_spread", "close_spread", "spread_move", "moved_toward", "margin",
                   "cover_close", "open_total", "close_total", "total_move", "total_pts", "ou_close"]]
             .sort_values("game_date"), width="stretch", hide_index=True)
st.download_button("Download CSV", view.to_csv(index=False), "line_moves.csv", "text/csv")
