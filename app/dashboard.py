"""NBA stats & odds analysis dashboard — entry page.  Run: uv run streamlit run app/dashboard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from components.data import db_path, header, query, table_names  # noqa: E402

st.set_page_config(page_title="NBA stats & odds analysis", layout="wide")
header("NBA stats & odds analysis", "Every pipeline stage is browsable: games → enriched team-games → "
       "features → predictions → value flags.")

if "games" not in table_names():
    st.warning(f"No data yet in {db_path()}. Run `uv run nba init-db && uv run nba ingest`.")
    st.stop()

s = query("select count(*) n, sum(status='final') fin, sum(status='scheduled') sch, max(case when status='final' "
          "then game_date end) last_final from games").iloc[0]
o = query("select count(*) n, sum(source='espn') espn, sum(source='oddsapi') api from odds_snapshots").iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", f"{int(s.n):,}", f"{int(s.sch or 0)} scheduled", delta_color="off")
c2.metric("Latest final game", s.last_final or "—")
c3.metric("Odds snapshots", f"{int(o.n):,}", f"ESPN {int(o.espn or 0):,} · Odds API {int(o.api or 0):,}",
          delta_color="off")
p = query("select count(*) n, max(created_at) last from predictions").iloc[0]
c4.metric("Predictions", f"{int(p.n):,}", f"last {str(p['last'] or '—')[:16]}", delta_color="off")

st.markdown("""
**Pages** (sidebar): **Today** — slate with model vs market · **Explorer** — filter any team-game set and
aggregate any metric · **Team trends** · **Backtest** · **Data** · **SQL** · **Flag log** · **Line movement** ·
**Data health** · **Custom features**.

Daily routine: `uv run nba daily` in the morning, `uv run nba odds --closing` ~30 min before the first tip,
`uv run nba grade` the next morning. Injuries are not included.
""")
st.caption(f"Database: `{db_path()}`")
