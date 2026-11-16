"""NBA stats & odds analysis dashboard — entry page.  Run: uv run streamlit run app/dashboard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="NBA stats & odds analysis", layout="wide")
st.navigation({
    "Games": [
        st.Page("pages/0_Game_Brief.py", title="Game Brief", default=True),
        st.Page("pages/11_Results_Review.py", title="Results Review"),
        st.Page("pages/3_Explorer.py", title="Explore"),
    ],
    "Advanced": [
        st.Page("pages/1_Today.py", title="Today · model playground"),
        st.Page("pages/2_Team_Trends.py", title="Team Trends"),
        st.Page("pages/4_Data.py", title="Data"),
        st.Page("pages/5_SQL.py", title="SQL"),
        st.Page("pages/6_Flag_Log.py", title="Flag Log"),
        st.Page("pages/7_Backtest.py", title="Backtest"),
        st.Page("pages/8_Custom_Features.py", title="Custom Features"),
        st.Page("pages/9_Line_Movement.py", title="Line Movement"),
        st.Page("pages/10_Data_Health.py", title="Data Health"),
    ],
}).run()
