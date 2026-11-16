"""Backtest: walk-forward metrics with bootstrap CIs, early/rest segments, holdout, flag calibration
by edge bucket, and model vs market margin error."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from components.charts import NEUTRAL, color_map, show, style  # noqa: E402
from components.data import header, load_table, query, rw_conn  # noqa: E402

st.set_page_config(page_title="Backtest", layout="wide")
header("Backtest", "Walk-forward: every model is refit monthly on games before the month it scores. "
       "Holdout seasons use one frozen pre-season fit. 1-unit ROI is a model-evaluation metric only.")

with st.expander("Run a new backtest"):
    c1, c2, c3 = st.columns(3)
    test_s = c1.multiselect("Test seasons", ["2023-24", "2024-25", "2025-26", "2026-27"],
                            default=["2024-25", "2025-26"], key="bt_test")
    hold_s = c2.multiselect("Holdout seasons (reported only)", ["2025-26", "2026-27"], default=["2026-27"], key="bt_hold")
    n_boot = c3.number_input("Bootstrap resamples", 0, 5000, 1000, 100, key="bt_boot")
    if st.button("Run backtest", key="bt_run"):
        from nba_predict import backtest

        with st.spinner("Running walk-forward backtest…"):
            c = rw_conn()
            try:
                backtest.run(c, test_seasons=tuple(test_s), holdout=tuple(s for s in hold_s if s not in test_s),
                             n_boot=int(n_boot))
            finally:
                c.close()
        st.success("Backtest finished.")

res = load_table("backtest_results")
if res.empty:
    st.info("No backtest yet. Run `uv run nba backtest --holdout 2026-27` or use the button above.")
    st.stop()
runs = sorted(res.run_at.unique(), reverse=True)
run = st.selectbox("Run", runs, key="bt_runat")
r = res[res.run_at == run]
models = sorted(m for m in r.model.unique() if m != "market")
cmap = color_map(models)


def pivot(d: pd.DataFrame) -> pd.DataFrame:
    order = list(dict.fromkeys(d.metric))
    t = d.pivot_table(index="metric", columns="model", values="value", aggfunc="first")
    return t.reindex([m for m in order if m in t.index])


def ci_chart(d: pd.DataFrame, title: str) -> None:
    mets = [m for m in ("log_loss", "margin_mae", "mae_gap") if m in set(d.metric)]
    if not mets:
        return
    fig = make_subplots(rows=1, cols=len(mets), shared_yaxes=True, subplot_titles=mets, horizontal_spacing=0.06)
    for j, met in enumerate(mets, start=1):
        for m in models:
            v = d[(d.model == m)].set_index("metric").value
            if met not in v:
                continue
            lo, hi = v.get(f"{met}_ci_lo", np.nan), v.get(f"{met}_ci_hi", np.nan)
            fig.add_trace(go.Scatter(
                x=[v[met]], y=[m], mode="markers", marker=dict(color=cmap[m], size=10), name=m,
                showlegend=False,
                error_x=dict(type="data", symmetric=False, array=[hi - v[met]], arrayminus=[v[met] - lo],
                             color=cmap[m], thickness=2, width=4),
                hovertemplate=f"{m} {met}: %{{x:.4f}}<br>95% CI [{lo:.4f}, {hi:.4f}]<extra></extra>"), row=1, col=j)
        if met == "log_loss" and "market" in set(d.model):
            mk = d[(d.model == "market") & (d.metric == "market_log_loss")]
            if len(mk):
                fig.add_vline(x=float(mk.value.iloc[0]), line=dict(color=NEUTRAL, dash="dash", width=1),
                              row=1, col=j, annotation_text="market", annotation_position="top")
        if met == "mae_gap":
            fig.add_vline(x=0, line=dict(color=NEUTRAL, dash="dash", width=1), row=1, col=j)
    fig.update_layout(title=title)
    show(style(fig, height=80 + 42 * len(models)))


for role, label in (("test", "Walk-forward test seasons (used to pick the live model)"),
                    ("holdout", "Holdout — reported only, never used for fitting choices or model selection")):
    rr = r[(r.role == role) & (r.segment == "all")]
    if rr.empty:
        continue
    st.header(label)
    for season in sorted(rr.season.unique()):
        d = rr[rr.season == season]
        st.subheader(f"{season} ({role})")
        ci_chart(d, "Point estimate with 95% bootstrap CI (lower is better; mae_gap > 0 = market more accurate)")
        st.dataframe(pivot(d).style.format("{:.4f}", na_rep=""), width="stretch")

st.header("Early season vs rest")
st.caption("early = either team has played fewer than 15 games this season")
seg = r[r.segment != "all"]
if len(seg):
    keep = seg[seg.metric.isin(["n", "log_loss", "margin_mae", "mae_gap", "accuracy"])]
    t = keep.pivot_table(index=["role", "season", "segment", "metric"], columns="model", values="value", aggfunc="first")
    st.dataframe(t.style.format("{:.4f}", na_rep=""), width="stretch")

latest = runs[0]
st.header("Flag calibration by edge bucket")
bf = load_table("backtest_flags")
if run != latest:
    st.info("Per-flag detail is kept for the latest run only; showing the latest run.")
if bf.empty:
    st.info("No flag detail yet (run a backtest with ESPN/Odds API lines in a test season).")
else:
    bins = [0.03, 0.05, 0.08, 0.12, 1.0]
    bf["edge_bucket"] = pd.cut(bf.edge, bins=bins, right=False,
                               labels=["3–5 pts", "5–8 pts", "8–12 pts", "12+ pts"])
    c1, c2 = st.columns(2)
    fmodels = c1.multiselect("Models", sorted(bf.model.unique()), default=sorted(bf.model.unique()), key="bt_fm")
    fmarkets = c2.multiselect("Markets", ["ml", "spread", "total"], default=["ml", "spread", "total"], key="bt_fmk")
    b = bf[bf.model.isin(fmodels) & bf.market.isin(fmarkets)]
    decided = b.result.isin(["win", "loss"])
    g = b.assign(win=(b.result == "win").astype(float).where(decided)).groupby(
        ["model", "market", "edge_bucket"], observed=True)
    tbl = g.agg(n_flags=("edge", "size"), mean_edge=("edge", "mean"), mean_model_prob=("model_prob", "mean"),
                mean_market_prob=("market_prob", "mean"), hit_rate=("win", "mean"),
                roi_1unit=("units", "mean")).reset_index()
    tbl["model_minus_hit"] = tbl.mean_model_prob - tbl.hit_rate
    st.caption("A calibrated model's hit rate tracks its mean model probability; if hit rate tracks the market "
               "probability instead, the 'edge' is noise. roi_1unit = mean 1-unit return, model evaluation only.")
    st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns if c not in ("model", "market", "edge_bucket", "n_flags")}),
                 width="stretch", hide_index=True)
    fig = go.Figure()
    for m in fmodels:
        tm = b[b.model == m].assign(win=(b.result == "win").astype(float).where(decided))
        q = tm.groupby(pd.cut(tm.model_prob, np.linspace(0, 1, 11)), observed=True).agg(p=("model_prob", "mean"),
                                                                                       hit=("win", "mean"), n=("win", "size"))
        fig.add_trace(go.Scatter(x=q.p, y=q.hit, mode="lines+markers", name=m, line=dict(color=cmap.get(m)),
                                 customdata=q.n, hovertemplate=f"{m}: model %{{x:.2f}} → hit %{{y:.2f}} (n %{{customdata}})<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(color=NEUTRAL, dash="dash", width=1), hoverinfo="skip"))
    show(style(fig, y_title="hit rate of flagged side", x_title="model probability of flagged side"))

st.header("Model vs market: rolling margin error")
bp = load_table("backtest_predictions")
gf = query("select game_id, game_date, margin, m_home_spread from game_features where m_home_spread is not null "
           "and margin is not null")
if bp.empty or gf.empty:
    st.info("Needs a backtest run and games with closing spreads.")
else:
    c1, c2, c3 = st.columns(3)
    m = c1.selectbox("Model", sorted(bp.model.unique()), index=sorted(bp.model.unique()).index("window")
                     if "window" in set(bp.model) else 0, key="bt_mvm_model")
    seasons = sorted(bp.season.unique())
    sz = c2.selectbox("Season", seasons, index=len(seasons) - 1, key="bt_mvm_season")
    win = c3.number_input("Rolling games", 20, 500, 100, 10, key="bt_mvm_win")
    d = bp[(bp.model == m) & (bp.season == sz)].merge(gf.drop(columns="game_date"), on="game_id").sort_values("game_date")
    if d.empty:
        st.info("No games with closing spreads in this season.")
    else:
        d["model_err"] = (d.margin - d.pred_margin).abs()
        d["market_err"] = (d.margin + d.m_home_spread).abs()
        roll = d[["model_err", "market_err"]].rolling(int(win), min_periods=max(10, int(win) // 4)).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.game_date, y=roll.model_err, mode="lines", name=f"{m} margin MAE",
                                 line=dict(color=cmap.get(m, "#2a78d6"))))
        fig.add_trace(go.Scatter(x=d.game_date, y=roll.market_err, mode="lines", name="closing spread MAE",
                                 line=dict(color=NEUTRAL)))
        show(style(fig, y_title=f"margin MAE, rolling {int(win)} games (pts)", x_title="game date"))
        gap = r[(r.model == m) & (r.season == sz) & (r.segment == "all")].set_index("metric").value
        if "mae_gap" in gap:
            st.metric(f"mae_gap {sz} ({m} − market)", f"{gap['mae_gap']:+.2f} pts",
                      f"95% CI [{gap.get('mae_gap_ci_lo', np.nan):+.2f}, {gap.get('mae_gap_ci_hi', np.nan):+.2f}]",
                      delta_color="off")
        st.caption(f"{len(d)} games. The market's predicted margin is −closing home spread.")
