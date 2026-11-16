import numpy as np
import pytest

from nba_predict import backtest
from nba_predict.analysis.custom import add_feature
from nba_predict.features.build import build_feature_table
from nba_predict.models import ml as ml_mod

METRICS = {"n", "accuracy", "brier", "log_loss", "margin_mae", "total_mae", "ece"}


@pytest.fixture
def feats(fixture_db, enriched):
    return build_feature_table(fixture_db)


def test_backtest_walk_forward_no_future(fixture_db, feats, monkeypatch):
    seen = []
    real_fit, real_pred = ml_mod.MLModel.fit, ml_mod.MLModel.predict

    def fit(self, df):
        seen.append(("fit", df.game_date.max()))
        return real_fit(self, df)

    def predict(self, df):
        seen.append(("pred", df.game_date.min()))
        return real_pred(self, df)

    monkeypatch.setattr(ml_mod.MLModel, "fit", fit)
    monkeypatch.setattr(ml_mod.MLModel, "predict", predict)
    backtest.run(fixture_db, test_seasons=("2025-26",), n_boot=50)
    fits = [d for k, d in seen if k == "fit"]
    assert len(fits) == 2  # Oct and Nov 2025
    # after each fit: an in-sample predict (sigma from training residuals), then the test month
    test_preds = [d for k, d in seen if k == "pred" and d >= "2025-10-01"]
    assert [p[:7] for p in test_preds] == ["2025-10", "2025-11"]
    for f, p in zip(fits, test_preds):
        assert f < f"{p[:7]}-01"  # every training game is before the scored month


def test_backtest_metrics_shape(fixture_db, feats):
    full = backtest.run(fixture_db, test_seasons=("2025-26",), n_boot=200)
    assert set(full.columns) == {"model", "season", "metric", "value", "segment", "role"}
    out = full[(full.segment == "all") & (full.role == "test")]
    assert {"home_always", "elo", "window", "window_noprior", "ml"} <= set(out.model)
    for m in ("home_always", "elo", "window", "ml"):
        got = set(out[out.model == m].metric)
        assert METRICS <= got, (m, METRICS - got)
        assert {"n_flags", "hit_rate", "roi", "log_loss_market_games"} <= got
    assert "market_log_loss" in set(out[out.model == "market"].metric)
    s2 = feats[(feats.season == "2025-26") & feats.home_win.notna()]
    ha = out[(out.model == "home_always") & (out.metric == "accuracy")].value.iloc[0]
    # training home win rate < .5 in the fixture -> home_always picks away every game
    assert ha == pytest.approx(1 - s2.home_win.mean()) or ha == pytest.approx(s2.home_win.mean())
    assert out[(out.metric == "n")].value.eq(len(s2)).all()
    win = out[out.model == "window"].set_index("metric").value
    assert np.isfinite(win["effective_hca"]) and np.isfinite(win["hca"])
    stored = fixture_db.execute("select count(*), count(distinct run_at) from backtest_results").fetchone()
    assert stored[0] == len(full) and stored[1] == 1
    assert not set(out.metric) & {"stake", "kelly", "bankroll"}


def test_best_model_default(conn, fixture_db, feats):
    assert backtest.best_model(conn) == "window"
    backtest.run(fixture_db, test_seasons=("2025-26",), n_boot=50)
    out = fixture_db.execute(
        "select model, avg(value) v from backtest_results where metric='log_loss' and segment='all' "
        "and role='test' and model in ('elo', 'window', 'ml') group by model order by v limit 1").fetchone()
    assert backtest.best_model(fixture_db) == out[0]


def test_ml_base_added_with_custom_feature(fixture_db, feats):
    add_feature(fixture_db, "elo_edge", "elo_prob - 0.5", use_in_model=True)
    out = backtest.run(fixture_db, test_seasons=("2025-26",), n_boot=50)
    assert "ml_base" in set(out.model)
    add_feature(fixture_db, "elo_edge", "elo_prob - 0.5", use_in_model=False)
    assert "ml_base" not in set(backtest.run(fixture_db, test_seasons=("2025-26",), n_boot=50).model)


def test_ece():
    p = np.array([0.05, 0.15, 0.95, 0.85])
    y = np.array([0, 0, 1, 1])
    assert backtest.ece(p, y) == pytest.approx(np.mean([0.05, 0.15, 0.05, 0.15]))


def test_segments_ci_and_market_gap(fixture_db, feats):
    out = backtest.run(fixture_db, test_seasons=("2025-26",), n_boot=300)
    w = out[(out.model == "window") & (out.role == "test")]
    allw = w[w.segment == "all"].set_index("metric").value
    for k in ("log_loss", "margin_mae", "mae_gap"):
        assert allw[f"{k}_ci_lo"] <= allw[k] <= allw[f"{k}_ci_hi"], k
    # gap = model margin MAE - market MAE (market predicts margin = -home_spread)
    s2 = feats[(feats.season == "2025-26") & feats.home_win.notna()]
    mkt_mae = (s2.margin + s2.m_home_spread).abs().mean()
    mk = out[(out.model == "market") & (out.segment == "all")].set_index("metric").value
    assert mk["market_margin_mae"] == pytest.approx(mkt_mae)
    assert allw["mae_gap"] == pytest.approx(allw["margin_mae"] - mkt_mae)
    # early: either team < 15 season games -> every fixture game (12 per team) is early
    n = w.set_index(["segment", "metric"]).value
    assert n[("early", "n")] == n[("all", "n")] and ("rest", "n") not in n.index


def test_scores_games_with_nan_windows(fixture_db, feats):
    # blank a window column for some test games: they must still be scored
    fixture_db.execute("update game_features set h_all_l30_net = NULL")
    fixture_db.commit()
    out = backtest.run(fixture_db, test_seasons=("2025-26",), n_boot=50)
    s2 = feats[(feats.season == "2025-26") & feats.home_win.notna()]
    n = out[(out.model == "ml") & (out.metric == "n") & (out.segment == "all")].value.iloc[0]
    assert n == len(s2)


def test_holdout_reported_never_fitted_or_selected(fixture_db, feats, monkeypatch):
    fit_max_dates = []
    real_fit = ml_mod.MLModel.fit
    monkeypatch.setattr(ml_mod.MLModel, "fit",
                        lambda self, df: fit_max_dates.append(df.game_date.max()) or real_fit(self, df))
    out = backtest.run(fixture_db, test_seasons=(), holdout=("2025-26",), n_boot=50)
    assert set(out.role) == {"holdout"} and set(out.season) == {"2025-26"}
    assert len(fit_max_dates) == 1 and fit_max_dates[0] < "2025-10-21"  # one frozen fit, pre-season
    assert backtest.best_model(fixture_db) == "window"  # holdout rows never select a model


def test_backtest_stores_latest_flags_and_predictions(fixture_db, feats):
    out = backtest.run(fixture_db, test_seasons=("2025-26",), n_boot=20)
    s2 = feats[(feats.season == "2025-26") & feats.home_win.notna()]
    models = set(out[out.model != "market"].model)
    preds = fixture_db.execute("select model, count(*), count(distinct run_at) from backtest_predictions "
                               "group by model").fetchall()
    assert {r[0] for r in preds} == models and all(r[1] == len(s2) and r[2] == 1 for r in preds)
    nf = out[(out.metric == "n_flags") & (out.segment == "all")].set_index("model").value
    got = dict(fixture_db.execute("select model, count(*) from backtest_flags group by model").fetchall())
    assert {m: float(got.get(m, 0)) for m in nf.index} == nf.to_dict()
    row = dict(fixture_db.execute("select * from backtest_flags limit 1").fetchone())
    assert {"edge", "model_prob", "market_prob", "result", "units", "market", "side", "game_date"} <= set(row)
    backtest.run(fixture_db, test_seasons=("2025-26",), n_boot=20)  # replaces, never accumulates
    assert fixture_db.execute("select count(distinct run_at) from backtest_flags").fetchone()[0] == 1
    assert fixture_db.execute("select count(*) from backtest_predictions").fetchone()[0] == len(s2) * len(models)
