import numpy as np
import pytest

from nba_predict.features.build import build_feature_table
from nba_predict.models.ml import MLModel, default_feature_cols, eligible


@pytest.fixture
def feats(fixture_db, enriched):
    return build_feature_table(fixture_db)


def test_ml_never_uses_market_cols(feats):
    cols = default_feature_cols(feats.assign(m_extra=1.0, my_custom=2.0), custom_cols=["my_custom"])
    assert not [c for c in cols if c.startswith("m_")]
    assert "elo_prob" in cols and {"h_rest", "a_rest", "h_b2b", "a_b2b", "my_custom"} <= set(cols)
    assert all(c.startswith("d_") for c in cols if c not in
               {"elo_prob", "elo_margin", "h_rest", "a_rest", "h_b2b", "a_b2b", "h_opener", "a_opener",
                "my_custom"})
    assert "margin" not in cols and "home_win" not in cols


def test_ml_handles_nan(feats):
    train = feats[feats.home_win.notna()]
    m = MLModel(default_feature_cols(feats)).fit(train)
    test = feats.copy()
    test.loc[:, [c for c in test.columns if c.startswith("d_all_l15")]] = np.nan
    out = m.predict(test)
    assert list(out.columns) == ["game_id", "home_win_prob", "pred_margin", "pred_total"]
    assert np.isfinite(out[["home_win_prob", "pred_margin", "pred_total"]].to_numpy()).all()
    assert out.home_win_prob.between(0, 1).all() and len(out) == len(feats)


def test_eligible_excludes_short_history_without_prior_season(feats):
    el = eligible(feats)
    s1_first = feats[feats.game_id == "0022400000"].index[0]   # first game ever: no history
    assert not el[s1_first]
    s2 = feats[(feats.season == "2025-26") & feats.home_win.notna()]
    assert el[s2.index].all()  # early season-2 games have last season's data


# --- C phase-3: tuned regularization, out-of-fold sigma, curated features -----------------------

import pandas as pd

from nba_predict.models.ml import ALPHA_GRID, C_GRID, curated_feature_cols


def _synthetic(n=600, n_noise=60, seed=0):
    """Date-ordered games: margin = 0.8*x + 2 + noise, total = 2*pace_sum - 180 + noise, plus many
    pure-noise columns (so an unregularized fit overfits in-sample)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 5, n)
    h_pace, a_pace = rng.normal(100, 2, n), rng.normal(100, 2, n)
    margin = 0.8 * x + 2 + rng.normal(0, 12, n)
    total = 2 * (h_pace + a_pace) - 180 + rng.normal(0, 15, n)
    df = pd.DataFrame({"game_id": [f"g{i}" for i in range(n)],
                       "game_date": pd.date_range("2024-10-22", periods=n, freq="8h").strftime("%Y-%m-%d"),
                       "d_all_szn_adjnet": x, "h_all_szn_pace": h_pace, "a_all_szn_pace": a_pace,
                       "margin": margin, "total": total, "home_win": (margin > 0).astype(float)})
    for j in range(n_noise):
        df[f"d_noise_{j}"] = rng.normal(0, 1, n)
    return df


def test_ml_tunes_regularization_on_time_ordered_folds():
    df = _synthetic()
    cols = ["d_all_szn_adjnet"] + [f"d_noise_{j}" for j in range(60)]
    m = MLModel(cols).fit(df.sample(frac=1, random_state=1))  # shuffled input: fit must re-sort by date
    assert m.C in C_GRID and m.alpha in ALPHA_GRID
    assert m.cv_folds and all(tr_max < va_min for tr_max, va_min in m.cv_folds)  # forward-chaining
    assert m.alpha >= 10  # 60 noise columns: needs real shrinkage


def test_ml_sigma_is_out_of_fold_not_in_sample():
    df = _synthetic()
    cols = ["d_all_szn_adjnet"] + [f"d_noise_{j}" for j in range(60)]
    m = MLModel(cols).fit(df)
    fitted = m.predict(df)
    in_sample = float(np.sqrt(np.mean((df.margin - fitted.pred_margin) ** 2)))
    assert m.sigma > in_sample  # in-sample RMS understates error -> overconfident spread probs
    assert 11 < m.sigma < 14 and 14 < m.sigma_total < 18  # true noise sd: 12 and 15


def test_fixed_params_skip_tuning():
    df = _synthetic(n=200, n_noise=5)
    m = MLModel(["d_all_szn_adjnet"], C=0.5, alpha=3.0).fit(df)
    assert (m.C, m.alpha) == (0.5, 3.0) and np.isfinite(m.sigma)


def test_total_uses_both_teams_pace_not_only_differences():
    df = _synthetic()
    m = MLModel(["d_all_szn_adjnet"]).fit(df.iloc[:450])
    te = df.iloc[450:]
    mae = np.abs(m.predict(te).pred_total.to_numpy() - te.total.to_numpy()).mean()
    const = (df.iloc[:450].total.mean() - te.total).abs().mean()
    assert mae < const - 0.3  # pace sum is used for totals even though only a d_* col was given


def test_curated_feature_set(feats):
    cols = curated_feature_cols(feats.assign(my_custom=1.0), custom_cols=["my_custom"])
    assert 20 <= len(cols) <= 50 and len(set(cols)) == len(cols)
    assert not [c for c in cols if c.startswith("m_")]
    assert {"elo_prob", "h_rest", "a_rest", "my_custom"} <= set(cols)
    assert all(c in feats.columns or c == "my_custom" for c in cols)
