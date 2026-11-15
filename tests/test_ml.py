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
