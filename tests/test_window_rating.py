import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nba_predict.models.window_rating import (
    WindowParams,
    fit,
    predict_profiles,
    predict_rows,
    team_rating,
)

WINS = ("l5", "l10", "l15", "szn")
MET = ("net", "adjnet", "ortg", "drtg", "pace")


def _row(h=None, a=None, **extra):
    """One feature row; every window/scope gets the same values per side unless overridden."""
    h = {"net": 2.0, "adjnet": 2.0, "ortg": 115.0, "drtg": 113.0, "pace": 100.0, **(h or {})}
    a = {"net": 2.0, "adjnet": 2.0, "ortg": 115.0, "drtg": 113.0, "pace": 100.0, **(a or {})}
    r = {"game_id": "g"}
    for side, vals in (("h", h), ("a", a)):
        for scope in ("all", "split"):
            for w in WINS:
                for m in MET:
                    r[f"{side}_{scope}_{w}_{m}"] = vals[m]
    r.update(h_b2b=0, a_b2b=0, h_rest=1, a_rest=1)
    r.update(extra)
    return r


def test_weights_normalized_any_window():
    p = WindowParams(weights={"l3": 1, "szn": 3})
    assert p.weights == {"l3": 0.25, "szn": 0.75}
    assert sum(WindowParams().weights.values()) == pytest.approx(1)


def test_even_teams_hca_only():
    out = predict_rows(pd.DataFrame([_row()]), WindowParams()).iloc[0]
    assert out.pred_margin == pytest.approx(2.5)
    assert out.home_win_prob == pytest.approx(norm.cdf(2.5 / 12.5))
    assert out.pred_total == pytest.approx(228.0)  # pace 100: (115+113)/2 * 2


def test_b2b_penalty():
    base = predict_rows(pd.DataFrame([_row()]), WindowParams()).iloc[0].pred_margin
    b2b = predict_rows(pd.DataFrame([_row(h_b2b=1)]), WindowParams()).iloc[0].pred_margin
    assert base - b2b == pytest.approx(1.5)


def test_rest_capped():
    p = WindowParams()
    m = predict_rows(pd.DataFrame([_row(h_rest=7, a_rest=0)]), p).iloc[0].pred_margin
    assert m == pytest.approx(2.5 + 0.4 * 3)


def test_net_gap_scaled_by_pace():
    r = _row(h={"net": 10.0, "adjnet": 10.0, "pace": 96.0}, a={"net": 0.0, "adjnet": 0.0, "pace": 104.0})
    m = predict_rows(pd.DataFrame([r]), WindowParams()).iloc[0].pred_margin
    assert m == pytest.approx(10 * 100 / 100 + 2.5)


def test_split_blend_and_missing_window_renormalized():
    p = WindowParams(weights={"l5": 0.5, "szn": 0.5}, split_weight=0.3, use_sos=False)
    cols = {f"{s}_{w}_{m}": np.nan for s in ("all", "split") for w in ("l5", "szn") for m in MET}
    cols.update({"all_l5_net": 4.0, "split_l5_net": 10.0, "all_szn_net": np.nan, "split_szn_net": np.nan,
                 "all_l5_ortg": 110.0, "all_l5_drtg": 106.0, "all_l5_pace": 99.0})
    net, ortg, drtg, pace = team_rating(cols, p)
    assert net == pytest.approx(0.7 * 4 + 0.3 * 10)  # szn missing -> l5 weight 1
    assert (ortg, drtg, pace) == (110.0, 106.0, 99.0)  # split NaN -> all


def test_use_sos_falls_back_to_net_when_adjnet_missing():
    p = WindowParams(weights={"l5": 1.0}, split_weight=0.0)
    cols = {"all_l5_net": 3.0, "all_l5_adjnet": 5.0, "all_l5_ortg": 1, "all_l5_drtg": 1, "all_l5_pace": 1}
    assert team_rating(cols, p)[0] == 5.0
    assert team_rating({**cols, "all_l5_adjnet": np.nan}, p)[0] == 3.0
    assert team_rating(cols, p.model_copy(update={"use_sos": False}))[0] == 3.0


def test_predict_with_two_prior_games():
    r = _row()
    for k in list(r):
        if any(f"_{w}_" in k for w in ("l10", "l15")) or "_split_" in k:
            r[k] = np.nan
    out = predict_rows(pd.DataFrame([r]), WindowParams()).iloc[0]
    assert np.isfinite([out.home_win_prob, out.pred_margin, out.pred_total]).all()


def test_rows_match_profiles():
    rng = np.random.default_rng(0)
    r = _row()
    for k in r:
        if k.startswith(("h_", "a_")) and k.endswith(MET):
            r[k] = float(rng.normal(100 if k.endswith(("ortg", "drtg", "pace")) else 0, 3))
    r.update(h_b2b=1, h_rest=0, a_rest=2)
    p = WindowParams()
    row = predict_rows(pd.DataFrame([r]), p).iloc[0]

    def prof(side):
        return {f"{scope}_{w}": {m: r[f"{side}_{scope}_{w}_{m}"] for m in MET}
                for scope in ("all", "split") for w in WINS}

    out = predict_profiles(prof("h"), prof("a"), dict(rest=0, b2b=1), dict(rest=2, b2b=0), p)
    for k in ("home_win_prob", "pred_margin", "pred_total"):
        assert out[k] == pytest.approx(row[k])


def test_fit_recovers_hca():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(600):
        hn, an = rng.normal(0, 5, 2)
        r = _row(h={"net": hn, "adjnet": hn}, a={"net": an, "adjnet": an},
                 h_rest=int(rng.integers(0, 4)), a_rest=int(rng.integers(0, 4)))
        r["game_id"] = f"g{i}"
        r["margin"] = hn - an + 3.0 + rng.normal(0, 4)
        r["total"] = 228 + rng.normal(0, 15)
        rows.append(r)
    p = fit(pd.DataFrame(rows))
    assert p.hca == pytest.approx(3.0, abs=0.3)  # noise sd 4 on 600 rows: se ~0.16
    assert 3 < p.sigma < 5 and 10 < p.sigma_total < 20
    assert sum(p.weights.values()) == pytest.approx(1) and 0 <= p.split_weight <= 1


def test_fit_sigma_is_rms_not_std():
    """A model whose margin residuals are biased must not report a too-small sigma."""
    from nba_predict.models import window_rating as wr

    rows = []
    for i in range(50):
        r = _row()
        r["game_id"] = f"g{i}"
        r["margin"] = 10.0 + (1 if i % 2 else -1)   # pred is hca only; residual mean ~ (10 - hca)
        r["total"] = 228.0 + 6 + (1 if i % 2 else -1)  # pred 228 -> residual mean 6, sd 1
        rows.append(r)
    df = pd.DataFrame(rows)
    p = wr.fit(df, WindowParams(hca=0.0))
    pred = predict_rows(df, p)
    resid = df.margin - pred.pred_margin
    assert p.sigma == pytest.approx(float(np.sqrt(np.mean(resid ** 2))))
    assert p.sigma_total == pytest.approx(float(np.sqrt(np.mean((df.total - pred.pred_total) ** 2))))
    assert p.sigma_total > 5  # RMS ~6.08; std would say 1.0
