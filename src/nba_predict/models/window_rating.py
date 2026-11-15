"""Window rating model: blended recent-form ratings → margin, win probability and total.

Per window: rating = (1 − split_weight)·all + split_weight·split (split NaN → all); ``net``
uses ``adjnet`` (= net + SOS) when ``use_sos`` (per window, falling back to ``net`` while SOS
is unknown). Windows are combined with ``weights``; missing windows are dropped and the
remaining weights renormalized. A team with no finite window gets net 0 (league average).

Optional Elo prior for early season (last-N windows still carry last season's rosters):
``pred_margin = w·elo_margin + (1 − w)·window_margin`` with ``w = k / (k + gp)``,
``k = elo_prior_games`` and ``gp`` = mean of both teams' season games played. ``k = 0`` (or a
missing ``elo_margin``) disables it; ``fit`` learns ``k`` unless ``fit_elo_prior=False``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from scipy.optimize import minimize
from scipy.stats import norm

RATING_METRICS = ("net", "ortg", "drtg", "pace")


class WindowParams(BaseModel):
    weights: dict[str, float] = {"l5": 0.10, "l10": 0.15, "l15": 0.20, "l20": 0.15, "l30": 0.10,
                                 "szn": 0.30}
    split_weight: float = 0.3
    use_sos: bool = True
    hca: float = 2.5
    b2b_pts: float = 1.5
    rest_pts: float = 0.4
    rest_cap: int = 3
    sigma: float = 12.5
    sigma_total: float = 18.0
    elo_prior_games: float = 0.0
    fit_elo_prior: bool = True

    @field_validator("weights")
    @classmethod
    def _normalize(cls, w: dict[str, float]) -> dict[str, float]:
        tot = sum(w.values())
        if not w or tot <= 0 or any(v < 0 for v in w.values()):
            raise ValueError("weights must be non-negative with a positive sum")
        return {k: v / tot for k, v in w.items()}


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype=float) if name in df else np.full(len(df), np.nan)


def _side_arrays(df: pd.DataFrame, side: str, windows, use_sos: bool) -> dict:
    """{metric: [(all, split) per window]} with the adjnet→net fallback applied."""
    out = {}
    for m in RATING_METRICS:
        pairs = []
        for w in windows:
            sc = []
            for scope in ("all", "split"):
                v = _col(df, f"{side}_{scope}_{w}_{m}")
                if m == "net" and use_sos:
                    adj = _col(df, f"{side}_{scope}_{w}_adjnet")
                    v = np.where(np.isfinite(adj), adj, v)
                sc.append(v)
            pairs.append(tuple(sc))
        out[m] = pairs
    return out


def _blend(pairs, wts, sw: float) -> np.ndarray:
    num = np.zeros_like(pairs[0][0])
    den = np.zeros_like(pairs[0][0])
    for (a, s), wt in zip(pairs, wts):
        v = np.where(np.isfinite(s), (1 - sw) * a + sw * s, a)
        ok = np.isfinite(v)
        num += wt * np.where(ok, v, 0.0)
        den += wt * ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _ratings(arrs: dict, wts, sw: float) -> dict[str, np.ndarray]:
    return {m: _blend(arrs[m], wts, sw) for m in RATING_METRICS}


def _sched(df: pd.DataFrame, side: str):
    return np.nan_to_num(_col(df, f"{side}_b2b")), np.nan_to_num(_col(df, f"{side}_rest"))


def _prior_weight(df: pd.DataFrame, k: float):
    """(w, elo_margin) arrays for the Elo prior; w = 0 where Elo is missing or k = 0."""
    elo = _col(df, "elo_margin")
    gp = np.nan_to_num((_col(df, "h_gp_szn") + _col(df, "a_gp_szn")) / 2, nan=0.0)
    w = np.where(np.isfinite(elo) & (k > 0), k / (k + gp) if k > 0 else 0.0, 0.0)
    return w, np.nan_to_num(elo)


def _predict(h: dict, a: dict, hb2b, ab2b, hrest, arest, p: WindowParams, prior=None):
    hp, ap = h["pace"], a["pace"]
    n = np.isfinite(hp).astype(float) + np.isfinite(ap)
    pace = (np.nan_to_num(hp) + np.nan_to_num(ap)) / np.where(n > 0, n, 1)
    pace = np.where(n > 0, pace, 100.0)  # mean of available paces; 100 if neither known
    net_gap = np.nan_to_num(h["net"]) - np.nan_to_num(a["net"])
    margin = (net_gap * pace / 100 + p.hca - p.b2b_pts * (hb2b - ab2b)
              + p.rest_pts * np.clip(hrest - arest, -p.rest_cap, p.rest_cap))
    if prior is not None:
        w, elo = prior
        margin = w * elo + (1 - w) * margin
    total = pace / 100 * ((h["ortg"] + a["drtg"]) / 2 + (a["ortg"] + h["drtg"]) / 2)
    return norm.cdf(margin / p.sigma), margin, total


def predict_rows(features: pd.DataFrame, params: WindowParams = WindowParams()) -> pd.DataFrame:
    p = params
    wins, wts = list(p.weights), list(p.weights.values())
    h = _ratings(_side_arrays(features, "h", wins, p.use_sos), wts, p.split_weight)
    a = _ratings(_side_arrays(features, "a", wins, p.use_sos), wts, p.split_weight)
    hb2b, hrest = _sched(features, "h")
    ab2b, arest = _sched(features, "a")
    prob, margin, total = _predict(h, a, hb2b, ab2b, hrest, arest, p,
                                   _prior_weight(features, p.elo_prior_games))
    return pd.DataFrame({"game_id": features["game_id"].to_numpy(), "home_win_prob": prob,
                         "pred_margin": margin, "pred_total": total})


def team_rating(prefix_cols: Mapping, params: WindowParams = WindowParams()) -> tuple[float, float, float, float]:
    """Ratings from one team's ``{scope}_{win}_{metric}`` mapping → (net, ortg, drtg, pace)."""
    df = pd.DataFrame([{f"x_{k}": v for k, v in prefix_cols.items()}])
    wins, wts = list(params.weights), list(params.weights.values())
    r = _ratings(_side_arrays(df, "x", wins, params.use_sos), wts, params.split_weight)
    return tuple(float(r[m][0]) for m in RATING_METRICS)


def predict_profiles(h: dict, a: dict, h_sched: dict, a_sched: dict,
                     params: WindowParams = WindowParams()) -> dict:
    """Playground path: ``h``/``a`` map ``"{scope}_{win}"`` → ``team_profile`` dict; ``*_sched``
    hold ``rest`` and ``b2b`` (optionally ``gp_szn``; ``h_sched["elo_margin"]`` enables the
    Elo prior). Same math as ``predict_rows``."""
    row = {"game_id": None}
    for side, prof in (("h", h), ("a", a)):
        for key, vals in prof.items():
            for m, v in vals.items():
                row[f"{side}_{key}_{m}"] = v
    for side, s in (("h", h_sched), ("a", a_sched)):
        row[f"{side}_rest"] = s.get("rest", 1)
        row[f"{side}_b2b"] = int(bool(s.get("b2b", 0)))
        row[f"{side}_gp_szn"] = s.get("gp_szn", np.nan)
    row["elo_margin"] = h_sched.get("elo_margin", np.nan)
    out = predict_rows(pd.DataFrame([row]), params).iloc[0]
    return {k: float(out[k]) for k in ("home_win_prob", "pred_margin", "pred_total")}


def fit(train: pd.DataFrame, base: WindowParams = WindowParams()) -> WindowParams:
    """Fit hca, b2b_pts, rest_pts, split_weight and window weights (softmax) on margin MSE;
    then sigma / sigma_total as the RMS of the residuals (not the std, which would hide a
    biased model's error)."""
    train = train[train["margin"].notna()]
    wins = list(base.weights)
    harr = _side_arrays(train, "h", wins, base.use_sos)
    aarr = _side_arrays(train, "a", wins, base.use_sos)
    hb2b, hrest = _sched(train, "h")
    ab2b, arest = _sched(train, "a")
    y = train["margin"].to_numpy(dtype=float)
    learn_prior = base.fit_elo_prior and np.isfinite(_col(train, "elo_margin")).any()
    nw = len(wins)

    def unpack(theta):
        z = np.exp(theta[4:4 + nw] - theta[4:4 + nw].max())
        k = float(np.log1p(np.exp(theta[-1]))) if learn_prior else base.elo_prior_games  # softplus
        return dict(hca=theta[0], b2b_pts=theta[1], rest_pts=theta[2],
                    split_weight=float(1 / (1 + np.exp(-theta[3]))), wts=z / z.sum(), k=k)

    def margin_for(u):
        h = _ratings(harr, u["wts"], u["split_weight"])
        a = _ratings(aarr, u["wts"], u["split_weight"])
        p = base.model_copy(update={k: u[k] for k in ("hca", "b2b_pts", "rest_pts")})
        return _predict(h, a, hb2b, ab2b, hrest, arest, p, _prior_weight(train, u["k"]))

    def loss(theta):
        _, m, _ = margin_for(unpack(theta))
        return float(np.mean((y - m) ** 2))

    sw0 = min(max(base.split_weight, 1e-3), 1 - 1e-3)
    theta0 = np.concatenate([[base.hca, base.b2b_pts, base.rest_pts, np.log(sw0 / (1 - sw0))],
                             np.log(np.maximum(list(base.weights.values()), 1e-6))])
    if learn_prior:
        k0 = max(base.elo_prior_games, 3.0)
        theta0 = np.append(theta0, np.log(np.expm1(k0)))
    res = minimize(loss, theta0, method="L-BFGS-B")
    u = unpack(res.x)
    _, m, tot = margin_for(u)
    resid_t = train["total"].to_numpy(dtype=float) - tot if "total" in train else np.array([np.nan])
    return base.model_copy(update=dict(
        hca=float(u["hca"]), b2b_pts=float(u["b2b_pts"]), rest_pts=float(u["rest_pts"]),
        split_weight=u["split_weight"], elo_prior_games=float(u["k"]),
        weights={w: float(v) for w, v in zip(wins, u["wts"])},
        sigma=float(np.sqrt(np.mean((y - m) ** 2))),
        sigma_total=(float(np.sqrt(np.nanmean(resid_t ** 2))) if np.isfinite(resid_t).any()
                     else base.sigma_total),
    ))
