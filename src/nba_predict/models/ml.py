"""ML model: logistic regression (home win) + ridge (margin, total) on home-minus-away
feature differences, Elo and schedule. Market columns (``m_*``) are never inputs."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_EXTRA = ["elo_prob", "elo_margin", "h_rest", "a_rest", "h_b2b", "a_b2b"]
_L_GP = re.compile(r"^([ha])_all_l(\d+)_gp$")


def default_feature_cols(df: pd.DataFrame, custom_cols: list[str] | None = None) -> list[str]:
    """All ``d_*`` columns + Elo + rest/b2b (+ opener flags if present) + given custom columns."""
    extra = BASE_EXTRA + ["h_opener", "a_opener"]
    cols = [c for c in df.columns if c.startswith("d_")]
    cols += [c for c in extra if c in df.columns]
    cols += [c for c in (custom_cols or []) if c in df.columns and c not in cols]
    return [c for c in cols if not c.startswith("m_")]


def eligible(df: pd.DataFrame) -> pd.Series:
    """False for games where either team has < 3 season games *and* no prior-season data
    (longest l-window has no more games than the season count). Used to exclude rows from
    training/scoring; such games are still predicted."""
    ok = pd.Series(True, index=df.index)
    for side in ("h", "a"):
        lcols = [c for c in df.columns if (m := _L_GP.match(c)) and m.group(1) == side]
        longest = df[lcols].max(axis=1) if lcols else pd.Series(0, index=df.index)
        short = df[f"{side}_gp_szn"] < 3
        ok &= ~(short & (longest <= df[f"{side}_gp_szn"]))
    return ok


# regularization grids searched by time-ordered inner CV inside every training window
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
ALPHA_GRID = (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
N_SPLITS = 3
MIN_TUNE_ROWS = 120  # below this: no inner CV (defaults, in-sample sigma)
_TOTAL_WINDOWS = ("l10", "l20", "l30", "szn")


def curated_feature_cols(df: pd.DataFrame, custom_cols: list[str] | None = None) -> list[str]:
    """About 40 hand-picked inputs instead of every ``d_*`` column: opponent-adjusted net at
    short/long/season windows (all + split), four factors at L10/season, L20 off/def ratings,
    Elo, schedule and games played (``t_*`` pace/points sums for totals are added in ``_X``)."""
    want = [f"d_all_{w}_adjnet" for w in ("l5", "l10", "l20", "l30", "szn")]
    want += [f"d_split_{w}_adjnet" for w in ("l10", "l20", "szn")]
    want += [f"d_all_{w}_{m}" for w in ("l10", "szn") for m in ("efg", "tov_pct", "orb_pct", "ftr")]
    want += ["d_all_l20_ortg", "d_all_l20_drtg", "d_all_l10_win_pct", "d_all_szn_win_pct"]
    want += ["elo_prob", "elo_margin", "h_rest", "a_rest", "h_b2b", "a_b2b", "h_3in4", "a_3in4",
             "h_opener", "a_opener", "h_gp_szn", "a_gp_szn"]
    cols = [c for c in want if c in df.columns]
    cols += [c for c in (custom_cols or []) if c in df.columns and c not in cols]
    return [c for c in cols if not c.startswith("m_")]


FEATURE_SET = "curated"  # chosen by the 2024-25/2025-26 walk-forward backtest (C, phase 3)


def model_feature_cols(df: pd.DataFrame, custom_cols: list[str] | None = None) -> list[str]:
    """The ML model's inputs: ``FEATURE_SET`` ('curated' | 'all')."""
    fn = curated_feature_cols if FEATURE_SET == "curated" else default_feature_cols
    return fn(df, custom_cols=custom_cols)


def _pipe(est) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("est", est)])


def _date_folds(dates: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining folds on whole dates: train on date blocks 0..k, validate on block k+1
    (no date is split across train/validation)."""
    uniq = np.unique(dates)
    if len(uniq) < n_splits + 1:
        return []
    blocks = np.array_split(uniq, n_splits + 1)
    out = []
    for k in range(n_splits):
        tr = np.isin(dates, np.concatenate(blocks[:k + 1]))
        va = np.isin(dates, blocks[k + 1])
        out.append((np.flatnonzero(tr), np.flatnonzero(va)))
    return out


class MLModel:
    """Logistic (win) + ridge (margin, total). ``C``/``alpha`` = None → tuned per fit over
    ``C_GRID``/``ALPHA_GRID`` by forward-chaining date folds inside the training data (log loss /
    MSE). ``sigma``/``sigma_total`` are the RMS of the *out-of-fold* residuals at the chosen
    regularization, so spread/total probabilities aren't overconfident."""

    def __init__(self, feature_cols: list[str], C: float | None = None, alpha: float | None = None,
                 alpha_total: float | None = None, n_splits: int = N_SPLITS):
        bad = [c for c in feature_cols if c.startswith("m_")]
        if bad:
            raise ValueError(f"market columns are never model inputs: {bad}")
        self.feature_cols = list(feature_cols)
        self.C, self.alpha = C, alpha
        self.alpha_total = alpha_total if alpha_total is not None else alpha
        self.n_splits = n_splits
        self.cv_folds: list[tuple[str, str]] = []

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=self.feature_cols).astype(float)
        # totals depend on both teams' pace/scoring (sums), which h−a differences can't express
        for w in _TOTAL_WINDOWS:
            hp, ap = f"h_all_{w}_pace", f"a_all_{w}_pace"
            if hp in df and ap in df:
                X[f"t_{w}_pace"] = df[hp].astype(float) + df[ap].astype(float)
                rt = [f"{s}_all_{w}_{m}" for s in ("h", "a") for m in ("ortg", "drtg")]
                if all(c in df for c in rt):
                    pts = (df[rt[0]] + df[rt[3]] + df[rt[2]] + df[rt[1]]).astype(float) / 2
                    X[f"t_{w}_pts"] = pts * X[f"t_{w}_pace"] / 200
        return X.replace([np.inf, -np.inf], np.nan)

    def _tune(self, make, grid, X, y, folds, score):
        """→ (best grid value, out-of-fold residuals/probs at it)."""
        best = None
        for v in grid:
            oof = np.full(len(y), np.nan)
            for tr, va in folds:
                if score == "logloss" and len(np.unique(y[tr])) < 2:
                    continue
                m = _pipe(make(v)).fit(X.iloc[tr], y[tr])
                oof[va] = m.predict_proba(X.iloc[va])[:, 1] if score == "logloss" else m.predict(X.iloc[va])
            ok = np.isfinite(oof)
            if not ok.any():
                continue
            if score == "logloss":
                p = np.clip(oof[ok], 1e-15, 1 - 1e-15)
                loss = float(-np.mean(y[ok] * np.log(p) + (1 - y[ok]) * np.log(1 - p)))
            else:
                loss = float(np.mean((y[ok] - oof[ok]) ** 2))
            if best is None or loss < best[0]:
                best = (loss, v, oof)
        return (best[1], best[2]) if best else (None, None)

    def fit(self, df: pd.DataFrame) -> "MLModel":
        d = df[df["home_win"].notna()]
        if "game_date" in d:
            d = d.sort_values("game_date", kind="stable")
        X = self._X(d)
        # columns that are all-NaN in training would be dropped by the imputer; fill with 0
        self._fill = {c: 0.0 for c in X.columns if X[c].isna().all()}
        X = X.fillna(self._fill)
        yw = d["home_win"].astype(int).to_numpy()
        ym = d["margin"].astype(float).to_numpy()
        t = d["total"].notna().to_numpy()
        yt = d["total"].astype(float).to_numpy()
        dates = d["game_date"].to_numpy() if "game_date" in d else np.arange(len(d))
        folds = _date_folds(dates, self.n_splits) if len(d) >= MIN_TUNE_ROWS else []
        self.cv_folds = [(str(dates[tr].max()), str(dates[va].min())) for tr, va in folds]
        tfolds = [(np.intersect1d(tr, np.flatnonzero(t)), np.intersect1d(va, np.flatnonzero(t)))
                  for tr, va in folds]
        Xt = X  # same row order; total rows selected through tfolds / mask

        def logit(c):
            return LogisticRegression(C=c, max_iter=2000)

        oof_m = oof_t = None
        if folds:
            self.C, _ = self._tune(logit, (self.C,) if self.C else C_GRID, X, yw, folds, "logloss")
            self.alpha, oof_m = self._tune(Ridge, (self.alpha,) if self.alpha else ALPHA_GRID, X, ym, folds, "mse")
            self.alpha_total, oof_t = self._tune(Ridge, (self.alpha_total,) if self.alpha_total else ALPHA_GRID,
                                                 Xt, np.where(t, yt, np.nan), tfolds, "mse")
        self.C = self.C or 0.1
        self.alpha = self.alpha or 100.0
        self.alpha_total = self.alpha_total or self.alpha
        self.win = _pipe(logit(self.C)).fit(X, yw)
        self.margin = _pipe(Ridge(alpha=self.alpha)).fit(X, ym)
        self.total = _pipe(Ridge(alpha=self.alpha_total)).fit(X[t], yt[t])

        def rms(r):
            r = r[np.isfinite(r)]
            return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")

        if oof_m is not None:
            self.sigma, self.sigma_total = rms(ym - oof_m), rms(yt - oof_t) if oof_t is not None else float("nan")
        else:  # too little data for inner CV: in-sample (optimistic) fallback
            self.sigma = rms(ym - self.margin.predict(X))
            self.sigma_total = rms(yt[t] - self.total.predict(X[t]))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._X(df).fillna(self._fill)
        return pd.DataFrame({
            "game_id": df["game_id"].to_numpy(),
            "home_win_prob": self.win.predict_proba(X)[:, 1],
            "pred_margin": self.margin.predict(X),
            "pred_total": self.total.predict(X),
        })
