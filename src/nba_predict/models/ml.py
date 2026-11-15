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


def _pipe(est) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("est", est)])


class MLModel:
    def __init__(self, feature_cols: list[str], C: float = 1.0, alpha: float = 1.0):
        bad = [c for c in feature_cols if c.startswith("m_")]
        if bad:
            raise ValueError(f"market columns are never model inputs: {bad}")
        self.feature_cols = list(feature_cols)
        self.C, self.alpha = C, alpha

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=self.feature_cols).astype(float)
        return X.replace([np.inf, -np.inf], np.nan)

    def fit(self, df: pd.DataFrame) -> "MLModel":
        d = df[df["home_win"].notna()]
        X = self._X(d)
        # columns that are all-NaN in training would be dropped by the imputer; fill with 0
        self._fill = {c: 0.0 for c in X.columns if X[c].isna().all()}
        X = X.fillna(self._fill)
        self.win = _pipe(LogisticRegression(C=self.C, max_iter=2000)).fit(X, d["home_win"].astype(int))
        self.margin = _pipe(Ridge(alpha=self.alpha)).fit(X, d["margin"].astype(float))
        t = d["total"].notna()
        self.total = _pipe(Ridge(alpha=self.alpha)).fit(X[t], d.loc[t, "total"].astype(float))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._X(df).fillna(self._fill)
        return pd.DataFrame({
            "game_id": df["game_id"].to_numpy(),
            "home_win_prob": self.win.predict_proba(X)[:, 1],
            "pred_margin": self.margin.predict(X),
            "pred_total": self.total.predict(X),
        })
