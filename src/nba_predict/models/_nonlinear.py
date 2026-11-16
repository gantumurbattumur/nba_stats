"""Shared temporal-scope adapters for optional nonlinear model families."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_predict.config import get_settings
from nba_predict.models.ml import model_feature_cols


class NonlinearModel:
    estimator = None
    model_name = "nonlinear"

    def __init__(self, history="last_season"):
        if history not in ("last_season", "long_history"):
            raise ValueError("history must be last_season or long_history")
        self.history = history

    def fit(self, frame: pd.DataFrame):
        d = frame[frame.home_win.notna()].copy()
        if self.history == "last_season":
            d = d[d.season.isin(get_settings().model_seasons)]
        else:
            d = d[d.season.astype(str).str[:4].astype(int) >= 2018]
        d = d.sort_values("game_date", kind="stable")
        self.cols = model_feature_cols(d)
        self.fill = {c: 0.0 for c in self.cols if d[c].isna().all()}
        X = d[self.cols].astype(float).fillna(self.fill)
        self.win = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), self.estimator())
        self.win.fit(X, d.home_win.astype(int))
        # Residual total regression shares the same curated feature view.
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.total = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=7, l2_regularization=2.0, random_state=0))
        self.total.fit(X[d.total.notna()], d.loc[d.total.notna(), "total"])
        self.sigma_total = float(np.sqrt(np.mean((self.total.predict(X[d.total.notna()]) - d.loc[d.total.notna(), "total"])**2)))
        return self

    def predict(self, frame):
        X = frame.reindex(columns=self.cols).astype(float).fillna(self.fill)
        return pd.DataFrame({"game_id": frame.game_id.to_numpy(),
                             "home_win_prob": self.win.predict_proba(X)[:, 1],
                             "pred_total": self.total.predict(X)})
