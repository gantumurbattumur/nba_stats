"""Dedicated total-points regression and over probability model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_predict.config import model_scope
from nba_predict.models.ml import model_feature_cols


class TotalsModel:
    """Use the curated pregame features to estimate total and its uncertainty."""
    def __init__(self):
        self.model = None

    def fit(self, frame: pd.DataFrame, *, all_seasons: bool = False):
        cols = model_feature_cols(frame)
        d = model_scope(frame, all_seasons=all_seasons)
        d = d[d.total.notna()].sort_values("game_date", kind="stable")
        self.cols = cols
        self.fill = {c: 0.0 for c in cols if d[c].isna().all()}
        X = d.reindex(columns=cols).astype(float).fillna(self.fill)
        self.total = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
        self.total.fit(X, d.total.astype(float))
        self.sigma_total = float(np.sqrt(np.mean((self.total.predict(X) - d.total.to_numpy()) ** 2)))
        return self

    def predict(self, frame: pd.DataFrame, closing_total=None) -> pd.DataFrame:
        if not hasattr(self, "total"):
            raise RuntimeError("fit must be called before predict")
        X = frame.reindex(columns=self.cols).astype(float).fillna(self.fill)
        base = pd.DataFrame({"game_id": frame.game_id.to_numpy(), "pred_total": self.total.predict(X)})
        sigma = max(float(self.sigma_total), 1.0)
        base["p_over"] = (norm.cdf((base.pred_total - np.asarray(closing_total)) / sigma)
                          if closing_total is not None else np.nan)
        return base[["game_id", "pred_total", "p_over"]]
