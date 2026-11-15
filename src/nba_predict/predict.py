"""Predict a date's games: each model is fit on all final games before the date."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3

import numpy as np
import pandas as pd

from nba_predict.analysis.custom import with_custom
from nba_predict.backtest import _custom_model_cols, _db_path, _elo_margin, _rms, effective_hca
from nba_predict.config import get_settings
from nba_predict.db import upsert
from nba_predict.features.build import load_features
from nba_predict.models import ml as ml_mod
from nba_predict.models import window_rating as wr


def _clean(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def predict_date(conn: sqlite3.Connection, date: dt.date, models=("elo", "window", "ml")) -> pd.DataFrame:
    """Fit each model on final games before ``date``, predict that date's (non-postponed) games and
    upsert ``predictions`` (one row per game × model; re-predict replaces). ``params_json`` holds
    the fitted params, sigma/sigma_total (used for spread/total probabilities) and training info."""
    s = get_settings()
    f = load_features(conn, s.windows)  # rebuilds when stale
    custom = _custom_model_cols(conn)
    if custom:
        f = with_custom(f, _db_path(conn), only_model=True)
    ds = date.isoformat()
    train = f[(f.status == "final") & f.home_win.notna() & (f.game_date < ds)]
    train = train[ml_mod.eligible(train)]
    target = f[(f.game_date == ds) & (f.status != "postponed")]
    if target.empty or len(train) < 10:
        return pd.DataFrame(columns=["game_id", "model", "home_win_prob", "pred_margin", "pred_total"])
    info = {"train_games": int(len(train)), "train_max_date": train.game_date.max()}
    outs = []
    for m in models:
        if m == "elo":
            pred = pd.DataFrame({"game_id": target.game_id.to_numpy(), "home_win_prob": target.elo_prob.to_numpy(),
                                 "pred_margin": _elo_margin(target), "pred_total": np.nan})
            params = {"sigma": _rms(train.margin.to_numpy() - _elo_margin(train)), "sigma_total": None}
        elif m == "window":
            p = wr.fit(train)
            pred = wr.predict_rows(target, p)
            params = {**p.model_dump(), "effective_hca": effective_hca(train, p)}
        elif m == "ml":
            cols = ml_mod.default_feature_cols(f, custom_cols=custom)
            model = ml_mod.MLModel(cols).fit(train)
            pred = model.predict(target)
            fitted = model.predict(train)
            params = {"sigma": _rms(train.margin.to_numpy() - fitted.pred_margin.to_numpy()),
                      "sigma_total": _rms(train.total.to_numpy() - fitted.pred_total.to_numpy()),
                      "C": model.C, "alpha": model.alpha, "features": cols}
        else:
            raise ValueError(f"unknown model {m!r}")
        pred = pred.assign(model=m)
        pj = json.dumps({**params, **info}, default=float)
        now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        upsert(conn, "predictions", [dict(game_id=r.game_id, model=m, created_at=now,
                                          home_win_prob=_clean(r.home_win_prob), pred_margin=_clean(r.pred_margin),
                                          pred_total=_clean(r.pred_total), params_json=pj)
                                     for r in pred.itertuples(index=False)], ["game_id", "model"])
        outs.append(pred)
    return pd.concat(outs, ignore_index=True)[["game_id", "model", "home_win_prob", "pred_margin", "pred_total"]]
