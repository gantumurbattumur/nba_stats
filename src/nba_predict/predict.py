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


def _after_tip(start_iso, status, now: dt.datetime):
    """1 if the game had started (or finished) at ``now``, 0 if not, None if unknown."""
    if status == "final":
        return 1
    if not start_iso:
        return None
    start = dt.datetime.fromisoformat(str(start_iso).replace("Z", "+00:00"))
    return int(now >= start)


def predict_date(conn: sqlite3.Connection, date: dt.date, models=("elo", "window", "ml"),
                 now: dt.datetime | None = None) -> pd.DataFrame:
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
    now = now or dt.datetime.now(dt.UTC)
    tips = {r[0]: _after_tip(r[1], r[2], now) for r in conn.execute(
        "select game_id, start_time_utc, status from games where game_date = ?", (ds,))}
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
            cols = ml_mod.model_feature_cols(f, custom_cols=custom)
            model = ml_mod.MLModel(cols).fit(train)  # tuned C/alpha, out-of-fold sigmas
            pred = model.predict(target)
            params = {"sigma": model.sigma, "sigma_total": model.sigma_total, "C": model.C,
                      "alpha": model.alpha, "alpha_total": model.alpha_total, "features": cols}
        else:
            raise ValueError(f"unknown model {m!r}")
        pred = pred.assign(model=m)
        pj = json.dumps({**params, **info}, default=float)
        created = now.astimezone(dt.UTC).isoformat(timespec="seconds")
        upsert(conn, "predictions", [dict(game_id=r.game_id, model=m, created_at=created,
                                          home_win_prob=_clean(r.home_win_prob), pred_margin=_clean(r.pred_margin),
                                          pred_total=_clean(r.pred_total), params_json=pj,
                                          made_after_tip=tips.get(r.game_id))
                                     for r in pred.itertuples(index=False)], ["game_id", "model"])
        outs.append(pred)
    return pd.concat(outs, ignore_index=True)[["game_id", "model", "home_win_prob", "pred_margin", "pred_total"]]
