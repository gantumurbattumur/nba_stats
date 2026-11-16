"""Walk-forward backtest of all models (fit on games before each test month, score the month).

``backtest_results`` rows: ``run_at, model, season, metric, value, segment, role``.
- segment: ``all`` | ``early`` (either team < 15 season games) | ``rest``
- role: ``test`` (walk-forward) | ``holdout`` (one frozen pre-season fit; reported only)
Metrics per (model, season, segment): n, accuracy, brier, log_loss (+ ``_ci_lo``/``_ci_hi``,
95% game bootstrap), margin_mae (+ CI), total_mae, ece; on games with a closing line also
log_loss_market_games, mae_gap (model − market margin MAE, + CI), n_spread_games and a flagged
side evaluation at the spec rule (edge >= edge_threshold, one side per market, a hypothetical
1 unit at the closing price): bets, hit_rate, roi. Model ``market``: n, market_log_loss,
market_margin_mae. Window models (segment all) add hca, effective_hca, split_weight,
elo_prior_games. These are model-evaluation metrics only — nothing is staked.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3

import numpy as np
import pandas as pd

from nba_predict.analysis.custom import with_custom
from nba_predict.config import get_settings
from nba_predict.features.build import load_features
from nba_predict.flags import evaluate_markets, market_view, pick_flags, settle
from nba_predict.models import ml as ml_mod
from nba_predict.models import window_rating as wr
from nba_predict.models.elo import EloParams
from nba_predict.odds_math import unit_return

log = logging.getLogger(__name__)
CORE_MODELS = ("home_always", "elo", "window", "ml")
PREDICTABLE = ("elo", "window", "ml")  # what predict_date produces; best_model picks among these
EARLY_GAMES = 15  # segment 'early' = either team has fewer season games than this


def _rms(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error with ``bins`` equal-width probability bins."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def _log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _db_path(conn: sqlite3.Connection) -> str:
    return conn.execute("pragma database_list").fetchone()[2]


def _custom_model_cols(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("select name from custom_features where use_in_model = 1")]


def effective_hca(train: pd.DataFrame, params: wr.WindowParams) -> float:
    """Points of home edge the fitted model adds on average over ``train``: mean of
    pred_margin(params) − pred_margin(neutral), where *neutral* removes every home-edge source:
    ``hca = 0``, ``split_weight = 0`` (home/road split ratings carry home court) and the Elo
    prior's built-in home advantage (``elo_margin − home_adv/28``). Without the Elo prior this
    is ``hca`` + the split blend's edge; with it, the prior's share is counted too."""
    a = wr.predict_rows(train, params).pred_margin.to_numpy()
    neutral = train
    if "elo_margin" in train:
        neutral = train.assign(elo_margin=train["elo_margin"] - EloParams().home_adv / 28.0)
    b = wr.predict_rows(neutral, params.model_copy(update={"hca": 0.0, "split_weight": 0.0})).pred_margin.to_numpy()
    return float(np.nanmean(a - b))


def _elo_margin(df: pd.DataFrame) -> np.ndarray:
    if "elo_margin" in df:
        return df["elo_margin"].to_numpy(dtype=float)
    return ((df.h_elo - df.a_elo + EloParams().home_adv) / 28.0).to_numpy(dtype=float)


def _fit_predict(name: str, train: pd.DataFrame, test: pd.DataFrame, feature_cols, custom_cols):
    """→ (pred DataFrame, sigma, sigma_total, extra metrics dict)."""
    extra: dict[str, float] = {}
    if name == "home_always":
        p, mg, tt = train.home_win.mean(), train.margin.mean(), train.total.mean()
        pred = pd.DataFrame({"game_id": test.game_id.to_numpy(), "home_win_prob": p,
                             "pred_margin": mg, "pred_total": tt})
        return pred, _rms(train.margin - mg), _rms(train.total - tt), extra
    if name == "elo":
        pred = pd.DataFrame({"game_id": test.game_id.to_numpy(), "home_win_prob": test.elo_prob.to_numpy(),
                             "pred_margin": _elo_margin(test), "pred_total": np.nan})
        return pred, _rms(train.margin.to_numpy() - _elo_margin(train)), float("nan"), extra
    if name.startswith("window"):
        base = wr.WindowParams()
        if name == "window_noprior":
            base = base.model_copy(update={"elo_prior_games": 0.0, "fit_elo_prior": False})
        params = wr.fit(train, base)
        extra = {"hca": params.hca, "effective_hca": effective_hca(train, params),
                 "split_weight": params.split_weight, "elo_prior_games": params.elo_prior_games}
        return wr.predict_rows(test, params), params.sigma, params.sigma_total, extra
    cols = feature_cols if name == "ml" else [c for c in feature_cols if c not in custom_cols]
    m = ml_mod.MLModel(cols).fit(train)  # C/alpha tuned on forward date folds inside ``train``
    extra = {"C": m.C, "alpha": m.alpha, "alpha_total": m.alpha_total}
    return m.predict(test), m.sigma, m.sigma_total, extra  # out-of-fold sigmas


def _flag_eval(conn, rows: pd.DataFrame, threshold: float, mv_cache: dict) -> dict[str, float]:
    """rows: game_id, home_pts, away_pts, home_win_prob, pred_margin, pred_total, sigma, sigma_total."""
    units, wins, losses = [], 0, 0
    for r in rows.itertuples(index=False):
        if r.game_id not in mv_cache:
            mv_cache[r.game_id] = market_view(conn, r.game_id, closing_only=True)
        mv = mv_cache[r.game_id]
        if mv is None:
            continue
        pred = dict(home_win_prob=r.home_win_prob, pred_margin=r.pred_margin, pred_total=r.pred_total)
        for fl in pick_flags(evaluate_markets(pred, mv, r.sigma, r.sigma_total), threshold):
            if fl["price"] is None:
                continue
            res = settle(fl["market"], fl["side"], fl["line"], r.home_pts, r.away_pts)
            units.append(unit_return(fl["price"], res))
            wins += res == "win"
            losses += res == "loss"
    n = len(units)
    return dict(bets=n, hit_rate=wins / (wins + losses) if wins + losses else float("nan"),
                roi=float(np.sum(units) / n) if n else float("nan"))


def _boot_ci(per_game: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean, resampling games."""
    x = per_game[np.isfinite(per_game)]
    if len(x) < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _ll_per_game(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _block(d: pd.DataFrame, conn, threshold: float, mv_cache: dict, rng, n_boot: int) -> dict[str, float]:
    """All metrics for one (model, season, segment) on merged truth+prediction rows ``d``."""
    y, p = d.home_win.to_numpy(dtype=float), d.home_win_prob.to_numpy(dtype=float)
    ll = _ll_per_game(p, y)
    ae = np.abs(d.margin.to_numpy(dtype=float) - d.pred_margin.to_numpy(dtype=float))
    met = dict(n=len(d), accuracy=float(((p > 0.5) == (y == 1)).mean()),
               brier=float(np.mean((p - y) ** 2)), log_loss=float(ll.mean()),
               margin_mae=float(np.nanmean(ae)),
               total_mae=(float(np.nanmean(np.abs(d.total - d.pred_total)))
                          if d.pred_total.notna().any() else float("nan")),
               ece=ece(p, y))
    met["log_loss_ci_lo"], met["log_loss_ci_hi"] = _boot_ci(ll, rng, n_boot)
    met["margin_mae_ci_lo"], met["margin_mae_ci_hi"] = _boot_ci(ae, rng, n_boot)
    mk = d.m_home_novig.notna().to_numpy()
    if mk.any():
        met["log_loss_market_games"] = float(ll[mk].mean())
        met.update(_flag_eval(conn, d[mk], threshold, mv_cache))
    sp = d.m_home_spread.notna().to_numpy()
    if sp.any():
        gap = ae[sp] - np.abs(d.margin.to_numpy(dtype=float)[sp] + d.m_home_spread.to_numpy(dtype=float)[sp])
        met["n_spread_games"] = int(sp.sum())
        met["mae_gap"] = float(np.mean(gap))  # model margin MAE − market (−spread) MAE
        met["mae_gap_ci_lo"], met["mae_gap_ci_hi"] = _boot_ci(gap, rng, n_boot)
    return met


def _market_block(t: pd.DataFrame) -> dict[str, float]:
    met: dict[str, float] = {}
    mk = t[t.m_home_novig.notna()]
    if len(mk):
        met.update(n=float(len(mk)), market_log_loss=_log_loss(mk.m_home_novig, mk.home_win))
    sp = t[t.m_home_spread.notna()]
    if len(sp):
        met.update(n_spread_games=float(len(sp)),
                   market_margin_mae=float((sp.margin + sp.m_home_spread).abs().mean()))
    return met


def _segments(t: pd.DataFrame):
    early = ((t.h_gp_szn < EARLY_GAMES) | (t.a_gp_szn < EARLY_GAMES)).to_numpy()
    return (("all", np.ones(len(t), bool)), ("early", early), ("rest", ~early))


def run(conn: sqlite3.Connection, test_seasons=("2024-25", "2025-26"), refit: str = "monthly",
        windows=None, holdout=(), n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Walk-forward over ``test_seasons`` (fit on all games before each month). ``holdout``
    seasons are scored with one frozen fit on games before the season starts; they are
    reported (``role='holdout'``) but never used by ``best_model``. Every metric is also
    broken out by ``segment`` early/rest. Returns long rows
    ``model, season, metric, value, segment, role`` and appends them to ``backtest_results``."""
    if refit != "monthly":
        raise ValueError("only refit='monthly' is supported")
    s = get_settings()
    rng = np.random.default_rng(seed)
    f = load_features(conn, windows or s.windows)
    custom = _custom_model_cols(conn)
    if custom:
        f = with_custom(f, _db_path(conn), only_model=True)
    feature_cols = ml_mod.model_feature_cols(f, custom_cols=custom)
    games = pd.read_sql_query("select game_id, home_pts, away_pts from games", conn)
    final = f[(f.status == "final") & f.home_win.notna()]
    trainable = final[ml_mod.eligible(final)]  # training only; every final test game is scored
    models = list(CORE_MODELS) + ["window_noprior"] + (["ml_base"] if custom else [])
    mv_cache: dict = {}
    out: list[tuple] = []
    plan = [(sz, "test") for sz in test_seasons] + [(sz, "holdout") for sz in holdout]
    for season, role in plan:
        sf = final[final.season == season]
        if sf.empty:
            continue
        if role == "holdout":
            chunks = [(sf.game_date.min(), sf)]
        else:
            chunks = [(f"{mo}-01", sf[sf.game_date.str[:7] == mo]) for mo in sorted(sf.game_date.str[:7].unique())]
        preds = {m: [] for m in models}
        extras = {m: [] for m in models}
        for start, test in chunks:
            train = trainable[trainable.game_date < start]
            if len(train) < 10 or train.home_win.nunique() < 2:
                log.warning("backtest %s from %s: too little training data (%d games)", season, start, len(train))
                continue
            for m in models:
                pred, sig, sig_t, extra = _fit_predict(m, train, test, feature_cols, custom)
                preds[m].append(pred.assign(sigma=sig, sigma_total=sig_t))
                extras[m].append(extra)
        truth = sf[["game_id", "home_win", "margin", "total", "m_home_novig", "m_home_spread",
                    "h_gp_szn", "a_gp_szn"]].merge(games, on="game_id")
        for seg, mask in _segments(truth):
            if mask.any():
                out += [("market", season, k, float(v), seg, role)
                        for k, v in _market_block(truth[mask]).items()]
        for m in models:
            if not preds[m]:
                continue
            d = truth.merge(pd.concat(preds[m], ignore_index=True), on="game_id")
            for seg, mask in _segments(d):
                if not mask.any():
                    continue
                met = _block(d[mask], conn, s.edge_threshold, mv_cache, rng, n_boot)
                if seg == "all":
                    for k in sorted({k for e in extras[m] for k in e}):
                        met[k] = float(np.mean([e[k] for e in extras[m] if k in e]))
                out += [(m, season, k, float(v), seg, role) for k, v in met.items()]
    res = pd.DataFrame(out, columns=["model", "season", "metric", "value", "segment", "role"])
    run_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    conn.executemany(
        "insert into backtest_results (run_at, model, season, metric, value, segment, role) "
        "values (?,?,?,?,?,?,?)", [(run_at, *r) for r in out])
    conn.commit()
    return res


def best_model(conn: sqlite3.Connection) -> str:
    """Model with the lowest log_loss averaged over test seasons (segment 'all') in the latest
    run that has test rows; holdout rows never count. Only models ``predict_date`` produces
    (``PREDICTABLE``) are eligible — baselines and ablations (home_always, window_noprior,
    ml_base) are reported but never chosen, else ``flag_values`` would find no predictions.
    'window' if none."""
    ph = ", ".join("?" for _ in PREDICTABLE)
    row = conn.execute(
        "select model from backtest_results where run_at = (select max(run_at) from backtest_results "
        "where role = 'test') and role = 'test' and segment = 'all' and metric = 'log_loss' "
        f"and model in ({ph}) group by model order by avg(value) limit 1", PREDICTABLE
    ).fetchone()
    return row[0] if row else "window"
