"""Agent C phase-3 review: contract checks for backtest/flags/odds/ESPN-open handling."""

import json
import logging
from datetime import date

import numpy as np
import pytest
import responses

from conftest import load_fixture
from nba_predict import backtest
from nba_predict.data.odds_client import URL, OddsApiClient
from nba_predict.db import upsert
from nba_predict.flags import flag_values, market_view
from nba_predict.models import window_rating as wr

G = "0022599000"  # fixture_db: scheduled BOS (home) vs NYK on 2025-11-10
D = date(2025, 11, 10)
HDR0 = {"x-requests-used": "500", "x-requests-remaining": "0", "x-requests-last": "3"}


def _bt_rows(conn, rows):
    conn.executemany("insert into backtest_results (run_at, model, season, metric, value, segment, role) "
                     "values ('2026-11-15T00:00:00', ?, '2025-26', 'log_loss', ?, 'all', 'test')", rows)
    conn.commit()


# --- best_model must name a model that predict_date produces ---------------------------------

@pytest.mark.parametrize("winner", ["home_always", "window_noprior", "ml_base"])
def test_best_model_only_returns_predictable_models(conn, winner):
    _bt_rows(conn, [(winner, 0.50), ("window", 0.60), ("ml", 0.65), ("elo", 0.62), ("market", 0.40)])
    assert backtest.best_model(conn) == "window"


# --- effective HCA counts every home-edge source ---------------------------------------------

from nba_predict.models.elo import EloParams

HOME_ELO_PTS = EloParams().home_adv / 28  # even teams' elo_margin


def _even_rows(n=40, gp=2.0, elo_margin=HOME_ELO_PTS):
    """Identical teams: every window equal for both sides; only home sources differ."""
    row = {"margin": 2.0, "total": 220.0, "h_gp_szn": gp, "a_gp_szn": gp, "elo_margin": elo_margin,
           "h_b2b": 0, "a_b2b": 0, "h_rest": 1, "a_rest": 1}
    for side in ("h", "a"):
        for w in ("l5", "l10", "l15", "l20", "l30", "szn"):
            for sc, net in (("all", 0.0), ("split", 3.0 if side == "h" else -3.0)):
                row.update({f"{side}_{sc}_{w}_net": net, f"{side}_{sc}_{w}_ortg": 112.0,
                            f"{side}_{sc}_{w}_drtg": 112.0 - net, f"{side}_{sc}_{w}_pace": 100.0})
    import pandas as pd
    return pd.DataFrame([{**row, "game_id": f"g{i}"} for i in range(n)])


def test_effective_hca_without_prior_is_hca_plus_split_edge():
    df = _even_rows()
    p = wr.WindowParams(hca=1.0, split_weight=0.25, elo_prior_games=0.0)
    # split adds 0.25 * (3 - (-3)) * 100/100 = 1.5 points
    assert backtest.effective_hca(df, p) == pytest.approx(2.5)


def test_effective_hca_counts_the_elo_prior_home_edge():
    # with a heavy prior (k >> gp) the prediction is ~ elo_margin = home_adv/28 for even teams,
    # so the model's home edge is that, not ~hca + split * (1 - w)
    df = _even_rows(gp=1.0)
    p = wr.WindowParams(hca=1.0, split_weight=0.25, elo_prior_games=1000.0)
    eff = backtest.effective_hca(df, p)
    assert eff == pytest.approx(HOME_ELO_PTS, abs=0.05)
    pred = wr.predict_rows(df, p).pred_margin.mean()
    assert eff == pytest.approx(pred, abs=1e-6)  # even teams: whole predicted margin is home edge


# --- Odds API: 200 with 0 remaining still returns the data -----------------------------------

@responses.activate
def test_last_credit_pull_returns_data_with_warning(caplog):
    responses.get(URL, json=load_fixture("oddsapi_nba.json"), status=200, headers=HDR0)
    with caplog.at_level(logging.WARNING):
        events, usage = OddsApiClient("k").nba_odds()
    assert len(events) > 0 and usage == {"used": 500, "remaining": 0, "last": 3}
    assert any("0 credits remaining" in r.getMessage() for r in caplog.records)


# --- flag_values contract --------------------------------------------------------------------

def _pred(conn, gid=G, p=0.6, margin=0.0, total=None, sigma=12.5, model="window"):
    upsert(conn, "predictions", [dict(game_id=gid, model=model, created_at="t", home_win_prob=p,
                                      pred_margin=margin, pred_total=total,
                                      params_json=json.dumps({"sigma": sigma, "sigma_total": 18.0}))],
           ["game_id", "model"])


def _snap(conn, gid=G, **kw):
    base = dict(game_id=gid, source="oddsapi", book="dk", pulled_at="2025-11-10T15:00:00Z", is_closing=0)
    upsert(conn, "odds_snapshots", [{**base, **kw}], ["game_id", "source", "book", "pulled_at"])


def test_only_scheduled_games_are_flagged(fixture_db):
    final_gid = fixture_db.execute("select game_id from games where status='final' limit 1").fetchone()[0]
    fixture_db.execute("update games set game_date = ? where game_id = ?", (D.isoformat(), final_gid))
    for gid in (G, final_gid):
        _snap(fixture_db, gid=gid, home_ml=-110, away_ml=-110)
        _pred(fixture_db, gid=gid, p=0.7)
    out = flag_values(fixture_db, D, model="window")
    assert set(out.game_id) == {G}


def test_settled_flags_never_touched(fixture_db):
    _snap(fixture_db, home_ml=-110, away_ml=-110)
    _pred(fixture_db, p=0.7)
    flag_values(fixture_db, D, model="window")
    fixture_db.execute("update value_flags set result='win', units=0.909 where game_id = ?", (G,))
    fixture_db.commit()
    before = [tuple(r) for r in fixture_db.execute("select * from value_flags")]
    _pred(fixture_db, p=0.2)  # now the other side would be flagged
    flag_values(fixture_db, D, model="window")
    rows = [tuple(r) for r in fixture_db.execute("select * from value_flags where result is not null")]
    assert rows == before
    sides = [r[0] for r in fixture_db.execute("select side from value_flags where market='ml'")]
    assert sides == ["home"]  # settled home kept; no away flag next to it (never both sides)


def test_spread_flag_uses_sigma_from_params_json(fixture_db):
    # pred margin 3 vs line -1.5 (home): P(cover) = Phi((3 - 1.5)/sigma); no-vig 0.5
    _snap(fixture_db, home_spread=-1.5, home_spread_price=-110, away_spread_price=-110)
    _pred(fixture_db, p=None, margin=3.0, sigma=12.5)
    e1 = flag_values(fixture_db, D, model="window").edge.iloc[0]
    _pred(fixture_db, p=None, margin=3.0, sigma=5.0)
    e2 = flag_values(fixture_db, D, model="window").edge.iloc[0]
    from scipy.stats import norm
    assert e1 == pytest.approx(norm.cdf(1.5 / 12.5) - 0.5) and e2 == pytest.approx(norm.cdf(1.5 / 5) - 0.5)


# --- ESPN open lines never act as closing ----------------------------------------------------

def test_closing_only_never_picks_an_open_row(conn):
    upsert(conn, "games", [dict(game_id="g", season="2025-26", game_date="2025-11-10", home_team_id=1,
                                away_team_id=2, status="final", home_pts=100, away_pts=90)], ["game_id"])
    upsert(conn, "odds_snapshots", [dict(game_id="g", source="espn", book="ESPN BET",
                                         pulled_at="2025-11-10T12:00:00Z", is_closing=0, home_spread=-3.0,
                                         home_spread_price=-110, away_spread_price=-110)],
           ["game_id", "source", "book", "pulled_at"])
    assert market_view(conn, "g", closing_only=True) is None
    assert market_view(conn, "g")["home_spread"] == -3.0


def test_open_rows_do_not_change_enriched_fingerprint(fixture_db):
    from nba_predict.analysis.enrich import source_fingerprint
    from nba_predict.features.build import features_fingerprint

    fp, ffp = source_fingerprint(fixture_db), features_fingerprint(fixture_db, (5,))
    gid = fixture_db.execute("select game_id from odds_snapshots limit 1").fetchone()[0]
    upsert(fixture_db, "odds_snapshots", [dict(game_id=gid, source="espn", book="ESPN BET",
                                               pulled_at="2025-10-29T12:00:00Z", is_closing=0,
                                               home_spread=-9.5, total=250.0)],
           ["game_id", "source", "book", "pulled_at"])
    assert source_fingerprint(fixture_db) == fp and features_fingerprint(fixture_db, (5,)) == ffp
