import json
from datetime import date

import pytest

from nba_predict.db import upsert
from nba_predict.flags import flag_values, grade
from nba_predict.predict import predict_date

D = "2025-11-10"
G = "0022599000"  # fixture: scheduled BOS (home) vs NYK on 2025-11-10


def _pred(conn, p=0.5, margin=0.0, total=None, gid=G, model="window"):
    upsert(conn, "predictions", [dict(game_id=gid, model=model, created_at="t", home_win_prob=p,
                                      pred_margin=margin, pred_total=total,
                                      params_json=json.dumps({"sigma": 12.5, "sigma_total": 18.0}))],
           ["game_id", "model"])


def _snap(conn, gid=G, **kw):
    base = dict(game_id=gid, source="oddsapi", book="dk", pulled_at="2025-11-10T15:00:00Z", is_closing=0)
    upsert(conn, "odds_snapshots", [{**base, **kw}], ["game_id", "source", "book", "pulled_at"])


def _flags(conn):
    return [dict(r) for r in conn.execute("select * from value_flags order by market, side")]


def test_flag_threshold_and_one_side(fixture_db):
    _snap(fixture_db, home_ml=-110, away_ml=-110)  # ML only: no-vig 0.5 / 0.5
    _pred(fixture_db, p=0.529)
    assert flag_values(fixture_db, date(2025, 11, 10), model="window").empty
    assert _flags(fixture_db) == []
    _pred(fixture_db, p=0.531)
    out = flag_values(fixture_db, date(2025, 11, 10), model="window")
    f = _flags(fixture_db)
    assert len(out) == 1 and len(f) == 1 and (f[0]["market"], f[0]["side"]) == ("ml", "home")
    assert f[0]["edge"] == pytest.approx(0.031) and f[0]["price"] == -110 and f[0]["book"] == "dk"
    _pred(fixture_db, p=0.45)  # re-run with the other side now ahead: never both sides
    flag_values(fixture_db, date(2025, 11, 10), model="window")
    f = _flags(fixture_db)
    assert [(x["market"], x["side"]) for x in f] == [("ml", "away")]
    assert not {c for c in f[0] if c in ("stake", "kelly", "bankroll")}


def test_no_odds_no_flag(fixture_db):
    out = predict_date(fixture_db, date(2025, 11, 10))
    assert set(out.model) == {"elo", "window", "ml"} and len(out) == 6
    n = fixture_db.execute("select count(*) from predictions where game_id in "
                           "(select game_id from games where game_date = ?)", (D,)).fetchone()[0]
    assert n == 6
    assert flag_values(fixture_db, date(2025, 11, 10), model="window").empty
    assert _flags(fixture_db) == []


def test_predict_params_and_no_leak(fixture_db):
    predict_date(fixture_db, date(2025, 11, 10), models=("window",))
    pj = json.loads(fixture_db.execute(
        "select params_json from predictions where model='window' limit 1").fetchone()[0])
    assert {"sigma", "sigma_total", "hca", "effective_hca", "weights", "train_games", "train_max_date"} <= set(pj)
    assert pj["train_max_date"] < D
    predict_date(fixture_db, date(2025, 11, 10), models=("window",))  # re-predict replaces
    assert fixture_db.execute("select count(*) from predictions where model='window'").fetchone()[0] == 2


def _final_game(conn, gid="gx", home_pts=105, away_pts=100, status="final"):
    upsert(conn, "games", [dict(game_id=gid, season="2025-26", game_date="2025-11-12", home_team_id=1610612738,
                                away_team_id=1610612752, home_pts=home_pts, away_pts=away_pts, status=status)],
           ["game_id"])


def _flag(conn, gid="gx", **kw):
    row = dict(game_id=gid, created_at="t", model="window", market="spread", side="home", line=-5.0,
               price=-110, book="dk", model_prob=0.55, market_prob=0.5, edge=0.05)
    upsert(conn, "value_flags", [{**row, **kw}], ["game_id", "market", "side"])


def test_grade_spread_push_units(conn):
    _final_game(conn, home_pts=105, away_pts=100)
    _flag(conn)
    assert grade(conn) == 1
    f = _flags(conn)[0]
    assert f["result"] == "push" and f["units"] == 0


def test_grade_win_loss_units(conn):
    _final_game(conn, home_pts=110, away_pts=100)
    _flag(conn)
    _flag(conn, market="total", side="under", line=205.5, price=105)
    grade(conn)
    by = {x["market"]: x for x in _flags(conn)}
    assert by["spread"]["result"] == "win" and by["spread"]["units"] == pytest.approx(100 / 110)
    assert by["total"]["result"] == "loss" and by["total"]["units"] == -1


@pytest.mark.parametrize("status,pts", [("scheduled", (None, None)), ("postponed", (None, None))])
def test_grade_skips_unfinished(conn, status, pts):
    _final_game(conn, home_pts=pts[0], away_pts=pts[1], status=status)
    _flag(conn)
    assert grade(conn) == 0
    assert _flags(conn)[0]["result"] is None


def test_clv_ml(conn):
    _final_game(conn)
    _flag(conn, market="ml", side="away", line=None, price=150)
    # closing no-vig away = 100/238 = 0.42017 (prices -138 / +138)
    _snap(conn, gid="gx", pulled_at="2025-11-12T23:00:00Z", is_closing=1, home_ml=-138, away_ml=138)
    grade(conn)
    f = _flags(conn)[0]
    assert f["clv"] == pytest.approx(0.02, abs=1e-3) and f["closing_price"] == 138


def test_clv_spread_and_total(conn):
    _final_game(conn)
    _flag(conn, market="spread", side="away", line=5.0, price=-110)
    _flag(conn, market="total", side="over", line=220.0, price=-110)
    _snap(conn, gid="gx", pulled_at="2025-11-12T23:00:00Z", is_closing=1, home_spread=-3.0,
          home_spread_price=-110, away_spread_price=-110, total=223.0, over_price=-110, under_price=-110)
    grade(conn)
    by = {x["market"]: x for x in _flags(conn)}
    assert by["spread"]["closing_line"] == 3.0 and by["spread"]["clv"] == pytest.approx(2.0)  # took +5, closed +3
    assert by["total"]["closing_line"] == 223.0 and by["total"]["clv"] == pytest.approx(3.0)


def test_grade_without_closing_leaves_clv_null(conn):
    _final_game(conn)
    _flag(conn)
    grade(conn)
    f = _flags(conn)[0]
    assert f["result"] == "push" and f["clv"] is None and f["closing_line"] is None
