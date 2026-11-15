import pytest
from scipy.stats import norm

from nba_predict.db import upsert
from nba_predict.flags import evaluate_markets, market_view, settle


def test_evaluate_markets_values():
    out = evaluate_markets(dict(home_win_prob=0.6, pred_margin=4.0, pred_total=230.0),
                           dict(ml_novig=(0.55, 0.45), home_spread=-3.0, spread_novig=(0.5, 0.5),
                                total=226.0, total_novig=(0.5, 0.5)), sigma=12.5, sigma_total=18.0)
    ml_home = next(o for o in out if o["market"] == "ml" and o["side"] == "home")
    assert ml_home["edge"] == pytest.approx(0.05)
    sp_home = next(o for o in out if o["market"] == "spread" and o["side"] == "home")
    assert sp_home["model_prob"] == pytest.approx(norm.cdf(1 / 12.5))
    over = next(o for o in out if o["side"] == "over")
    assert over["model_prob"] == pytest.approx(norm.cdf(4 / 18))
    sp_away = next(o for o in out if o["market"] == "spread" and o["side"] == "away")
    assert sp_away["line"] == 3.0 and sp_away["model_prob"] == pytest.approx(1 - norm.cdf(1 / 12.5))
    assert ml_home["price"] is None and ml_home["book"] is None and len(out) == 6


def test_evaluate_markets_skips_missing():
    out = evaluate_markets(dict(home_win_prob=0.6, pred_margin=4.0, pred_total=float("nan")),
                           dict(ml_novig=None, home_spread=-3.0, spread_novig=(0.5, 0.5),
                                total=226.0, total_novig=(0.5, 0.5)), 12.5, 18.0)
    assert {o["market"] for o in out} == {"spread"}


def _snap(conn, rows):
    upsert(conn, "odds_snapshots", rows, ["game_id", "source", "book", "pulled_at"])


def test_market_view_missing_prices(conn):
    _snap(conn, [dict(game_id="g", source="espn", book="DK", pulled_at="2025-11-01T23:59:59Z",
                      is_closing=1, home_ml=-150, away_ml=130, home_spread=-3.5, total=221.5,
                      over_price=-110, under_price=-110)])
    mv = market_view(conn, "g")
    assert mv["home_spread"] is None and mv["spread_novig"] is None and mv["spread_price"] is None
    assert mv["ml_novig"][0] == pytest.approx(0.6 / (0.6 + 100 / 230)) and mv["ml_price"] == (-150, 130)
    assert mv["total"] == 221.5 and mv["total_novig"] == pytest.approx((0.5, 0.5))
    assert market_view(conn, "nope") is None


def test_market_view_prefers_latest_oddsapi_and_closing(conn):
    rows = [
        dict(game_id="g", source="espn", book="DK", pulled_at="2025-11-01T23:59:59Z", is_closing=1,
             home_ml=-150, away_ml=130, home_spread=-3.5, home_spread_price=-110, away_spread_price=-110,
             total=221.5, over_price=-110, under_price=-110),
        # morning pull (two books), then a closing pull (one book)
        dict(game_id="g", source="oddsapi", book="a", pulled_at="2025-11-01T14:00:00Z", is_closing=0,
             home_ml=-200, away_ml=170, home_spread=-5.0, home_spread_price=-105, away_spread_price=-115,
             total=220.0, over_price=-110, under_price=-110),
        dict(game_id="g", source="oddsapi", book="b", pulled_at="2025-11-01T14:00:00Z", is_closing=0,
             home_ml=-180, away_ml=155, home_spread=-4.0, home_spread_price=-110, away_spread_price=-110,
             total=222.0, over_price=-115, under_price=-105),
        dict(game_id="g", source="oddsapi", book="a", pulled_at="2025-11-01T23:00:00Z", is_closing=1,
             home_ml=-120, away_ml=100, home_spread=-2.0, home_spread_price=-110, away_spread_price=-110,
             total=219.0, over_price=-110, under_price=-110),
    ]
    _snap(conn, rows)
    latest = market_view(conn, "g")
    assert latest["home_spread"] == -2.0 and latest["ml_price"] == (-120, 100)
    _snap(conn, [{**rows[3], "pulled_at": "2025-11-02T01:00:00Z", "is_closing": 0, "home_spread": -1.0}])
    assert market_view(conn, "g")["home_spread"] == -1.0          # latest pull of any kind
    assert market_view(conn, "g", closing_only=True)["home_spread"] == -2.0  # latest closing pull


def test_market_view_consensus_and_best_price(conn):
    rows = [dict(game_id="g", source="oddsapi", book=b, pulled_at="t", is_closing=0, home_ml=h, away_ml=a,
                 home_spread=sp, home_spread_price=-110, away_spread_price=-110, total=t,
                 over_price=-110, under_price=-110)
            for b, h, a, sp, t in (("a", -110, -110, -1.0, 220.0), ("b", -120, 100, -2.0, 221.0),
                                   ("c", -105, -115, -3.0, 225.0))]
    _snap(conn, rows)
    mv = market_view(conn, "g")
    assert mv["ml_price"] == (-105, 100) and mv["ml_book"] == ("c", "b")
    assert mv["home_spread"] == -2.0 and mv["total"] == 221.0


@pytest.mark.parametrize("market,side,line,hp,ap,res", [
    ("ml", "home", None, 101, 100, "win"), ("ml", "away", None, 101, 100, "loss"),
    ("spread", "home", -5.0, 105, 100, "push"), ("spread", "home", -5.0, 107, 100, "win"),
    ("spread", "away", 5.0, 104, 100, "win"), ("spread", "away", 5.0, 106, 100, "loss"),
    ("total", "over", 200.5, 101, 100, "win"), ("total", "under", 201.0, 101, 100, "push"),
    ("total", "under", 202.0, 101, 100, "win"),
])
def test_settle(market, side, line, hp, ap, res):
    assert settle(market, side, line, hp, ap) == res
