import pytest

from nba_predict.odds_math import (
    american_to_decimal,
    american_to_prob,
    best_price,
    consensus_novig,
    devig_two_way,
    unit_return,
)


def test_american_to_prob():
    assert american_to_prob(-110) == pytest.approx(110 / 210)
    assert american_to_prob(150) == pytest.approx(0.4)
    assert american_to_prob(-200) == pytest.approx(2 / 3)


def test_american_to_decimal():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal(100) == pytest.approx(2.0)


def test_devig():
    assert devig_two_way(110 / 210, 110 / 210) == pytest.approx((0.5, 0.5))
    a, b = devig_two_way(american_to_prob(-150), american_to_prob(130))
    # plan said a > 0.58; exact value is 0.6 / (0.6 + 100/230) = 0.57983
    assert a + b == pytest.approx(1.0) and a == pytest.approx(0.57983, abs=1e-4)


def test_unit_return():
    assert unit_return(-110, "win") == pytest.approx(100 / 110)
    assert unit_return(150, "win") == pytest.approx(1.5)
    assert unit_return(150, "loss") == -1 and unit_return(-110, "push") == 0


def test_unit_return_bad_result():
    with pytest.raises(ValueError):
        unit_return(-110, "void")


def test_consensus_and_best():
    assert consensus_novig([(-110, -110), (-120, 100)])[0] == pytest.approx(0.5112, abs=1e-3)
    assert consensus_novig([]) is None
    assert best_price([-115, -105, -110]) == -105 and best_price([120, 135]) == 135


def test_consensus_sums_to_one():
    a, b = consensus_novig([(-150, 130), (-140, 120), (-160, 135)])
    assert a + b == pytest.approx(1.0)
