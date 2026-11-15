import numpy as np
import pandas as pd
import pytest

from nba_predict.analysis.metrics import METRICS, aggregate, add_game_metrics

EXPECTED_KEYS = {"ortg", "drtg", "net", "pace", "efg", "tov_pct", "orb_pct", "ftr", "margin",
                 "pts", "opp_pts", "win_pct", "record", "ats_record", "ats_pct", "ou_record",
                 "over_pct", "cover_margin", "total_pts"}


def _two_games():
    base = dict(min=240, fgm=40, fg3m=10, fta=20, oreb=10, tov=12, opp_fga=90, opp_fta=20,
                opp_oreb=10, opp_tov=12, opp_dreb=30, dreb=30)
    g1 = {**base, "pts": 120, "opp_pts": 100, "fga": 80, "wl": "W"}
    g2 = {**base, "pts": 90, "opp_pts": 110, "fga": 100, "wl": "L"}
    return add_game_metrics(pd.DataFrame([g1, g2]))


def test_registry_keys_and_kinds():
    assert EXPECTED_KEYS <= set(METRICS)
    assert {m.kind for m in METRICS.values()} <= {"rating", "pct", "mean", "record"}


def test_sum_based_rating():
    df = _two_games()
    out = aggregate(df, ["ortg", "drtg", "net"]).iloc[0]
    exp_ortg = 100 * df.pts.sum() / df.game_poss.sum()
    assert out["ortg"] == pytest.approx(exp_ortg)
    assert out["ortg"] != pytest.approx(df.ortg.mean())  # not a mean of game ratios
    assert out["net"] == pytest.approx(out["ortg"] - out["drtg"])
    assert out["gp"] == 2


def test_records_and_group_by():
    df = _two_games().assign(team_abbr=["BOS", "NYK"], ats=["cover", None],
                             ou=["over", "under"], cover_margin=[3.0, np.nan])
    out = aggregate(df, ["record", "ats_record", "ats_pct", "ou_record", "over_pct", "cover_margin"])
    r = out.iloc[0]
    assert (r.record, r.ats_record, r.ou_record) == ("1-1", "1-0-0", "1-1-0")
    assert r.ats_pct == 1.0 and r.over_pct == 0.5 and r.cover_margin == 3.0
    g = aggregate(df, ["win_pct"], group_by=["team_abbr"]).set_index("team_abbr")
    assert g.loc["BOS", "win_pct"] == 1.0 and g.loc["NYK", "gp"] == 1


def test_raw_aggs():
    out = aggregate(_two_games(), ["pts"], raw_aggs={"fga": "median", "tov": "sum"}).iloc[0]
    assert out["fga_median"] == 90 and out["tov_sum"] == 24 and out["pts"] == 105


def test_empty_frame():
    empty = _two_games().iloc[0:0]
    out = aggregate(empty, ["net", "record", "ats_pct"])
    assert len(out) == 1 and out.iloc[0]["gp"] == 0 and out.iloc[0]["record"] == "0-0"
    assert np.isnan(out.iloc[0]["net"])
    assert aggregate(empty.assign(team_abbr=[]), ["net"], group_by=["team_abbr"]).empty


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        aggregate(_two_games(), ["nope"])


def test_pace_uses_game_minutes_when_one_side_is_wrong():
    # real glitch: nba_api had DEN MIN=168 vs NOP 240 for 0022600231
    df = _two_games().assign(opp_min=[240, 240])
    df.loc[0, "min"] = 168
    df = add_game_metrics(df)
    assert df.loc[0, "game_min"] == 240 and df.loc[0, "pace"] == pytest.approx(df.loc[0, "game_poss"])
    assert aggregate(df, ["pace"]).iloc[0]["pace"] == pytest.approx(df.game_poss.mean())
