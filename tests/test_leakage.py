"""Stronger leakage checks than ``test_features.test_no_leakage``.

That test adds +40 pts to *both* teams of every game on/after D, which leaves margins, W/L,
net ratings, SOS and ATS unchanged, so a same-day leak in any of those went unnoticed (verified
by flipping ``allow_exact_matches`` in ``enrich._opp_strength``). Here results on/after D are
flipped asymmetrically and later games are deleted outright, and the feature table is rebuilt
*without* an explicit ``build_enriched`` call (a stale ``team_games_enriched`` must not be used).
"""

import pandas as pd
import pytest

from nba_predict.features.build import build_feature_table

D = "2025-10-29"   # mid-season-2 date in fixture_db
D2 = "2025-11-03"  # later date: games after it are deleted


def _feature_cols(df):
    return [c for c in df.columns if c.startswith(("h_", "a_", "d_"))]


def _mutate_from(conn, d):
    # home side +40 pts on/after d: flips W/L, margins, net, ATS/O-U for those games
    conn.execute("update team_games set pts = pts + 40, fgm = fgm + 20 "
                 "where game_date >= ? and is_home = 1", (d,))
    conn.execute("update games set home_pts = home_pts + 40 where game_date >= ?", (d,))
    # and drop every game strictly after D2 (incl. the 2 scheduled ones)
    conn.execute("delete from team_games where game_date > ?", (D2,))
    conn.execute("delete from games where game_date > ?", (D2,))
    conn.commit()


def test_no_leakage_asymmetric_change_and_removal(fixture_db):
    base = build_feature_table(fixture_db).set_index("game_id")
    assert D in set(base.game_date)
    _mutate_from(fixture_db, D)
    after = build_feature_table(fixture_db).set_index("game_id")  # must notice the change itself
    keep = base.index[base.game_date <= D]
    cols = _feature_cols(base)
    pd.testing.assert_frame_equal(base.loc[keep, cols], after.loc[keep, cols])
    # not vacuous: games after D do see the flipped results (so the rebuild really happened)
    mid = base.index[(base.game_date > D) & (base.game_date <= D2)]
    assert len(mid) and not base.loc[mid, "d_all_l5_net"].equals(after.loc[mid, "d_all_l5_net"])
    assert len(after) == (base.game_date <= D2).sum()


def test_opp_strength_is_strictly_pre_game(fixture_db):
    from nba_predict.analysis.enrich import build_enriched

    e0 = build_enriched(fixture_db).set_index(["game_id", "team_id"])
    _mutate_from(fixture_db, D)
    e1 = build_enriched(fixture_db).set_index(["game_id", "team_id"])
    keep = e1.index[e1.game_date <= D]
    for c in ("opp_win_pct_pre", "opp_net_pre", "opp_net_rank_pre", "rest_days", "b2b"):
        pd.testing.assert_series_equal(e0.loc[keep, c], e1.loc[keep, c], check_dtype=False)


def test_two_games_same_date_do_not_see_each_other(fixture_db):
    # not possible in a real NBA schedule, but a data glitch must not leak: BOS plays a 2nd
    # game on D; both BOS rows on D get identical pre-game state, and the next game sees both
    from nba_predict.db import upsert

    bos, lal = 1610612738, 1610612747
    gid = "0022509999"
    upsert(fixture_db, "games", [dict(game_id=gid, season="2025-26", game_date=D,
                                      home_team_id=bos, away_team_id=lal, home_pts=150,
                                      away_pts=80, status="final", season_type="regular")],
           ["game_id"])
    box = dict(min=240, fgm=60, fga=90, fg3m=10, fg3a=30, ftm=20, fta=25, oreb=10, dreb=40,
               reb=50, ast=30, stl=8, blk=5, tov=10, pf=18)
    upsert(fixture_db, "team_games", [
        dict(game_id=gid, team_id=bos, opp_team_id=lal, season="2025-26", game_date=D, is_home=1,
             wl="W", plus_minus=70.0, season_type="regular", **{**box, "pts": 150}),
        dict(game_id=gid, team_id=lal, opp_team_id=bos, season="2025-26", game_date=D, is_home=0,
             wl="L", plus_minus=-70.0, season_type="regular", **{**box, "pts": 80, "fgm": 30}),
    ], ["game_id", "team_id"])
    f = build_feature_table(fixture_db)
    on_d = f[(f.game_date == D) & ((f.home_team_id == bos) | (f.away_team_id == bos))]
    assert len(on_d) == 2
    vals = [r[f"{'h' if r.home_team_id == bos else 'a'}_all_l5_net"] for _, r in on_d.iterrows()]
    gps = [r[f"{'h' if r.home_team_id == bos else 'a'}_gp_szn"] for _, r in on_d.iterrows()]
    assert vals[0] == pytest.approx(vals[1]) and gps[0] == gps[1]
    nxt = f[(f.game_date > D) & ((f.home_team_id == bos) | (f.away_team_id == bos))].iloc[0]
    side = "h" if nxt.home_team_id == bos else "a"
    assert nxt[f"{side}_gp_szn"] == gps[0] + 2
