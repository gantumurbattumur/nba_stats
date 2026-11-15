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
