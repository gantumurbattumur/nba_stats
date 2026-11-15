import pytest

from nba_predict.analysis.enrich import build_enriched
from nba_predict.db import read_df, upsert

BOS = 1610612738


def _bos(enriched):
    return enriched[enriched.team_id == BOS].sort_values("game_date").reset_index(drop=True)


def test_shape_and_written(fixture_db, enriched):
    assert len(enriched) == 96  # 48 final games x 2
    assert read_df(fixture_db, "select count(*) n from team_games_enriched").n[0] == 96
    need = {"opp_pts", "opp_dreb", "team_abbr", "opp_abbr", "conference", "division",
            "opp_conference", "poss", "game_poss", "ortg", "drtg", "net", "pace", "efg",
            "tov_pct", "orb_pct", "ftr", "margin", "win", "rest_days", "b2b", "three_in_four",
            "team_spread", "total_line", "team_ml", "is_favorite", "cover_margin", "ats", "ou",
            "opp_win_pct_pre", "opp_net_pre", "opp_net_rank_pre", "month", "weekday",
            "season_type", "total_pts"}
    assert need <= set(enriched.columns)


def test_poss_and_ratings_hand_checked(enriched):
    g = _bos(enriched).iloc[0]  # see conftest.fixture_db docstring
    assert (g.game_id, g.team_abbr, g.opp_abbr, g.is_home) == ("0022400000", "BOS", "NYK", 1)
    assert g.poss == pytest.approx(94.48) and g.opp_poss == pytest.approx(100.68)
    assert g.game_poss == pytest.approx(97.58)
    assert g.ortg == pytest.approx(107.6040, abs=1e-4) and g.drtg == pytest.approx(114.7776, abs=1e-4)
    assert g.net == pytest.approx(g.ortg - g.drtg) and g.pace == pytest.approx(97.58)
    assert g.efg == pytest.approx(45 / 84) and g.tov_pct == pytest.approx(11 / 94.48)
    assert g.orb_pct == pytest.approx(0.2) and g.ftr == pytest.approx(17 / 84)
    assert g.margin == -7 and not g.win and g.total_pts == 217
    assert (g.conference, g.division, g.opp_conference) == ("East", "Atlantic", "East")


def test_rest_and_b2b(enriched):
    b = _bos(enriched)
    assert b.rest_days[0] == 7 and not b.b2b[0]                 # first game of season
    assert b.rest_days[1] == 0 and b.b2b[1]                     # 10-22 then 10-23
    assert b.rest_days[2] == 1 and not b.b2b[2]
    s2 = b[b.season == "2025-26"].iloc[0]
    assert s2.rest_days == 7                                    # season boundary resets
    # offsets 0,1,3: three games in 4 nights on the 3rd date; 3,5,6 too
    assert list(b.three_in_four[:5]) == [False, False, True, False, True]


def test_ats_from_home_spread(conn):
    """home spread -5: home wins by 5 -> push both sides; wins by 7 -> home cover, away no_cover."""
    box = dict(min=240, fgm=40, fga=85, fg3m=10, fg3a=30, ftm=15, fta=20, oreb=10, dreb=30,
               reb=40, ast=20, stl=5, blk=5, tov=12, pf=20, season="2025-26", season_type="regular")
    games, tgs, snaps = [], [], []
    for gid, d, margin in (("g1", "2025-11-01", 5), ("g2", "2025-11-03", 7)):
        games.append(dict(game_id=gid, season="2025-26", game_date=d, home_team_id=1610612738,
                          away_team_id=1610612752, home_pts=100 + margin, away_pts=100, status="final"))
        tgs += [dict(game_id=gid, team_id=1610612738, opp_team_id=1610612752, game_date=d, is_home=1,
                     wl="W", **{**box, "pts": 100 + margin}),
                dict(game_id=gid, team_id=1610612752, opp_team_id=1610612738, game_date=d, is_home=0,
                     wl="L", **{**box, "pts": 100})]
        snaps.append(dict(game_id=gid, source="espn", book="DK", pulled_at=f"{d}T23:59:59Z",
                          is_closing=1, home_spread=-5.0, total=205.0, home_ml=-200, away_ml=170))
    upsert(conn, "games", games, ["game_id"])
    upsert(conn, "team_games", tgs, ["game_id", "team_id"])
    upsert(conn, "odds_snapshots", snaps, ["game_id", "source", "book", "pulled_at"])
    e = build_enriched(conn).set_index(["game_id", "is_home"])
    assert e.loc[("g1", 1), "ats"] == "push" and e.loc[("g1", 0), "ats"] == "push"
    assert e.loc[("g2", 1), "ats"] == "cover" and e.loc[("g2", 0), "ats"] == "no_cover"
    assert e.loc[("g2", 1), "team_spread"] == -5 and e.loc[("g2", 0), "team_spread"] == 5
    assert e.loc[("g2", 0), "cover_margin"] == -2 and e.loc[("g2", 1), "cover_margin"] == 2
    assert e.loc[("g1", 1), "ou"] == "push" and e.loc[("g2", 1), "ou"] == "over"  # 205, 207 vs 205
    assert bool(e.loc[("g1", 1), "is_favorite"]) and not bool(e.loc[("g1", 0), "is_favorite"])
    assert e.loc[("g1", 1), "team_ml"] == -200 and e.loc[("g1", 0), "team_ml"] == 170


def test_oddsapi_median_preferred_over_espn(fixture_db):
    gid = "0022500024"  # first 2025-26 game, has an ESPN line
    snaps = [dict(game_id=gid, source="oddsapi", book=b, pulled_at="2025-10-21T23:00:00Z",
                  is_closing=1, home_spread=sp, total=230.0, home_ml=ml, away_ml=-ml)
             for b, sp, ml in (("a", -1.0, 105), ("b", -2.0, -105), ("c", -6.0, 120))]
    upsert(fixture_db, "odds_snapshots", snaps, ["game_id", "source", "book", "pulled_at"])
    e = build_enriched(fixture_db)
    home = e[(e.game_id == gid) & (e.is_home == 1)].iloc[0]
    assert home.team_spread == -2.0 and home.total_line == 230.0
    assert home.team_ml == 105  # median in decimal space, never an invalid price like 0


def test_opp_strength_pre_game_only(fixture_db):
    base = build_enriched(fixture_db)
    target = base[(base.season == "2024-25")].sort_values("game_date").iloc[40]
    fixture_db.execute("update team_games set pts = pts + 50 where game_id = ?", (target.game_id,))
    after = build_enriched(fixture_db)
    cols = ["opp_win_pct_pre", "opp_net_pre", "opp_net_rank_pre"]
    b = base.set_index(["game_id", "team_id"]).loc[(target.game_id, target.team_id), cols]
    a = after.set_index(["game_id", "team_id"]).loc[(target.game_id, target.team_id), cols]
    assert list(a) == list(b)


def test_opp_strength_values(enriched):
    b = _bos(enriched)
    assert b.opp_win_pct_pre[:5].isna().all()  # opponents have < 5 prior games
    s1 = enriched[enriched.season == "2024-25"].sort_values("game_date")
    late = s1.iloc[-1]
    prior = s1[(s1.team_id == late.opp_team_id) & (s1.game_date < late.game_date)]
    assert late.opp_win_pct_pre == pytest.approx(prior.win.mean())
    assert late.opp_net_pre == pytest.approx(100 * (prior.pts.sum() - prior.opp_pts.sum()) / prior.game_poss.sum())
    assert 1 <= late.opp_net_rank_pre <= 4


def test_missing_odds_null(enriched):
    s1 = enriched[enriched.season == "2024-25"]
    assert s1.ats.isna().all() and s1.team_spread.isna().all() and s1.ou.isna().all()
    assert s1.net.notna().all() and s1.ortg.notna().all()
    s2 = enriched[enriched.season == "2025-26"]
    assert s2.ats.notna().all() and set(s2.ats) <= {"cover", "no_cover", "push"}


def test_month_weekday(enriched):
    g = _bos(enriched).iloc[0]  # 2024-10-22 was a Tuesday
    assert g.month == 10 and g.weekday == 1
