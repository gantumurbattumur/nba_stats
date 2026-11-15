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


def test_load_enriched_rebuilds_when_stale(fixture_db):
    from nba_predict.analysis.enrich import load_enriched

    n0 = len(load_enriched(fixture_db))
    assert n0 == 96
    calls = []
    import nba_predict.analysis.enrich as en
    orig = en.build_enriched
    en.build_enriched = lambda c: calls.append(1) or orig(c)
    try:
        assert len(load_enriched(fixture_db)) == 96 and calls == []  # fresh: reused
        # new final game (row count + max date change)
        fixture_db.execute("update games set status='final', home_pts=100, away_pts=90 "
                           "where game_id='0022599000'")
        fixture_db.executemany(
            "insert into team_games (game_id, team_id, opp_team_id, season, game_date, is_home, wl,"
            " min, pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf,"
            " plus_minus, season_type) select '0022599000', ?, ?, '2025-26', '2025-11-10', ?, ?,"
            " 240, ?, 40, 85, 10, 30, 10, 15, 10, 30, 40, 20, 5, 5, 12, 20, ?, 'regular'",
            [(1610612738, 1610612752, 1, "W", 100, 10.0), (1610612752, 1610612738, 0, "L", 90, -10.0)])
        fixture_db.commit()
        assert len(load_enriched(fixture_db)) == 98 and len(calls) == 1
        # box-score correction with unchanged row count
        fixture_db.execute("update team_games set pts = pts + 1 where game_id='0022400000'")
        fixture_db.commit()
        load_enriched(fixture_db)
        assert len(calls) == 2
        # new/changed closing line (ESPN backfill) with unchanged team_games
        fixture_db.execute("update odds_snapshots set home_spread = home_spread - 1 "
                           "where id = (select min(id) from odds_snapshots)")
        fixture_db.commit()
        load_enriched(fixture_db)
        assert len(calls) == 3
        load_enriched(fixture_db)
        assert len(calls) == 3
    finally:
        en.build_enriched = orig


def test_load_enriched_rebuilds_table_without_fingerprint(fixture_db):
    # DBs built before the fingerprint existed: stale rows in the table, no meta row
    from nba_predict.analysis.enrich import build_enriched, load_enriched

    build_enriched(fixture_db)
    fixture_db.execute("delete from team_games_enriched where season = '2025-26'")
    fixture_db.execute("delete from derived_meta")
    fixture_db.commit()
    assert len(load_enriched(fixture_db)) == 96


def test_rest_days_never_negative_on_same_date_glitch(fixture_db):
    """Two games for one team on one date (data glitch) must not give rest_days = -1."""
    from nba_predict.features.build import build_feature_table

    row = dict(fixture_db.execute("select * from games where game_id = '0022400000'").fetchone())
    tg = [dict(r) for r in fixture_db.execute("select * from team_games where game_id = '0022400000'")]
    dup = "0022400999"
    upsert(fixture_db, "games", [{**row, "game_id": dup}], ["game_id"])
    upsert(fixture_db, "team_games", [{**t, "game_id": dup} for t in tg], ["game_id", "team_id"])
    e = build_enriched(fixture_db)
    assert (e.rest_days >= 0).all()
    f = build_feature_table(fixture_db)
    assert (f.h_rest >= 0).all() and (f.a_rest >= 0).all()
