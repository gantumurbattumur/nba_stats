from datetime import date

import numpy as np
import pandas as pd
import pytest

from nba_predict.analysis.enrich import build_enriched
from nba_predict.analysis.filters import FilterSpec, apply_filter
from nba_predict.features import build as fb
from nba_predict.features.build import build_feature_table, team_profile

BOS = 1610612738
METRIC_KEYS = ["ortg", "drtg", "net", "pace", "efg", "tov_pct", "orb_pct", "ftr", "margin",
               "win_pct", "ats_pct", "over_pct", "cover_margin", "sos"]


@pytest.fixture
def feats(fixture_db, enriched):
    return build_feature_table(fixture_db)


def test_one_row_per_game_and_written(fixture_db, feats):
    assert len(feats) == fixture_db.execute("select count(*) from games").fetchone()[0] == 50
    assert fixture_db.execute("select count(*) from game_features").fetchone()[0] == 50
    for c in ("game_id", "season", "game_date", "home_team_id", "away_team_id", "h_rest", "a_b2b",
              "h_3in4", "a_gp_szn", "m_home_spread", "m_total", "m_home_ml", "m_away_ml",
              "m_home_novig", "margin", "total", "home_win", "h_all_l5_adjnet", "d_split_szn_net",
              "h_split_l15_gp"):
        assert c in feats.columns, c
    for side in ("h", "a"):
        for scope in ("all", "split"):
            for w in ("l5", "l8", "l10", "l15", "szn"):
                for m in METRIC_KEYS:
                    assert f"{side}_{scope}_{w}_{m}" in feats.columns


def test_no_leakage(fixture_db):
    base = build_feature_table(fixture_db).set_index("game_id")
    d = "2025-10-29"  # mid-season 2 date
    assert d in set(base.game_date)
    fixture_db.execute("update team_games set pts = pts + 40, fgm = fgm + 20 where game_date >= ?", (d,))
    fixture_db.execute("update games set home_pts = home_pts + 40 where game_date >= ?", (d,))
    fixture_db.commit()
    build_enriched(fixture_db)
    after = build_feature_table(fixture_db).set_index("game_id")
    cols = [c for c in base.columns if c.startswith(("h_", "a_", "d_"))]
    on_or_before = base.index[base.game_date <= d]
    pd.testing.assert_frame_equal(base.loc[on_or_before, cols], after.loc[on_or_before, cols])
    later = base.index[base.game_date > d]
    assert not base.loc[later, "h_all_l5_ortg"].equals(after.loc[later, "h_all_l5_ortg"])


def _sample_games(feats):
    f = feats.sort_values("game_date")
    return [f.iloc[10], f.iloc[30], f.iloc[45], f.iloc[-1]]  # incl. a scheduled game


@pytest.mark.parametrize("n,win", [(5, "l5"), (10, "l10"), (None, "szn")])
def test_window_matches_team_profile(feats, enriched, n, win):
    from nba_predict.teams import by_nba_id

    for row in _sample_games(feats):
        for side, team_col, split_venue in (("h", "home_team_id", "home"), ("a", "away_team_id", "away")):
            abbr = by_nba_id(row[team_col]).abbr
            as_of = date.fromisoformat(row.game_date)
            for scope, venue in (("all", "all"), ("split", split_venue)):
                p = team_profile(enriched, abbr, as_of, n, venue)
                assert row[f"{side}_{scope}_{win}_gp"] == p["gp"]
                for m in METRIC_KEYS + ["adjnet"]:
                    v = row[f"{side}_{scope}_{win}_{m}"]
                    if np.isnan(p[m]):
                        assert np.isnan(v), (row.game_id, side, scope, win, m)
                    else:
                        assert v == pytest.approx(p[m], abs=1e-9), (row.game_id, side, scope, win, m)


def test_short_history_uses_available_games(feats):
    row = feats[feats.game_id == "0022400004"].iloc[0]  # BOS (home) 3rd game of season 1
    assert row.home_team_id == BOS
    assert row.h_all_l15_gp == 2 and np.isfinite(row.h_all_l15_net)
    assert row.h_all_szn_gp == 2 and row.h_gp_szn == 2
    first = feats.sort_values("game_date").iloc[0]
    assert first.h_all_l5_gp == 0 and np.isnan(first.h_all_l5_net)  # no history: NaN, no crash


def test_l_windows_reach_into_last_season(feats):
    s2 = feats[(feats.season == "2025-26") & (feats.home_team_id == BOS)].sort_values("game_date").iloc[0]
    assert s2.h_gp_szn == 0 and s2.h_all_szn_gp == 0 and np.isnan(s2.h_all_szn_net)
    assert s2.h_all_l5_gp == 5 and s2.h_all_l15_gp == 12  # whole of season 1


def test_cross_season_stops_after_threshold(fixture_db, enriched, monkeypatch):
    monkeypatch.setattr(fb, "CROSS_SEASON_GAMES", 3)
    f = build_feature_table(fixture_db)
    bos = f[(f.season == "2025-26") & ((f.home_team_id == BOS) | (f.away_team_id == BOS))]
    bos = bos[bos.status != "scheduled"].sort_values("game_date")
    g5 = bos.iloc[4]  # 4 prior in-season games >= 3 -> in-season only
    side = "h" if g5.home_team_id == BOS else "a"
    assert g5[f"{side}_all_l5_gp"] == 4
    p = team_profile(enriched, "BOS", date.fromisoformat(g5.game_date), 5, "all")
    assert p["gp"] == 4 and p["net"] == pytest.approx(g5[f"{side}_all_l5_net"])
    g3 = bos.iloc[2]  # 2 prior in-season games < 3 -> reaches back
    side = "h" if g3.home_team_id == BOS else "a"
    assert g3[f"{side}_all_l5_gp"] == 5


def test_scheduled_games_have_features(feats):
    sched = feats.query("home_win.isna()")
    assert len(sched) == 2 and sched.h_all_l5_net.notna().all()
    assert sched.margin.isna().all() and (sched.h_all_l5_gp == 5).all()
    assert (sched.h_gp_szn == 12).all() and (sched.h_rest == 0).all()  # day after the last date


def test_diffs_and_market(feats):
    r = feats[feats.season == "2025-26"].dropna(subset=["home_win"]).iloc[-3]  # late: sos known
    assert r.d_all_l5_net == pytest.approx(r.h_all_l5_net - r.a_all_l5_net)
    assert r.d_split_szn_adjnet == pytest.approx(r.h_split_szn_adjnet - r.a_split_szn_adjnet)
    assert r.m_home_spread < 0 or r.m_home_spread > 0
    assert 0 < r.m_home_novig < 1 and (r.m_home_novig > 0.5) == (r.m_home_spread < 0)
    assert feats[feats.season == "2024-25"].m_home_spread.isna().all()
    g = feats[feats.game_id == "0022400000"].iloc[0]
    assert (g.margin, g.total, g.home_win) == (-7, 217, 0)


def test_any_window(fixture_db, enriched):
    f = build_feature_table(fixture_db, windows=(3,))
    assert "h_all_l3_net" in f.columns and "h_all_l5_net" not in f.columns
    assert "h_all_szn_net" in f.columns


def test_profile_honors_filter(enriched):
    as_of = date(2025, 10, 30)
    p = team_profile(enriched, "BOS", as_of, 10, "all", FilterSpec(opp_min_win_pct=0.5))
    prior = enriched[(enriched.team_abbr == "BOS") & (enriched.game_date < as_of.isoformat())]
    n = int((prior.opp_win_pct_pre >= 0.5).sum())
    assert n > 0 and p["gp"] == min(n, 10)
    sub = apply_filter(prior, FilterSpec(opp_min_win_pct=0.5)).sort_values("game_date").tail(10)
    assert p["pts"] == pytest.approx(sub.pts.mean()) if "pts" in p else True


def test_bad_team_minutes_do_not_reach_pace_anywhere(fixture_db):
    # 0022600231-style glitch: one team's MIN = 168 while the opponent has 240
    base = build_feature_table(fixture_db)
    gid = "0022400004"
    fixture_db.execute("update team_games set min = 168 where game_id = ? and is_home = 0", (gid,))
    fixture_db.commit()
    e = build_enriched(fixture_db)
    assert (e[e.game_id == gid].game_min == 240).all()
    after = build_feature_table(fixture_db)
    pace_cols = [c for c in base.columns if c.endswith("_pace")]
    pd.testing.assert_frame_equal(base[pace_cols], after[pace_cols])
    row = e[e.game_id == gid].iloc[0]
    p = team_profile(e, row.team_abbr, date(2024, 12, 1), None, "all")
    p0 = team_profile(build_enriched_without_glitch(fixture_db, gid), row.team_abbr,
                      date(2024, 12, 1), None, "all")
    assert p["pace"] == pytest.approx(p0["pace"])


def build_enriched_without_glitch(conn, gid):
    conn.execute("update team_games set min = 240 where game_id = ?", (gid,))
    conn.commit()
    return build_enriched(conn)


# --- staleness of game_features (derived_meta fingerprint) -----------------------------------

def _spy_builds(monkeypatch):
    calls = []
    real = fb.build_feature_table

    def spy(conn, windows=(5, 8, 10, 15)):
        calls.append(tuple(windows))
        return real(conn, windows)

    monkeypatch.setattr(fb, "build_feature_table", spy)
    return calls


def test_load_features_reuses_when_fresh(fixture_db, monkeypatch):
    build_feature_table(fixture_db)
    calls = _spy_builds(monkeypatch)
    f = fb.load_features(fixture_db)
    assert calls == [] and len(f) == 50


def test_load_features_rebuilds_on_new_scheduled_game(fixture_db, monkeypatch):
    from nba_predict.db import upsert

    build_feature_table(fixture_db)
    calls = _spy_builds(monkeypatch)
    upsert(fixture_db, "games", [dict(game_id="0022599999", season="2025-26", game_date="2025-11-11",
                                      home_team_id=1610612747, away_team_id=BOS, status="scheduled")],
           ["game_id"])
    f = fb.load_features(fixture_db)
    assert len(calls) == 1 and "0022599999" in set(f.game_id)
    fb.load_features(fixture_db)
    assert len(calls) == 1  # fresh again


def test_load_features_rebuilds_on_status_or_score_change(fixture_db, monkeypatch):
    build_feature_table(fixture_db)
    calls = _spy_builds(monkeypatch)
    fixture_db.execute("update games set status = 'postponed' where game_id = '0022599000'")
    fixture_db.commit()
    fb.load_features(fixture_db)
    assert len(calls) == 1


def test_load_features_rebuilds_on_windows_and_enriched_change(fixture_db, monkeypatch):
    build_feature_table(fixture_db)
    calls = _spy_builds(monkeypatch)
    f = fb.load_features(fixture_db, windows=(3,))
    assert calls == [(3,)] and "h_all_l3_net" in f.columns
    fixture_db.execute("update odds_snapshots set total = total + 1 where rowid = 1")
    fixture_db.commit()
    fb.load_features(fixture_db, windows=(3,))
    assert len(calls) == 2  # closing line changed -> enriched fingerprint changed
