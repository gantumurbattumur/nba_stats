import pandas as pd
import pytest

from nba_predict.models.elo import EloParams, run_elo, season_revert

# formula checks use the plan's original parameters; the tuned defaults are tested separately
PLAN = EloParams(k=20.0, home_adv=100.0)


def _games(rows):
    cols = ["game_id", "season", "game_date", "home_team_id", "away_team_id", "home_pts",
            "away_pts", "status"]
    return pd.DataFrame(rows, columns=cols)


def test_first_game_prob():
    out = run_elo(_games([("g1", "2024-25", "2024-10-22", 1, 2, 110, 100, "final")]), PLAN)
    r = out.iloc[0]
    assert r.h_elo == 1505 and r.a_elo == 1505
    assert r.elo_prob == pytest.approx(0.6400649, abs=1e-7)
    assert r.elo_margin == pytest.approx(100 / 28)


def test_zero_sum_update():
    g = _games([("g1", "2024-25", "2024-10-22", 1, 2, 110, 100, "final"),
                ("g2", "2024-25", "2024-10-23", 1, 3, 90, 120, "final"),
                ("g3", "2024-25", "2024-10-24", 2, 3, 101, 99, "final")])
    out = run_elo(g, PLAN).set_index("game_id")
    # after g1: team1 + team2 still sum to 2*1505
    t1_after_g1 = out.loc["g2", "h_elo"]
    t2_after_g1 = out.loc["g3", "h_elo"]
    assert t1_after_g1 + t2_after_g1 == pytest.approx(2 * 1505)
    assert t1_after_g1 > 1505  # winner gains


def test_update_formula():
    p = PLAN
    g = _games([("g1", "2024-25", "2024-10-22", 1, 2, 120, 100, "final"),
                ("g2", "2024-25", "2024-10-23", 1, 2, 100, 100, "scheduled")])
    out = run_elo(g, p).set_index("game_id")
    prob = 1 / (1 + 10 ** (-(100) / 400))
    mult = (20 + 3) ** 0.8 / (7.5 + 0.006 * 100)  # home won: winner diff includes home adv
    assert out.loc["g2", "h_elo"] == pytest.approx(1505 + 20 * mult * (1 - prob))


def test_away_win_uses_negative_home_adv_in_multiplier():
    g = _games([("g1", "2024-25", "2024-10-22", 1, 2, 90, 100, "final"),
                ("g2", "2024-25", "2024-10-23", 2, 1, 0, 0, "scheduled")])
    out = run_elo(g, PLAN).set_index("game_id")
    prob = 1 / (1 + 10 ** (-(100) / 400))
    mult = (10 + 3) ** 0.8 / (7.5 + 0.006 * (-100))
    assert out.loc["g2", "h_elo"] == pytest.approx(1505 + 20 * mult * prob)  # team 2 won away


def test_season_revert():
    p = PLAN
    assert season_revert(1605, p) == pytest.approx(1580)
    g = _games([("g1", "2024-25", "2024-10-22", 1, 2, 130, 100, "final"),
                ("g2", "2025-26", "2025-10-21", 1, 2, 110, 100, "final")])
    out = run_elo(g, p).set_index("game_id")
    prob = 1 / (1 + 10 ** (-100 / 400))
    end_s1 = 1505 + 20 * ((30 + 3) ** 0.8 / (7.5 + 0.006 * 100)) * (1 - prob)
    # pre-game rating for the first game of season 2 = reverted end-of-season-1 rating
    assert out.loc["g2", "h_elo"] == pytest.approx(season_revert(end_s1, p))
    assert out.loc["g2", "a_elo"] == pytest.approx(season_revert(3010 - end_s1, p))


def test_pre_game_values():
    g = _games([("g1", "2024-25", "2024-10-22", 1, 2, 150, 80, "final")])
    assert run_elo(g, PLAN).iloc[0].h_elo == 1505  # own result not used


def test_postponed_does_not_update():
    g = _games([("g1", "2024-25", "2024-10-22", 1, 2, None, None, "postponed"),
                ("g2", "2024-25", "2024-10-23", 1, 2, None, None, "scheduled")])
    out = run_elo(g, PLAN).set_index("game_id")
    assert out.loc["g2", "h_elo"] == 1505


def test_elo_merged_into_features(fixture_db, enriched):
    from nba_predict.features.build import build_feature_table

    f = build_feature_table(fixture_db)
    assert {"h_elo", "a_elo", "elo_prob"} <= set(f.columns)
    assert f.elo_prob.between(0, 1).all() and f.sort_values("game_date").iloc[0].h_elo == 1505
    sched = f[f.status == "scheduled"]
    assert (sched.h_elo != 1505).all()  # current ratings carried to scheduled games


def test_defaults_tuned_on_2023_24_only():
    # C phase 3: grid k in 20..40, home_adv in 20..100 scored by log loss on 2023-24 final games
    # only (no test-season peeking): best k=25, home_adv=40 (log loss 0.6205; k=20/100 gave 0.6191 on
    # 2024-26 vs 0.6059 for the tuned values, margin bias +1.85 -> -0.35)
    p = EloParams()
    assert (p.k, p.home_adv) == (25.0, 40.0)
    out = run_elo(_games([("g1", "2024-25", "2024-10-22", 1, 2, 110, 100, "final")]))
    assert out.iloc[0].elo_margin == pytest.approx(40 / 28)


def test_feature_fingerprint_tracks_elo_params(fixture_db, monkeypatch):
    from nba_predict.features import build as fb

    fp = fb.features_fingerprint(fixture_db, (5,))
    monkeypatch.setattr(fb, "EloParams", lambda: EloParams(home_adv=55.0))
    assert fb.features_fingerprint(fixture_db, (5,)) != fp  # stale Elo columns must rebuild
