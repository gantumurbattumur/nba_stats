"""Standard NBA Elo: K, home advantage, margin-of-victory multiplier, season mean reversion.

All output ratings are *pre-game*; scheduled games get the current (post-latest-game) ratings.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel


class EloParams(BaseModel):
    # tuned on 2023-24 final games only (log loss; grid k 20..40, home_adv 20..100). The plan's
    # k=20 / home_adv=100 over-predicted the home side: +3.6 pts vs ~+1.8 actual.
    k: float = 25.0
    home_adv: float = 40.0
    revert: float = 0.25
    mean: float = 1505.0


def season_revert(r: float, p: EloParams) -> float:
    return p.mean + (1 - p.revert) * (r - p.mean)


def win_prob(h: float, a: float, p: EloParams) -> float:
    return 1.0 / (1.0 + 10 ** (-(h - a + p.home_adv) / 400.0))


def run_elo(games: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """games: game_id, season, game_date, home_team_id, away_team_id, home_pts, away_pts, status.
    Returns game_id, h_elo, a_elo, elo_prob, elo_margin (pre-game)."""
    p = params
    g = games.sort_values(["game_date", "game_id"], kind="stable")
    rating: dict[int, float] = {}
    last_season: dict[int, str] = {}
    rows = []
    for r in g.itertuples(index=False):
        ids = (int(r.home_team_id), int(r.away_team_id))
        for t in ids:
            if t not in rating:
                rating[t] = p.mean
            elif last_season[t] != r.season:
                rating[t] = season_revert(rating[t], p)
            last_season[t] = r.season
        h, a = rating[ids[0]], rating[ids[1]]
        prob = win_prob(h, a, p)
        rows.append((r.game_id, h, a, prob, (h - a + p.home_adv) / 28.0))
        if r.status != "final" or pd.isna(r.home_pts) or pd.isna(r.away_pts):
            continue
        mov = float(r.home_pts) - float(r.away_pts)
        home_won = mov > 0
        diff_winner = (h - a + p.home_adv) if home_won else (a - h - p.home_adv)
        mult = ((abs(mov) + 3) ** 0.8) / (7.5 + 0.006 * diff_winner)
        delta = p.k * mult * ((1.0 if home_won else 0.0) - prob)
        rating[ids[0]] = h + delta
        rating[ids[1]] = a - delta
    return pd.DataFrame(rows, columns=["game_id", "h_elo", "a_elo", "elo_prob", "elo_margin"])
