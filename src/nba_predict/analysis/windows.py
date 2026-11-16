"""Window-strength analysis: how well each last-N window's net-rating gap predicts the margin."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prior_net(enriched: pd.DataFrame, n: int) -> pd.Series:
    """Per team-game: sum-based net rating over the team's previous ``n`` games (all season types,
    crossing seasons), NaN until ``n`` prior games exist. Index aligned with ``enriched``."""
    e = enriched.sort_values(["team_id", "game_date"], kind="stable")
    g = e.groupby("team_id")
    roll = {c: g[c].transform(lambda s: s.shift(1).rolling(n, min_periods=n).sum())
            for c in ("pts", "opp_pts", "game_poss")}
    net = 100 * (roll["pts"] - roll["opp_pts"]) / roll["game_poss"].where(roll["game_poss"] > 0)
    return net.reindex(enriched.index)


def window_strength(enriched: pd.DataFrame, windows, seasons=None) -> pd.DataFrame:
    """``window, corr, n``: Pearson correlation of (home prior-N net − away prior-N net) with the
    home margin, over home rows of ``seasons`` (all if None)."""
    rows = []
    for n in sorted({int(w) for w in windows if int(w) >= 1}):
        e = enriched.assign(_pn=prior_net(enriched, n))
        opp = e[["game_id", "team_id", "_pn"]].rename(columns={"team_id": "opp_team_id", "_pn": "_opn"})
        h = e[e.is_home == 1].merge(opp, on=["game_id", "opp_team_id"])
        if seasons:
            h = h[h.season.isin(seasons)]
        d = (h._pn - h._opn)
        ok = d.notna() & h.margin.notna()
        rows.append(dict(window=n, corr=float(np.corrcoef(d[ok], h.margin[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                         n=int(ok.sum())))
    return pd.DataFrame(rows, columns=["window", "corr", "n"])
