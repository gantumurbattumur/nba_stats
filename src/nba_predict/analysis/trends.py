"""Transparent, pre-date statistical tendencies. Percentages are fractions in [0, 1].

The model's disagreement chooses the direction. Three independent checks support it:
1. Margin disagreement >= 2 points / total >= 3 points / win probability >= 3 percentage points.
2. Last 10 overall: both teams' relevant rates support the direction at 55% (opponent <=45%).
3. Last 10 in the upcoming venue: the same rule, home team at home and away team on the road.
Trend checks require at least 5 decided games for EACH team. Pushes and missing lines do not
enter the percentage denominator. 3/2/1/0 supporting checks → Strong/Lean/Slight/None.
Missing model or line, or exactly zero disagreement, gives None. Labels describe tendencies,
not calibrated probabilities. Moneyline uses win rates; totals use over/under rates.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from nba_predict.config import get_settings, model_scope
from nba_predict.ingest import season_for_date
from nba_predict.odds_math import american_to_prob, devig_two_way
from nba_predict.teams import by_abbr, by_nba_id

LABELS = ("Strong", "Lean", "Slight", "None")
SPLITS = ("all", "home", "away", "fav", "dog")
RULE = ("Direction follows model vs line. One supporting factor each for: model difference ≥2 spread points, "
        "≥3 total points or ≥3 percentage points for moneyline; both teams' last-10 trend; "
        "both teams' last-10 venue trend. Trends need ≥55% support and ≥5 decided games per team. "
        "3 factors = Strong, 2 = Lean, 1 = Slight, 0 = None. Missing evidence is not agreement.")


def team_cover_trends(enriched: pd.DataFrame, team, as_of, windows=(5, 10, 15, None),
                      splits=SPLITS) -> pd.DataFrame:
    """Window AFTER split filtering; season window stays in as_of's season, N may cross into
    the configured prior season. All input games are strictly before the ET date as_of.
    ``games`` counts all rows; ats_n / ou_n count only decided outcomes with a line.
    """
    day = dt.date.fromisoformat(str(as_of)[:10])
    t = by_abbr(team) if isinstance(team, str) else by_nba_id(team)
    if t is None:
        raise ValueError(f"Unknown team: {team}")
    if set(splits) - set(SPLITS):
        raise ValueError(f"Splits must be among {SPLITS}")
    history = model_scope(enriched)
    history = history[(history.team_id == t.nba_id) & (history.game_date < day.isoformat())]
    if "status" in history:
        history = history[history.status == "final"]
    history = history.sort_values(["game_date", "game_id"], kind="stable")
    rows = []
    for split in splits:
        g = history
        if split in ("home", "away"):
            g = g[g.is_home == int(split == "home")]
        elif split in ("fav", "dog"):
            g = g[g.team_spread.lt(0) if split == "fav" else g.team_spread.gt(0)]
        for n in windows:
            if n is not None and n < 1:
                raise ValueError("Windows must be positive or None")
            w = g[g.season == season_for_date(day)] if n is None else g.tail(n)
            covers, losses = int(w.ats.eq("cover").sum()), int(w.ats.eq("no_cover").sum())
            overs, unders = int(w.ou.eq("over").sum()), int(w.ou.eq("under").sum())
            an, on = covers + losses, overs + unders
            wn = int(w.win.notna().sum())
            rows.append(dict(team_id=t.nba_id, team=t.abbr, split=split,
                             window="season" if n is None else f"last{n}", games=len(w),
                             covers=covers, no_covers=losses, ats_pushes=int(w.ats.eq("push").sum()),
                             ats_n=an, ats_pct=covers / an if an else np.nan,
                             overs=overs, unders=unders, ou_pushes=int(w.ou.eq("push").sum()),
                             ou_n=on, over_pct=overs / on if on else np.nan,
                             under_pct=unders / on if on else np.nan, win_n=wn,
                             win_pct=float(w.win.mean()) if wn else np.nan,
                             pace=float(w.pace.mean()) if len(w) else np.nan))
    return pd.DataFrame(rows)


def _number(value):
    try:
        return float(value) if np.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def lean(game_row, trends: pd.DataFrame, model_pred, line) -> dict:
    """Return spread/total/moneyline labels, directions, disagreement and auditable factors.

    ``trends`` combines team_cover_trends for both teams; ``line`` has home_spread, total,
    home_ml / away_ml (or ml_novig). No current outcome enters a factor.
    """
    home, away = int(game_row["home_team_id"]), int(game_row["away_team_id"])
    prob = _number(model_pred.get("home_win_prob"))
    margin, total = _number(model_pred.get("pred_margin")), _number(model_pred.get("pred_total"))
    spread, total_line = _number(line.get("home_spread")), _number(line.get("total"))
    novig = line.get("ml_novig")
    hm, am = _number(line.get("home_ml")), _number(line.get("away_ml"))
    if novig is None and hm is not None and am is not None:
        novig = devig_two_way(american_to_prob(hm), american_to_prob(am))
    market_prob = _number(novig[0]) if novig is not None else None
    deltas = {"spread": None if margin is None or spread is None else margin + spread,
              "total": None if total is None or total_line is None else total - total_line,
              "moneyline": None if prob is None or market_prob is None else prob - market_prob}

    def rate(tid, split, col, ncol):
        r = trends[(trends.team_id == tid) & (trends.split == split) & (trends.window == "last10")]
        if r.empty:
            return None, 0
        return _number(r.iloc[0][col]), int(r.iloc[0][ncol])

    out = {}
    for market, delta in deltas.items():
        threshold = {"spread": 2.0, "total": 3.0, "moneyline": 0.03}[market]
        direction = None if delta is None or delta == 0 else (
            ("over" if delta > 0 else "under") if market == "total" else ("home" if delta > 0 else "away"))
        def factor(name, supported, detail, n=None):
            return dict(factor=name, supported=supported,
                        status="–" if supported is None else "✓" if supported else "✗", detail=detail, n=n)
        unit, scale = ("percentage points", 100) if market == "moneyline" else ("points", 1)
        factors = [factor("Model vs line", None if delta is None else abs(delta) >= threshold,
                          "Model or line unavailable" if delta is None else
                          f"{delta * scale:+.1f} {unit}; threshold {threshold * scale:g}")]
        col, ncol = {"spread": ("ats_pct", "ats_n"), "total": ("over_pct", "ou_n"),
                     "moneyline": ("win_pct", "win_n")}[market]
        for name, hs, aws in (("Recent trend (last 10)", "all", "all"),
                              ("Venue trend (last 10)", "home", "away")):
            hr, hn = rate(home, hs, col, ncol)
            ar, an = rate(away, aws, col, ncol)
            supported = None
            if direction and hr is not None and ar is not None and min(hn, an) >= 5:
                if market == "total":
                    supported = (hr >= .55 and ar >= .55) if direction == "over" else (hr <= .45 and ar <= .45)
                else:
                    supported = (hr >= .55 and ar <= .45) if direction == "home" else (hr <= .45 and ar >= .55)
            htxt, atxt = ("—" if hr is None else f"{hr:.0%}"), ("—" if ar is None else f"{ar:.0%}")
            factors.append(factor(name, supported, f"Home {htxt} (n={hn}); away {atxt} (n={an})", min(hn, an)))
        count = sum(f["supported"] is True for f in factors) if direction else 0
        out[market] = dict(label={3: "Strong", 2: "Lean", 1: "Slight", 0: "None"}[count],
                           direction=direction, agreement_count=count, disagreement=delta, factors=factors)
    return out


def lean_history(conn, season: str, results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hit rates by label/market AND prediction origin. Prior-season fit is explicitly in-sample;
    current-season retro rows use only pre-date training. Pushes/missing outcomes are excluded.
    Never merge reconstructed and recorded pre-tip observations without provenance.
    """
    from nba_predict.analysis.filters import FilterSpec
    from nba_predict.analysis.results import build_results

    in_sample = season in get_settings().train_seasons
    if results is None:
        results = build_results(conn, FilterSpec(seasons=[season]),
                                prediction_mode="in_sample" if in_sample else "pregame")
    rows = []
    results = results[results.season == season] if not results.empty else results
    origins = sorted(results.prediction_origin.unique()) if not results.empty else ["unavailable"]
    for market in ("spread", "total", "moneyline"):
        for origin in origins:
            for label in LABELS:
                r = results[(results.prediction_origin == origin) & (results[f"{market}_label"] == label)] if not results.empty else results
                hits = r[f"{market}_hit"].dropna() if not r.empty else pd.Series(dtype=float)
                n = len(hits)
                rows.append(dict(season=season, evaluation="in-sample" if in_sample else "pre-date",
                                 origin=origin, market=market, label=label, n=n,
                                 hit_rate=float(hits.mean()) if n else np.nan, small_sample=n < 30))
    return pd.DataFrame(rows)
