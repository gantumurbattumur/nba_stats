"""Market views, model-vs-market evaluation and settlement. Analysis only: a "value flag" is
a recorded disagreement between model and market; nothing here sizes or places anything."""

from __future__ import annotations

import math
import sqlite3
from statistics import median

from scipy.stats import norm

from nba_predict.odds_math import best_price, consensus_novig

_SNAP_COLS = ("book", "pulled_at", "is_closing", "home_ml", "away_ml", "home_spread",
              "home_spread_price", "away_spread_price", "total", "over_price", "under_price")


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _pick_rows(rows: list[dict], closing_only: bool) -> list[dict]:
    """Latest pull of the preferred source: Odds API, else ESPN."""
    for source in ("oddsapi", "espn"):
        rs = [r for r in rows if r["source"] == source and (r["is_closing"] == 1 or not closing_only)]
        if rs:
            latest = max(r["pulled_at"] for r in rs)
            return [r for r in rs if r["pulled_at"] == latest]
    return []


def _two_way(rows, a_key, b_key, book_key="book"):
    pairs = [(r[a_key], r[b_key]) for r in rows if _ok(r[a_key]) and _ok(r[b_key])]
    if not pairs:
        return None, None, None
    novig = consensus_novig(pairs)
    a_best = best_price([p[0] for p in pairs])
    b_best = best_price([p[1] for p in pairs])
    books = (next(r[book_key] for r in rows if r[a_key] == a_best),
             next(r[book_key] for r in rows if r[b_key] == b_best))
    return novig, (int(a_best), int(b_best)), books


def market_view(conn: sqlite3.Connection, game_id: str, closing_only: bool = False) -> dict | None:
    """Consensus no-vig probabilities, best prices (+ books) and median lines for one game.
    Source = latest Odds API pull (latest *closing* pull when ``closing_only``), else ESPN.
    A market lacking a line or both prices has its keys set to ``None``; ``None`` if the
    game has no snapshot at all."""
    rows = [dict(r) for r in conn.execute(
        f"select source, {', '.join(_SNAP_COLS)} from odds_snapshots where game_id = ?", (game_id,))]
    rows = _pick_rows(rows, closing_only)
    if not rows:
        return None
    out: dict = {"source": rows[0]["source"], "pulled_at": rows[0]["pulled_at"]}
    out["ml_novig"], out["ml_price"], out["ml_book"] = _two_way(rows, "home_ml", "away_ml")

    sp = [r for r in rows if _ok(r["home_spread"])]
    nv, pr, bk = _two_way(sp, "home_spread_price", "away_spread_price")
    if sp and nv is not None:
        out.update(home_spread=float(median(r["home_spread"] for r in sp)), spread_novig=nv,
                   spread_price=pr, spread_book=bk)
    else:
        out.update(home_spread=None, spread_novig=None, spread_price=None, spread_book=None)

    tt = [r for r in rows if _ok(r["total"])]
    nv, pr, bk = _two_way(tt, "over_price", "under_price")
    if tt and nv is not None:
        out.update(total=float(median(r["total"] for r in tt)), total_novig=nv, total_price=pr,
                   total_book=bk)
    else:
        out.update(total=None, total_novig=None, total_price=None, total_book=None)
    return out


def evaluate_markets(pred: dict, market: dict, sigma: float, sigma_total: float) -> list[dict]:
    """Model vs market for every side: ``{market, side, line, price, book, model_prob,
    market_prob, edge}`` with ``edge = model_prob - market_prob`` (no-vig)."""
    out: list[dict] = []

    def add(mkt, sides, probs, novig, line_by_side, price_key, book_key):
        prices = market.get(price_key) or (None, None)
        books = market.get(book_key) or (None, None)
        for i, side in enumerate(sides):
            out.append(dict(market=mkt, side=side, line=line_by_side[i],
                            price=prices[i], book=books[i], model_prob=float(probs[i]),
                            market_prob=float(novig[i]), edge=float(probs[i] - novig[i])))

    p = pred.get("home_win_prob")
    if market.get("ml_novig") is not None and _ok(p):
        add("ml", ("home", "away"), (p, 1 - p), market["ml_novig"], (None, None), "ml_price", "ml_book")
    m, hs = pred.get("pred_margin"), market.get("home_spread")
    if market.get("spread_novig") is not None and _ok(hs) and _ok(m):
        ph = norm.cdf((m + hs) / sigma)
        add("spread", ("home", "away"), (ph, 1 - ph), market["spread_novig"], (hs, -hs),
            "spread_price", "spread_book")
    t, tl = pred.get("pred_total"), market.get("total")
    if market.get("total_novig") is not None and _ok(tl) and _ok(t):
        po = norm.cdf((t - tl) / sigma_total)
        add("total", ("over", "under"), (po, 1 - po), market["total_novig"], (tl, tl),
            "total_price", "total_book")
    return out


def settle(market: str, side: str, line: float | None, home_pts: int, away_pts: int) -> str:
    """'win' | 'loss' | 'push' for one side given the final score. Spread lines are from the
    side's own perspective (away line = -home_spread)."""
    margin = home_pts - away_pts
    if market == "ml":
        diff = margin if side == "home" else -margin
    elif market == "spread":
        diff = (margin if side == "home" else -margin) + float(line)
    elif market == "total":
        diff = (home_pts + away_pts - float(line)) * (1 if side == "over" else -1)
    else:
        raise ValueError(f"unknown market {market!r}")
    return "win" if diff > 0 else "loss" if diff < 0 else "push"


def pick_flags(evals: list[dict], threshold: float) -> list[dict]:
    """At most one side per market: the higher edge, if ``edge >= threshold``."""
    best: dict[str, dict] = {}
    for e in evals:
        if e["edge"] >= threshold and (e["market"] not in best or e["edge"] > best[e["market"]]["edge"]):
            best[e["market"]] = e
    return list(best.values())


# --- logging and grading value flags (Task 13) -------------------------------------------------

def flag_values(conn: sqlite3.Connection, date, model: str | None = None):
    """Record model-vs-market disagreements for ``date``'s games: for each market, the side with the
    higher edge if ``edge >= edge_threshold``. Re-running replaces that game's *unsettled* flags
    (so both sides of a market are never flagged); settled flags are never touched. Games without
    any snapshot are skipped. ``model`` defaults to ``backtest.best_model``."""
    import datetime as dt
    import json

    import pandas as pd

    from nba_predict.backtest import best_model
    from nba_predict.config import get_settings
    from nba_predict.db import upsert

    model = model or best_model(conn)
    thr = get_settings().edge_threshold
    rows = conn.execute(
        "select p.game_id, p.home_win_prob, p.pred_margin, p.pred_total, p.params_json from predictions p "
        "join games g using (game_id) where g.game_date = ? and p.model = ? and g.status = 'scheduled'",
        (date.isoformat(), model)).fetchall()
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    written = []
    for r in rows:
        mv = market_view(conn, r["game_id"])
        if mv is None:
            continue
        pj = json.loads(r["params_json"] or "{}")
        sigma = pj.get("sigma") or 12.5
        sigma_t = pj.get("sigma_total") or 18.0
        pred = dict(home_win_prob=r["home_win_prob"], pred_margin=r["pred_margin"], pred_total=r["pred_total"])
        flags = pick_flags(evaluate_markets(pred, mv, sigma, sigma_t), thr)
        conn.execute("delete from value_flags where game_id = ? and result is null", (r["game_id"],))
        recs = [dict(game_id=r["game_id"], created_at=now, model=model, **fl) for fl in flags]
        if recs:
            upsert(conn, "value_flags", recs, ["game_id", "market", "side"])
        conn.commit()
        written += recs
    return pd.DataFrame(written, columns=["game_id", "created_at", "model", "market", "side", "line", "price",
                                          "book", "model_prob", "market_prob", "edge"])


def _clv(flag: dict, close: dict) -> tuple[float | None, int | None, float | None]:
    """(closing_line, closing_price, clv) from the side's perspective. ML: closing no-vig prob of
    the side − implied prob of the taken price. Spread/total: line points gained vs the close."""
    from nba_predict.odds_math import american_to_prob

    i = 0 if flag["side"] in ("home", "over") else 1
    m = flag["market"]
    if m == "ml" and close.get("ml_novig"):
        price = (close.get("ml_price") or (None, None))[i]
        clv = close["ml_novig"][i] - american_to_prob(flag["price"]) if flag["price"] else None
        return None, price, clv
    if m == "spread" and close.get("home_spread") is not None:
        cl = close["home_spread"] if i == 0 else -close["home_spread"]
        return cl, (close.get("spread_price") or (None, None))[i], float(flag["line"]) - cl
    if m == "total" and close.get("total") is not None:
        cl = close["total"]
        gain = cl - float(flag["line"]) if i == 0 else float(flag["line"]) - cl
        return cl, (close.get("total_price") or (None, None))[i], gain
    return None, None, None


def grade(conn: sqlite3.Connection, date=None) -> int:
    """Settle unsettled flags whose game is final (others are skipped): result, ``units`` (the
    return of a hypothetical 1 unit, evaluation only) and closing line/price/CLV from the latest
    closing Odds API snapshot, else ESPN. Returns flags settled."""
    from nba_predict.odds_math import unit_return

    sql = ("select f.*, g.home_pts, g.away_pts from value_flags f join games g using (game_id) "
           "where f.result is null and g.status = 'final' and g.home_pts is not null")
    params: tuple = ()
    if date is not None:
        sql += " and g.game_date = ?"
        params = (date.isoformat(),)
    n = 0
    for r in [dict(x) for x in conn.execute(sql, params)]:
        res = settle(r["market"], r["side"], r["line"], r["home_pts"], r["away_pts"])
        units = unit_return(r["price"], res) if r["price"] is not None else None
        close = market_view(conn, r["game_id"], closing_only=True) or {}
        cl, cp, clv = _clv(r, close)
        conn.execute("update value_flags set result = ?, units = ?, closing_line = ?, closing_price = ?, "
                     "clv = ? where id = ?", (res, units, cl, cp, clv, r["id"]))
        n += 1
    conn.commit()
    return n
