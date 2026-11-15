"""Pure odds functions (American odds). Analysis only: ``unit_return`` evaluates a
hypothetical 1-unit flag; nothing here sizes stakes."""

from __future__ import annotations

from statistics import median
from typing import Literal


def american_to_prob(odds: int) -> float:
    """Implied probability (includes the bookmaker margin)."""
    odds = float(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def american_to_decimal(odds: int) -> float:
    odds = float(odds)
    if odds < 0:
        return 1.0 + 100.0 / -odds
    return 1.0 + odds / 100.0


def devig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    """Remove the margin by scaling both implied probabilities to sum to 1."""
    s = p_a + p_b
    return p_a / s, p_b / s


def unit_return(odds: int, result: Literal["win", "loss", "push"]) -> float:
    """Return of a hypothetical 1-unit flag at ``odds`` (model evaluation only)."""
    if result == "win":
        return american_to_decimal(odds) - 1.0
    if result == "loss":
        return -1.0
    if result == "push":
        return 0.0
    raise ValueError(f"unknown result {result!r}")


def consensus_novig(pairs: list[tuple[int, int]]) -> tuple[float, float] | None:
    """Median across books of per-book no-vig probabilities; ``None`` if no books."""
    if not pairs:
        return None
    a = median(devig_two_way(american_to_prob(x), american_to_prob(y))[0] for x, y in pairs)
    return a, 1.0 - a


def best_price(prices: list[int]) -> int:
    """Price with the highest decimal payout."""
    return max(prices, key=american_to_decimal)
