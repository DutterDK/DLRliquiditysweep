"""Reward calculation utilities."""

from __future__ import annotations
import pandas as pd
from typing import Union, List


def zero_reward(*args, **kwargs) -> float:  # placeholder
    """Return zero reward (placeholder for future P/L logic)."""
    return 0.0


def calc_realized_pnl(entry: float, exit_: float, direction: int) -> float:
    """
    Return realised P/L in raw price units (positive = profit).

    Parameters
    ----------
    entry      : entry price (float)
    exit_      : exit price (float)
    direction  : +1 for long, -1 for short

    Examples
    --------
    >>> calc_realized_pnl(1.0001, 1.0004, +1)
    0.0003
    >>> calc_realized_pnl(1.0004, 1.0001, -1)
    0.0003
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 (long) or -1 (short)")
    return (exit_ - entry) * direction


def drawdown_penalty(equity, lam=1.0):
    """Calculate drawdown penalty for a series of equity values."""
    if len(equity) < 2:
        return 0.0
    max_equity = equity[0]
    penalty = 0.0
    for eq in equity[1:]:
        if eq > max_equity:
            max_equity = eq
        else:
            penalty += lam * (max_equity - eq)
    return -penalty
