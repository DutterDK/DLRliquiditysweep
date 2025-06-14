"""Reward calculation utilities."""

from __future__ import annotations
import pandas as pd
from typing import Union, List
import numpy as np


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


def drawdown_penalty(equity_series, lam: float = 0.001):
    """Calculate drawdown penalty.

    Args:
        equity_series: Series of equity values
        lam: Penalty coefficient

    Returns:
        float: Drawdown penalty
    """
    if not isinstance(equity_series, pd.Series):
        equity_series = pd.Series(equity_series)
    peak = equity_series.cummax()
    dd   = (peak - equity_series) / peak.replace(0, np.nan)
    inc  = dd.diff().fillna(0).clip(lower=0)
    return -lam * inc.iloc[-1]
