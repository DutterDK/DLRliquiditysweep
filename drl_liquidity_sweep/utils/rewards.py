"""Reward calculation utilities."""

from __future__ import annotations
import pandas as pd


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


def drawdown_penalty(equity: pd.Series, lam: float) -> float:
    """
    Return negative penalty proportional to *incremental* drawdown
    between the last two equity points.

    Parameters
    ----------
    equity : pd.Series
        Historical equity values
    lam : float
        Scaling coefficient (λ). penalty = -lam * max(0, dd_increment)

    Returns
    -------
    float
        Negative penalty proportional to incremental drawdown
    """
    if len(equity) < 2:
        return 0.0
    prev_peak = equity.iloc[:-1].max()
    prev_dd = prev_peak - equity.iloc[-2]
    new_peak = max(prev_peak, equity.iloc[-1])
    new_dd = new_peak - equity.iloc[-1]
    dd_increment = max(0.0, new_dd - prev_dd)
    return -lam * dd_increment
