import pytest
import pandas as pd
from drl_liquidity_sweep.utils.rewards import (
    calc_realized_pnl, drawdown_penalty
)


def test_calc_realized_pnl():
    """Test P/L calculation for long and short positions."""
    # Profitable trades
    assert calc_realized_pnl(1.0001, 1.0004, +1) == pytest.approx(0.0003)
    assert calc_realized_pnl(1.0004, 1.0001, -1) == pytest.approx(0.0003)
    # Losing trades
    assert calc_realized_pnl(1.0004, 1.0001, +1) == pytest.approx(-0.0003)
    assert calc_realized_pnl(1.0001, 1.0004, -1) == pytest.approx(-0.0003)


def test_drawdown_penalty():
    """Test drawdown penalty calculation."""
    lam = 2.0
    # No new drawdown between last two points → penalty 0
    eq = pd.Series([100_000, 100_200, 100_150, 100_180])  # new peak at idx 1
    assert drawdown_penalty(eq, lam) == 0.0

    # New deeper drawdown → penalty proportional to increment
    eq2 = pd.Series([100_000, 100_200, 100_150, 100_100])  # new deeper DD 100
    penalty = drawdown_penalty(eq2, lam)
    # prev dd 50, new dd 100, increment 50
    assert penalty == pytest.approx(-lam * 50.0)
