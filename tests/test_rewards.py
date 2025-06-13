import pytest
import pandas as pd
import numpy as np
from drl_liquidity_sweep.utils.rewards import calc_realized_pnl, drawdown_penalty


def test_calc_realized_pnl():
    """Test P/L calculation for long and short positions."""
    # Test long position
    assert calc_realized_pnl(1.0001, 1.0004, 1) == pytest.approx(0.0003)
    assert calc_realized_pnl(1.0004, 1.0001, 1) == pytest.approx(-0.0003)

    # Test short position
    assert calc_realized_pnl(1.0004, 1.0001, -1) == pytest.approx(0.0003)
    assert calc_realized_pnl(1.0001, 1.0004, -1) == pytest.approx(-0.0003)

    # Test invalid direction
    with pytest.raises(ValueError, match="direction must be \\+1 \\(long\\) or -1 \\(short\\)"):
        calc_realized_pnl(1.0, 1.1, 0)


def test_drawdown_penalty():
    """Test drawdown penalty calculation."""
    # Test empty series
    assert drawdown_penalty(pd.Series([]), 1.0) == 0.0
    assert drawdown_penalty(pd.Series([1.0]), 1.0) == 0.0

    # Test no drawdown
    equity = pd.Series([1.0, 1.1, 1.2])
    assert drawdown_penalty(equity, 1.0) == 0.0

    # Test drawdown
    equity = pd.Series([1.0, 1.1, 1.05])
    assert drawdown_penalty(equity, 1.0) == pytest.approx(-0.05)

    # Test drawdown with scaling
    equity = pd.Series([1.0, 1.1, 1.05])
    assert drawdown_penalty(equity, 2.0) == pytest.approx(-0.1)

    # Test multiple drawdowns
    equity = pd.Series([1.0, 1.1, 1.05, 1.15, 1.0])
    assert drawdown_penalty(equity, 1.0) == pytest.approx(-0.15) 