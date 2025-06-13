import pytest
from drl_liquidity_sweep.utils.rewards import (
    calc_realized_pnl, drawdown_penalty
)


def test_calc_realized_pnl():
    """Test P/L calculation for long and short positions."""
    # Long position, profit
    assert calc_realized_pnl(1.0, 1.1, 1) == pytest.approx(0.1)
    # Long position, loss
    assert calc_realized_pnl(1.1, 1.0, 1) == pytest.approx(-0.1)
    # Short position, profit
    assert calc_realized_pnl(1.1, 1.0, -1) == pytest.approx(0.1)
    # Short position, loss
    assert calc_realized_pnl(1.0, 1.1, -1) == pytest.approx(-0.1)
    # Invalid direction
    with pytest.raises(ValueError):
        calc_realized_pnl(1.0, 1.1, 0)


def test_drawdown_penalty():
    """Test drawdown penalty calculation."""
    # No drawdown
    eq = [1, 2, 3, 4, 5]
    assert drawdown_penalty(eq, 1.0) == pytest.approx(0.0)
    # Single drawdown
    eq = [1, 2, 1, 2, 1]
    penalty = drawdown_penalty(eq, 1.0)
    assert penalty < 0
    # Multiple drawdowns
    eq = [1, 0.5, 1, 0.5, 1]
    penalty = drawdown_penalty(eq, 1.0)
    assert penalty <= 0
