"""Test drawdown penalty integration in the environment."""
import pytest
import pandas as pd
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def test_drawdown_penalty():
    """Test drawdown penalty calculation in environment."""
    # Create test data with specific bid/ask prices
    data = pd.DataFrame({
        'mid': [1.0001, 1.0000],
        'bid': [1.0000, 0.9999],
        'ask': [1.0002, 1.0001]
    })

    # Initialize environment with high lambda for clear penalty
    env = LiquiditySweepEnv(data, lambda_dd=1000.0)

    # Reset environment
    obs, _ = env.reset()

    # Step 1: Enter long position at 1.0002 (ask)
    obs, reward, done, truncated, info = env.step(1)
    assert env.position == 1
    assert env.entry_price == 1.0002
    assert reward == 0.0  # No reward yet as we just entered the position

    # Step 2: Close long position at 1.0000 (bid) -> loss = -0.0002
    obs, reward, done, truncated, info = env.step(2)
    assert env.position == 0
    # Expected reward = realized P/L (-0.0002) + drawdown penalty (-0.3003)
    assert reward == pytest.approx(-0.3003, 1e-4)
