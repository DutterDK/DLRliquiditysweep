import pandas as pd
import pytest
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv

def test_daily_reset():
    """Test that episodes terminate at day changes when reset_on_day_change is True."""
    # Create data spanning two days
    data = pd.DataFrame({
        'bid': [1.0] * 5 + [1.1] * 5,  # Different prices for different days
        'ask': [1.002] * 5 + [1.102] * 5,
        'volume': [1] * 10,
        'mid': [1.001] * 5 + [1.101] * 5,  # Different mid prices for different days
    }, index=pd.date_range(
        start='2024-01-01 00:00:00',
        end='2024-01-02 00:00:00',
        periods=10
    ))

    # Test with reset_on_day_change=True (default)
    env = LiquiditySweepEnv(data)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        obs, _, terminated, truncated, _ = env.step(0)  # Hold action
        done = terminated or truncated
        steps += 1
    assert terminated  # Should terminate at day change
    assert steps < len(data)  # Should not use all data
    assert env.current_step < len(data)  # Should have remaining data for next episode

    # Test with reset_on_day_change=False
    env = LiquiditySweepEnv(data, reset_on_day_change=False)
    obs, _ = env.reset()
    done = False
    steps = 0
    last_terminated = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(0)  # Hold action
        done = terminated or truncated
        steps += 1
        last_terminated = terminated
    # Should only terminate at the end of data
    assert last_terminated  # Should terminate at end of data
    assert steps == len(data)  # Should use all data 