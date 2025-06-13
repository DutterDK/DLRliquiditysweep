import pytest
import pandas as pd
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def test_daily_reset():
    """Test that episodes terminate at the end of each trading day."""
    # Create data spanning two days
    data = pd.DataFrame(
        {
            "bid": [1.0] * 5 + [1.1] * 5,  # Different prices for different days
            "ask": [1.002] * 5 + [1.102] * 5,
            "volume": [1] * 10,
            "mid": [1.001] * 5 + [1.101] * 5,  # Different mid prices for different days
        },
        index=pd.date_range(
            start="2024-01-01 00:00:00", end="2024-01-02 00:00:00", periods=10
        ),
    )
    env = LiquiditySweepEnv(data)
    # First episode
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        obs, _, terminated, truncated, _ = env.step(0)  # Hold action
        done = terminated or truncated
        steps += 1
    # Should terminate at end of first day
    assert terminated  # Episode should be terminated, not truncated
    assert steps < len(data)  # Should terminate before using all data
    assert env.current_step < len(data)  # Should have remaining data for next episode
