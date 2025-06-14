"""Tests for equity history tracking."""
import pandas as pd
import numpy as np
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def dummy_two_bar_df():
    """Create a simple 2-bar DataFrame for testing."""
    data = {
        "bid": [100.0, 101.0],
        "ask": [100.5, 101.5],
        "volume": [1000, 1000],
        "mid": [100.25, 101.25],
    }
    return pd.DataFrame(data)


def test_equity_history_updates():
    """Test that equity history is updated correctly after each step."""
    # Create environment with dummy data
    env = LiquiditySweepEnv(dummy_two_bar_df())

    # Reset environment
    obs, _ = env.reset()

    # Take a long position
    obs, reward, terminated, truncated, info = env.step(1)

    # Check that equity history was updated
    assert len(env.equity_history) == 2  # Initial + after step
    assert env.equity_history[-1] == env.equity  # Last entry matches current equity

    # Take another step
    obs, reward, terminated, truncated, info = env.step(0)

    # Check that equity history was updated again
    assert len(env.equity_history) == 3  # Initial + 2 steps
    assert env.equity_history[-1] == env.equity  # Last entry matches current equity

    # Check that we get terminated at the end
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated  # Should be terminated at end of data
