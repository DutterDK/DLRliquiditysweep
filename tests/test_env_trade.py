"""Tests for trading logic."""
import pandas as pd
import numpy as np
import pytest
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def test_long_trading_logic():
    """Test long trading logic."""
    # Create test data
    data = pd.DataFrame(
        {
            "mid": [1.0000, 1.0005, 1.0005],
            "bid": [0.9999, 1.0004, 1.0004],
            "ask": [1.0001, 1.0006, 1.0006],
            "volume": [1, 1, 1],
        }
    )

    # Initialize environment
    env = LiquiditySweepEnv(data, lambda_dd=1.0, commission=0.00005)

    # Reset environment
    obs, _ = env.reset()

    # Step 1: Enter long position
    obs, reward, done, truncated, info = env.step(1)  # Long at 1.0001
    assert env.position == 1
    assert env.entry_price == 1.0001
    assert reward == 0.0  # No reward yet as we just entered the position

    # Step 2: Hold position
    obs, reward, done, truncated, info = env.step(0)  # Hold
    assert env.position == 1  # Still long
    assert reward == pytest.approx(0.0003)  # Unrealized P/L: 1.0004 - 1.0001 = 0.0003

    # Step 3: Close position
    obs, reward, done, truncated, info = env.step(2)  # Close long
    assert env.position == 0  # Back to flat
    assert reward == pytest.approx(0.0003 - 0.00005)  # Realized P/L minus commission


def _dummy_df():
    return pd.DataFrame(
        {
            # time index is optional for this test
            "bid": [0.9999, 1.0004, 1.0004],
            "ask": [1.0001, 1.0006, 1.0006],
            "mid": [1.0000, 1.0005, 1.0005],
            "spread": [0.0002, 0.0002, 0.0002],
            "volume": [1, 1, 1],
        }
    )


def test_short_trading_logic():
    df = _dummy_df()
    env = LiquiditySweepEnv(df, commission=0.00005, lambda_dd=0)
    env.reset()

    # Step 1: open short
    env.step(2)

    # Expected unrealised P/L on hold
    exp_hold = env.entry_price - df.loc[1, "ask"]

    # Step 2: hold
    _, r_hold, _, _, _ = env.step(0)
    assert r_hold == pytest.approx(exp_hold, abs=1e-6)

    # Expected realised P/L on close (add commission)
    exp_close = env.entry_price - df.loc[2, "ask"] - env.commission

    # Step 3: close short
    _, r_close, _, _, _ = env.step(1)
    assert r_close == pytest.approx(exp_close, abs=1e-6)

    # Position should be flat
    assert env.position == 0
