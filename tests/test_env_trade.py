import pytest
import pandas as pd
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def test_long_trading_logic():
    """Test long trading logic."""
    # Create test data
    data = pd.DataFrame(
        {
            "mid": [1.0000, 1.0005, 1.0005],
            "bid": [0.9999, 1.0004, 1.0004],
            "ask": [1.0001, 1.0006, 1.0006],
            "volume": [1, 1, 1],  # Add volume column
        }
    )

    # Initialize environment
    env = LiquiditySweepEnv(data, lambda_dd=1.0)

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
    assert reward == 0.0  # No reward for holding

    # Step 3: Close long position
    obs, reward, done, truncated, info = env.step(2)  # Close long at 1.0004
    assert env.position == 0  # Back to flat
    # Expected reward = realized P/L (0.0003) + drawdown penalty (0.0)
    assert reward == pytest.approx(0.0003, 1e-4)


def test_short_trading_logic():
    """Test short trading logic."""
    # Create test data
    data = pd.DataFrame(
        {
            "mid": [1.0000, 1.0005, 1.0005],
            "bid": [0.9999, 1.0004, 1.0004],
            "ask": [1.0001, 1.0006, 1.0006],
            "volume": [1, 1, 1],  # Add volume column
        }
    )

    # Initialize environment
    env = LiquiditySweepEnv(data, lambda_dd=1.0)

    # Reset environment
    obs, _ = env.reset()

    # Step 1: Enter short position
    obs, reward, done, truncated, info = env.step(2)  # Short at 0.9999
    assert env.position == -1
    assert env.entry_price == 0.9999
    assert reward == 0.0  # No reward yet as we just entered the position

    # Step 2: Hold position
    obs, reward, done, truncated, info = env.step(0)  # Hold
    assert env.position == -1  # Still short
    assert reward == 0.0  # No reward for holding

    # Step 3: Close short position
    obs, reward, done, truncated, info = env.step(1)  # Close short at 1.0006
    assert env.position == 0  # Back to flat
    # Expected reward = realized P/L (-0.0007) + drawdown penalty (-0.0007)
    assert reward == pytest.approx(-0.0014, 1e-4)
