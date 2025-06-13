"""Tests for the liquidity sweep environment."""

import numpy as np
import pandas as pd
import pytest

from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="1s", tz="UTC")
    data = {
        "bid": np.linspace(1.0, 1.1, 10),
        "ask": np.linspace(1.002, 1.102, 10),
        "volume": np.ones(10),
    }
    df = pd.DataFrame(data, index=dates)
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    return df


def test_env_init(sample_data):
    """Test environment initialization."""
    env = LiquiditySweepEnv(sample_data)
    assert env.action_space.n == 3
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    assert env.current_step == 0
    assert env.position == 0
    assert env.entry_price == 0.0
    assert obs.shape == env.observation_space.shape


def test_env_reset(sample_data):
    """Test environment reset."""
    env = LiquiditySweepEnv(sample_data)
    env.current_step = 5  # Move forward
    obs, info = env.reset()
    assert env.current_step == 0
    assert env.observation_space.contains(obs)
    assert env.position == 0
    assert env.entry_price == 0.0


def test_env_step(sample_data):
    """Test environment step logic."""
    env = LiquiditySweepEnv(sample_data)
    obs, info = env.reset()
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    for _ in range(len(sample_data)):
        obs, reward, terminated, truncated, info = env.step(action)
    assert terminated or truncated


def test_env_termination(sample_data):
    """Test environment termination."""
    env = LiquiditySweepEnv(sample_data)
    for _ in range(len(sample_data)):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated:
            break
    assert terminated


def test_env_invalid_action(sample_data):
    """Test invalid action handling."""
    env = LiquiditySweepEnv(sample_data)
    with pytest.raises(AssertionError, match="invalid action"):
        env.step(3)  # invalid action


def _dummy_df(rows=10):
    """Create a dummy DataFrame for smoke testing."""
    dates = pd.date_range("2024-01-01", periods=rows, freq="1s", tz="UTC")
    data = {
        "mid": [1.0000 + i * 0.0001 for i in range(rows)],
        "bid": [1.0000 + i * 0.0001 - 0.0001 for i in range(rows)],
        "ask": [1.0000 + i * 0.0001 + 0.0001 for i in range(rows)],
        "spread": [0.0002] * rows,
        "volume": [1] * rows,
    }
    return pd.DataFrame(data, index=dates)


def test_env_runs():
    """Smoke test to verify basic environment functionality."""
    env = LiquiditySweepEnv(_dummy_df(12))
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    for _ in range(12):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated:
            break
    assert terminated or truncated


def test_env_close_long_position(sample_data):
    """Test closing a long position."""
    env = LiquiditySweepEnv(sample_data, commission=0.00005)
    obs, _ = env.reset()
    # Step 1: Open long position
    obs, _, _, _, _ = env.step(1)
    # Step 2: Hold position for the rest of the data
    for _ in range(len(sample_data) - 2):
        obs, _, _, _, _ = env.step(0)
    # Step 3: Close long position
    obs, reward, done, truncated, info = env.step(2)
    assert env.position == 0  # Back to flat
    # Calculate expected reward
    entry_ask = sample_data.iloc[0]["ask"]
    exit_bid = sample_data.iloc[-1]["bid"]
    expected_reward = (
        exit_bid - entry_ask - 2 * env.commission
    )  # Commission on entry and exit
    assert reward == pytest.approx(expected_reward, abs=1e-4)
