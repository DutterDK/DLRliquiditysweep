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
    assert env.observation_space.shape == (10,)
    assert env.t == 0
    assert env.position == 0
    assert env.entry_price is None


def test_env_reset(sample_data):
    """Test environment reset."""
    env = LiquiditySweepEnv(sample_data)
    env.t = 5  # Move forward
    obs, info = env.reset()
    assert env.t == 0
    assert env.position == 0
    assert env.entry_price is None
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10,)
    assert isinstance(info, dict)


def test_env_step(sample_data):
    """Test environment step."""
    env = LiquiditySweepEnv(sample_data)
    obs, reward, terminated, truncated, info = env.step(0)  # hold
    assert env.t == 1
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10,)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert isinstance(info, dict)


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
    assert len(obs) == 10
    for _ in range(12):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert reward == 0.0
        if terminated:
            break
    assert terminated
