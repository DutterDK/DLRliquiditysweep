from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from drl_liquidity_sweep.utils.rewards import drawdown_penalty


class LiquiditySweepEnv(gym.Env):
    """A simple trading environment for liquidity sweep strategies."""

    def __init__(
        self,
        data: pd.DataFrame,
        lambda_dd: float = 0.0,
        commission: float = 0.0,
        max_position: int = 1,
        max_drawdown: float = 1.0,
        reward_scale: float = 1.0,
        reset_on_day_change: bool = True,
    ):
        """Initialize environment.

        Args:
            data: DataFrame with columns ['bid', 'ask', 'volume', 'mid']
            lambda_dd: Lambda for drawdown penalty calculation
            commission: Trading commission
            max_position: Maximum allowed position size (default: 1)
            max_drawdown: Maximum allowed drawdown (default: 1.0)
            reward_scale: Scale factor for rewards (default: 1.0)
            reset_on_day_change: Whether to terminate episodes at day changes
        """
        super().__init__()
        self.data = data
        self.lambda_dd = lambda_dd
        self.commission = commission
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.reward_scale = reward_scale
        self.reset_on_day_change = reset_on_day_change
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.equity = 0.0
        self.equity_prev = 0.0  # Track previous equity for penalty
        self.equity_history = [0.0]
        self.current_date = None
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # Verify required columns
        required_cols = {"bid", "ask", "volume", "mid"}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"Data must contain columns: {required_cols}")

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state.

        Returns:
            observation: Initial state observation
            info: Additional information
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.equity = 0.0
        self.equity_prev = 0.0  # Track previous equity for penalty
        self.equity_history = [0.0]
        idx = self.data.index
        if isinstance(idx, pd.DatetimeIndex):
            self.current_date = idx[0].date()
        else:
            self.current_date = None
        obs = self._get_obs()
        return obs, {}

    def _get_current_date(self):
        idx = self.data.index[self.current_step]
        if isinstance(idx, pd.Timestamp):
            return idx.date()
        return None

    def _get_obs(self):
        """Get the current observation as a feature vector."""
        idx = self.current_step
        if idx >= len(self.data):
            idx = len(self.data) - 1
        row = self.data.iloc[idx]
        mid = row["mid"]
        if "spread" in row:
            spread = row["spread"]
        elif "ask" in row and "bid" in row:
            spread = row["ask"] - row["bid"]
        else:
            spread = 0.0
        volume = row["volume"]
        if hasattr(row.name, 'hour'):
            hour = row.name.hour + row.name.minute / 60.0 + row.name.second / 3600.0
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
        else:
            hour_sin = 0.0
            hour_cos = 0.0
        obs = [
            mid,
            spread,
            volume,
            hour_sin,
            hour_cos,
            self.position,
            self.entry_price,
            self.equity,
            np.max(self.equity_history),
            np.min(self.equity_history),
            np.mean(self.equity_history),
            np.std(self.equity_history),
        ]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment.

        Args:
            action: 0=hold, 1=buy, 2=sell

        Returns:
            observation: Current state observation
            reward: Reward for the step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        if action not in [0, 1, 2]:
            raise AssertionError("invalid action")
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        if self.current_step >= len(self.data):
            obs = self._get_obs() if len(self.data) > 0 else np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, reward, True, truncated, info
        row = self.data.iloc[self.current_step]
        realised = 0.0
        closing_position = False
        # --- Trading logic ---
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = row["ask"]
            elif self.position == -1:
                # Close short position
                realised = self.entry_price - row["ask"]
                closing_position = True
                self.position = 0
                self.entry_price = 0.0
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = row["bid"]
            elif self.position == 1:
                # Close long position
                realised = row["bid"] - self.entry_price
                closing_position = True
                self.position = 0
                self.entry_price = 0.0

        reward = realised
        if closing_position:
            reward -= self.commission

        # Drawdown penalty BEFORE equity update
        dd_pen = drawdown_penalty(
            pd.Series([self.equity, self.equity + reward]),
            lam=self.lambda_dd
        )
        reward += dd_pen

        self.equity += reward
        self.equity_history.append(self.equity)
        self.equity_prev = self.equity
        self.current_step += 1
        if self.reset_on_day_change and self.current_step < len(self.data):
            next_date = self.data.index[self.current_step]
            if isinstance(next_date, pd.Timestamp):
                next_date = next_date.date()
            if next_date != self.current_date:
                terminated = True
                self.current_date = next_date
        if self.current_step >= len(self.data):
            terminated = True
            obs = self._get_obs() if len(self.data) > 0 else np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, reward, terminated, truncated, info
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        pass  # Implement if needed

    def close(self):
        """Clean up resources."""
        pass  # Implement if needed

