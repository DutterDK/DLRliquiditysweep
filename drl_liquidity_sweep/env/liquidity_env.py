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
        self.max_equity = 0.0  # Track max equity
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # Verify required columns
        required_cols = {"bid", "ask", "volume", "mid"}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"Data must contain columns: {required_cols}")

    def reset(self, *args, **kwargs):
        """Reset environment to initial state."""
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.equity = 0
        self.max_equity = 0
        idx = self.data.index
        if hasattr(idx, 'tz') or isinstance(idx, pd.DatetimeIndex):
            self.current_date = idx[self.current_step].date()
        else:
            self.current_date = idx[self.current_step]
        obs = self._get_obs()
        info = {}
        return obs, info

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
        """Execute one time step within the environment."""
        # Validate action
        assert action in [0, 1, 2], f"invalid action: {action}"

        current_bid = self.data.iloc[self.current_step]["bid"]
        current_ask = self.data.iloc[self.current_step]["ask"]
        reward = 0.0
        closing_trade = False

        if action == 1:  # agent says "long"
            if self.position == 0:  # open new long
                self.position = +1
                self.entry_price = current_ask
            elif self.position == -1:  # CLOSE short → flat
                closing_trade = True
                reward = self.entry_price - current_ask  # short P/L
                self.position = 0
                self.entry_price = 0.0
        elif action == 2:  # agent says "short"
            if self.position == 0:  # open new short
                self.position = -1
                self.entry_price = current_bid
            elif self.position == +1:  # CLOSE long → flat
                closing_trade = True
                reward = current_bid - self.entry_price  # long P/L
                self.position = 0
                self.entry_price = 0.0
        # action == 0 ("hold")  → do nothing; keep whatever position we already have

        # subtract commission only on a closing trade
        if closing_trade:
            reward -= self.commission

        # Update equity and max equity
        self.equity += reward
        self.max_equity = max(self.max_equity, self.equity)

        # Apply drawdown penalty if enabled
        if self.lambda_dd > 0 and self.max_equity > 0:
            drawdown = (self.max_equity - self.equity) / self.max_equity
            reward -= self.lambda_dd * drawdown

        terminated = False
        truncated = False
        info = {}
        obs = self._get_obs()

        # Calculate next index before any checks
        next_idx = self.current_step + 1

        # Check if we've reached the end of data
        if next_idx >= len(self.data):
            terminated = True
            return obs, reward, terminated, truncated, info

        # Check day change BEFORE advancing pointer
        if self.reset_on_day_change:
            idx = self.data.index
            if next_idx < len(self.data):
                if hasattr(idx, 'tz') or isinstance(idx, pd.DatetimeIndex):
                    next_date = idx[next_idx].date()
                else:
                    next_date = idx[next_idx]
            else:
                next_date = self.current_date
            if next_date != self.current_date:
                terminated = True
                # Reset environment for next day
                self.current_step = next_idx
                self.position = 0
                self.entry_price = 0
                self.equity = 0
                self.max_equity = 0
                if hasattr(idx, 'tz') or isinstance(idx, pd.DatetimeIndex):
                    self.current_date = idx[next_idx].date()
                else:
                    self.current_date = idx[next_idx]
                return obs, reward, terminated, truncated, info

        # Advance step counter
        self.current_step = next_idx

        # Update current date
        idx = self.data.index
        if hasattr(idx, 'tz') or isinstance(idx, pd.DatetimeIndex):
            self.current_date = idx[self.current_step].date()
        else:
            self.current_date = idx[self.current_step]

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        pass  # Implement if needed

    def close(self):
        """Clean up resources."""
        pass  # Implement if needed

    def _execute_action(self, action):
        """Execute trading action and return reward."""
        reward = 0
        current_bid = self.data.iloc[self.current_step]["bid"]
        current_ask = self.data.iloc[self.current_step]["ask"]

        # Handle position changes
        if self.position != 0:  # We have an open position
            if (self.position == 1 and action == 2) or (self.position == -1 and action == 1):
                # Close position
                if self.position == 1:  # Close long
                    reward = (current_bid - self.entry_price) - self.commission
                else:  # Close short
                    reward = (self.entry_price - current_ask) - self.commission
                self.position = 0
                self.entry_price = 0
            else:
                # Hold position
                reward = 0
        else:  # No position
            if action == 1:  # Enter long
                self.position = 1
                self.entry_price = current_ask
                reward = 0  # No commission on entry
            elif action == 2:  # Enter short
                self.position = -1
                self.entry_price = current_bid
                reward = 0  # No commission on entry
            else:  # Hold
                reward = 0

        # Update equity and max equity
        self.equity += reward
        self.max_equity = max(self.max_equity, self.equity)

        # Apply drawdown penalty if enabled
        if self.lambda_dd > 0 and self.max_equity > 0:
            drawdown = (self.max_equity - self.equity) / self.max_equity
            reward -= self.lambda_dd * drawdown

        return reward

