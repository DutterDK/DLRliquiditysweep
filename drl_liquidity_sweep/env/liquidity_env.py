from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from drl_liquidity_sweep.utils.rewards import drawdown_penalty

class LiquiditySweepEnv(gym.Env):
    """Trading environment for liquidity sweep strategy."""

    def __init__(
        self,
        data: pd.DataFrame,
        commission: float = 0.0,
        lambda_dd: float = 0.0,
        max_position: int = 1,
        max_drawdown: float = 1.0,
        reward_scale: float = 1.0,
        reset_on_day_change: bool = True,
    ):
        """Initialize environment.

        Args:
            data: DataFrame with columns: bid,ask,volume,mid,spread
            commission: Commission per trade
            lambda_dd: Drawdown penalty coefficient
            max_position: Maximum allowed position size
            max_drawdown: Maximum allowed drawdown before termination
            reward_scale: Scale factor for rewards
            reset_on_day_change: Whether to reset at day changes
        """
        super().__init__()
        self.data = data.copy()
        
        # Add spread column if not present
        if "spread" not in self.data.columns:
            self.data["spread"] = self.data["ask"] - self.data["bid"]
            
        self.commission = commission
        self.lambda_dd = lambda_dd
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.reward_scale = reward_scale
        self.reset_on_day_change = reset_on_day_change

        # Action space: 0=hold, 1=long, 2=short
        self.action_space = spaces.Discrete(3)

        # Observation space: 14 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        # Trading state
        self.position = 0  # Current position (-1=short, 0=flat, 1=long)
        self.entry_price = 0.0  # Entry price of current position
        self.equity = 0.0  # Current equity
        self.equity_history = [0.0]  # History of equity values
        self.current_step = 0  # Current step in data
        self.current_date = None  # Current date
        self.debug_actions = []  # Initialize debug_actions as an empty list
        self.equity_prev = 0.0  # Previous equity

        # Verify required columns
        required_cols = {"bid", "ask", "volume", "mid"}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"Data must contain columns: {required_cols}")

    def reset(self, seed=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.position = 0
        self.entry_price = 0.0
        self.equity = 0.0
        self.equity_history = [0.0]
        self.current_step = 0
        
        # Handle date initialization
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.current_date = self.data.index[0].date()
        else:
            self.current_date = self.data.index[0]
            
        return self._get_observation(), {}

    def step(self, action):
        action = int(action)
        """Execute one time step within the environment."""
        assert action in (0, 1, 2), f"invalid action: {action}"
        self.debug_actions.append(action)  # Append action to debug_actions

        current_data = self.data.iloc[self.current_step]
        bid, ask = current_data["bid"], current_data["ask"]
        reward = 0.0
        terminated = False
        truncated = False
        closing_trade = False

        # --- Position closing logic ---
        if self.position == 1 and action == 2:  # Close long
            closing_trade = True
            reward = bid - self.entry_price  # no commission here
            self.position = 0
            self.entry_price = 0.0
        elif self.position == -1 and action == 1:  # Close short
            closing_trade = True
            reward = self.entry_price - ask  # no commission here
            self.position = 0
            self.entry_price = 0.0
        elif action == 1:  # OPEN long
            if self.position == 0:
                self.position = 1
                self.entry_price = ask
                reward = 0.0  # No reward on entry
        elif action == 2:  # OPEN short
            if self.position == 0:
                self.position = -1
                self.entry_price = bid
                reward = 0.0  # No reward on entry
        else:  # HOLD
            if self.position == 1:  # Long
                reward = bid - self.entry_price
            elif self.position == -1:  # Short
                reward = self.entry_price - ask
            else:  # Flat
                reward = 0.0

        # Subtract commission only if closing trade
        if closing_trade:
            reward -= self.commission

        # Apply drawdown penalty
        dd_pen = drawdown_penalty(
            pd.Series([self.equity_prev, self.equity]), lam=self.lambda_dd
        )
        reward += dd_pen
        self.equity_prev = self.equity
        self.equity += reward
        self.equity_history.append(self.equity)

        # Check max drawdown
        if self.equity < -self.max_drawdown:
            terminated = True

        # Check day change
        if self.reset_on_day_change and self.current_step < len(self.data) - 1:
            if isinstance(self.data.index, pd.DatetimeIndex):
                next_date = self.data.index[self.current_step + 1].date()
                if next_date != self.current_date:
                    terminated = True
                    self.current_date = next_date

        # Check if we've reached the end of the data
        if self.current_step >= len(self.data) - 1:
            terminated = True      # natural task end
            truncated  = False     # not an artificial truncation
            return self._get_observation(), reward, terminated, truncated, {}

        # Increment step counter
        self.current_step += 1
        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        mid = row["mid"]
        spread = row["spread"]
        volume = row["volume"]

        # session time
        ts = self.data.index[self.current_step]
        if isinstance(ts, pd.Timestamp):
            h = ts.hour + ts.minute/60 + ts.second/3600
            hour_sin = np.sin(2*np.pi*h/24)
            hour_cos = np.cos(2*np.pi*h/24)
        else:
            hour_sin = hour_cos = 0.0

        # rolling 30-min high/low
        start = max(0, self.current_step - 1800)
        roll_hi = self.data["mid"].iloc[start:self.current_step+1].max()
        roll_lo = self.data["mid"].iloc[start:self.current_step+1].min()
        dist_hi = roll_hi - mid
        dist_lo = mid - roll_lo

        obs = [
            mid, spread, volume,
            hour_sin, hour_cos,
            self.position,
            self.equity,
            self.entry_price,
            max(self.equity_history, default=0.0),
            min(self.equity_history, default=0.0),
            np.mean(self.equity_history) if self.equity_history else 0.0,
            np.std(self.equity_history)  if self.equity_history else 0.0,
            dist_hi, dist_lo,
        ]
        return np.array(obs, dtype=np.float32)

    def render(self):
        """Render the environment."""
        pass  # Implement if needed

    def close(self):
        """Clean up resources."""
        pass  # Implement if needed

