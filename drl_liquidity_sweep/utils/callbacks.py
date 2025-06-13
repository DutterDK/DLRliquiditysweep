from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np, os, torch


class TradingMetricsCallback(BaseCallback):
    """Custom callback for logging trading-specific metrics to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_equities = []
        self.episode_drawdowns = []

    def _on_step(self) -> bool:
        # Get the environment
        env = self.training_env.envs[0]

        # Log metrics at the end of each episode
        if self.locals.get("dones", [False])[0]:
            # Calculate episode metrics
            episode_return = env.equity - env.equity_history[0]
            episode_length = len(env.equity_history)
            max_drawdown = np.min(env.equity_history) - np.max(env.equity_history)

            # Store metrics
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            self.episode_equities.append(env.equity)
            self.episode_drawdowns.append(max_drawdown)

            # Log to TensorBoard
            self.logger.record("trading/episode_return", episode_return)
            self.logger.record("trading/episode_length", episode_length)
            self.logger.record("trading/equity", env.equity)
            self.logger.record("trading/max_drawdown", max_drawdown)
            self.logger.record("trading/position", env.position)

            # Log rolling statistics
            if len(self.episode_returns) > 1:
                self.logger.record(
                    "trading/mean_return", np.mean(self.episode_returns[-100:])
                )
                self.logger.record(
                    "trading/std_return", np.std(self.episode_returns[-100:])
                )
                self.logger.record(
                    "trading/mean_drawdown", np.mean(self.episode_drawdowns[-100:])
                )

            # Log market metrics if available
            if hasattr(env, "data"):
                current_data = env.data.iloc[env.current_step - 1]
                self.logger.record("market/spread", current_data.get("spread", 0.0))
                self.logger.record("market/volume", current_data.get("volume", 0.0))
                self.logger.record("market/mid_price", current_data.get("mid", 0.0))

        return True


class ActionHistogram(BaseCallback):
    def __init__(self, freq=10_000):
        super().__init__()
        self.freq = freq
        self.hist = np.zeros(3, dtype=int)

    def _on_step(self):
        actions = self.locals["actions"]
        for a in actions:
            self.hist[a] += 1
        if self.num_timesteps % self.freq == 0:
            print(f"[{self.num_timesteps:,}] action hist {self.hist}")
            self.hist[:] = 0
        return True


class CheckpointEveryStep(BaseCallback):
    def __init__(self, save_freq, save_path="models"):
        super().__init__()
        self.save_freq, self.save_path = save_freq, save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            fname = f"{self.save_path}/ppo_{self.num_timesteps}.zip"
            self.model.save(fname)
            torch.save(self.training_env.get_attr("equity"), f"{fname}.equity.pt")
            print("   ✔ checkpoint", fname)
        return True
