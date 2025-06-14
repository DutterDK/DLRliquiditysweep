"""
Evaluate a trained PPO agent on the liquidity sweep environment.
"""

import yaml
from pathlib import Path

import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO

from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def main():
    # Load config
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)

    # Load and slice data
    df = load_tick_csv(cfg["env"]["data_file"], to_seconds=cfg["env"]["bar_seconds"])

    # Convert string dates to datetime
    test_start = pd.to_datetime(cfg["env"]["test_start"])
    test_end = pd.to_datetime(cfg["env"]["test_end"])

    # Slice test data
    test_mask = (df.index >= test_start) & (df.index <= test_end)
    df_test = df.loc[test_mask]

    # Create environment
    env = LiquiditySweepEnv(
        df_test, lambda_dd=cfg["env"]["lambda_dd"], commission=cfg["env"]["commission"]
    )

    # Load model
    model = PPO.load("models/ppo_liquidity_sweep")

    # Evaluate
    obs, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    # Print results
    print(f"\nTest Results ({test_start.date()} to {test_end.date()}):")
    print(f"Final Equity: {env.equity:.4f}")
    print(f"Max Drawdown: {min(env.equity_history):.4f}")
    print(f"Total Trades: {len([x for x in env.equity_history if x != 0])}")


if __name__ == "__main__":
    main()
