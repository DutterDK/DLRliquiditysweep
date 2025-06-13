"""
Train a PPO agent on the liquidity sweep environment.
"""

import argparse
import yaml
from pathlib import Path

import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import TensorBoardCallback

from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load and slice data
    df = load_tick_csv(cfg["env"]["data_file"],
                      to_seconds=cfg["env"]["bar_seconds"])
    
    # Convert string dates to datetime
    train_start = pd.to_datetime(cfg["env"]["train_start"])
    train_end = pd.to_datetime(cfg["env"]["train_end"])
    
    # Slice training data
    train_mask = (df.index >= train_start) & (df.index <= train_end)
    df_train = df.loc[train_mask]
    
    print(f"\nTraining on {len(df_train):,} bars from {train_start.date()} to {train_end.date()}")
    
    # Create environment
    env = LiquiditySweepEnv(
        df_train,
        lambda_dd=cfg["env"]["lambda_dd"],
        commission=cfg["env"]["commission"]
    )
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=cfg["ppo"]["tensorboard_log"],
        **{k: v for k, v in cfg["ppo"].items() if k != "tensorboard_log"}
    )
    
    # Train
    model.learn(
        total_timesteps=cfg["ppo"]["total_timesteps"],
        callback=TensorBoardCallback()
    )
    
    # Save model
    model_path = Path("models/ppo_liquidity_sweep")
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main() 