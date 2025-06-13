import gymnasium as gym
import yaml
import os
from stable_baselines3 import PPO
from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def make_env(cfg):
    """Create and return the trading environment."""
    df = load_tick_csv(
        cfg["env"]["data_file"],
        to_seconds=cfg["env"]["bar_seconds"] > 0
    )
    return LiquiditySweepEnv(
        data=df,
        window_size=cfg["env"]["bar_seconds"],
        lambda_dd=cfg["env"]["lambda_dd"]
    )


def main(cfg_path="config/default_config.yaml"):
    """Main training function."""
    # Load config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Create environment
    env = gym.wrappers.TimeLimit(make_env(cfg), max_episode_steps=2000)
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        learning_rate=cfg["ppo"]["learning_rate"],
        gamma=cfg["ppo"]["gamma"],
        clip_range=cfg["ppo"]["clip_range"],
        verbose=1,
        seed=cfg["misc"]["seed"],
    )
    # Train model
    model.learn(cfg["ppo"]["total_timesteps"])
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_liquidity_sweep")


if __name__ == "__main__":
    main()
