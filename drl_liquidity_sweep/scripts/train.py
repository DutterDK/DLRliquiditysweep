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


def main():
    # Load configuration
    with open("config/default_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    # Create environment
    env = make_env(cfg)
    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, **cfg["ppo"])
    # Train the model
    model.learn(total_timesteps=cfg["misc"]["total_timesteps"])
    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_liquidity_sweep")
    # Evaluate: print final equity
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    print("Final equity:", env.equity)


if __name__ == "__main__":
    main()
