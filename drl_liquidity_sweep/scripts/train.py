import gymnasium as gym
import yaml
import os
from pathlib import Path
from stable_baselines3 import PPO
from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def make_env(cfg):
    """Create and return the trading environment."""
    data = load_tick_csv(cfg["env"]["data_file"], to_seconds=cfg["env"]["bar_seconds"])
    return LiquiditySweepEnv(data)


def main(cfg_path=None):
    if cfg_path is None:
        # Get the directory containing this script
        script_dir = Path(__file__).parent.parent
        cfg_path = script_dir / "config" / "default_config.yaml"

    # Load configuration
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Create logs directory
    os.makedirs(cfg["misc"]["log_dir"], exist_ok=True)

    # Create environment
    env = make_env(cfg)

    # Remove total_timesteps from PPO kwargs if present
    ppo_kwargs = dict(cfg["ppo"])
    total_timesteps = ppo_kwargs.pop("total_timesteps", None)
    ppo_kwargs.pop("tensorboard_log", None)  # Remove tensorboard_log from kwargs
    ppo_kwargs.pop("verbose", None)  # Remove verbose from kwargs

    # Create model with TensorBoard logging
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=cfg["misc"]["log_dir"],
        **ppo_kwargs
    )

    # Train the model
    model.learn(total_timesteps=cfg["misc"].get("total_timesteps", total_timesteps))

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
