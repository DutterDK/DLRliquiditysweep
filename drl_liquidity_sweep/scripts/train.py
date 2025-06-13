import yaml
import os
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv
from drl_liquidity_sweep.data.loader import load_tick_csv
from stable_baselines3 import PPO


def make_env(cfg):
    """Create and return the trading environment."""
    data = load_tick_csv(
        cfg["env"]["data_file"], to_seconds=cfg["env"]["bar_seconds"]
    )
    env = LiquiditySweepEnv(
        data,
        lambda_dd=cfg["env"]["lambda_dd"],
        commission=cfg["env"]["commission"],
    )
    return env


def main(config_path: str):
    """Train a PPO agent on the trading environment."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["misc"]["log_dir"], exist_ok=True)
    env = make_env(cfg)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg["ppo"]["learning_rate"],
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        n_epochs=cfg["ppo"]["n_epochs"],
        gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        clip_range=cfg["ppo"]["clip_range"],
        verbose=1,
        tensorboard_log=cfg["misc"]["log_dir"],
    )
    model.learn(total_timesteps=cfg["misc"]["total_timesteps"])
    model.save("models/ppo_liquidity_sweep")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
    print(f"Final equity: {env.equity:.4f}")


if __name__ == "__main__":
    main("drl_liquidity_sweep/config/default_config.yaml")
