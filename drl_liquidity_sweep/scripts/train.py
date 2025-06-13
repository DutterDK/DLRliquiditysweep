import yaml
import os
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv
from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.utils.callbacks import TradingMetricsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(cfg):
    """Create and return the trading environment."""
    data = load_tick_csv(cfg["env"]["data_file"], to_seconds=cfg["env"]["bar_seconds"])
    env = LiquiditySweepEnv(
        data,
        lambda_dd=cfg["env"]["lambda_dd"],
        commission=cfg["env"]["commission"],
        reset_on_day_change=cfg["env"].get("reset_on_day_change", True),
    )
    return env


def main(config_path: str):
    """Train a PPO agent on the trading environment."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Create necessary directories
    log_dir = cfg["misc"]["log_dir"]
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create environment and model
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
        tensorboard_log=log_dir,
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["misc"].get("save_freq", 10000),
        save_path=model_dir,
        name_prefix="ppo_liquidity_sweep",
    )
    trading_callback = TradingMetricsCallback()

    # Train the model
    model.learn(
        total_timesteps=cfg["misc"]["total_timesteps"],
        callback=[checkpoint_callback, trading_callback],
    )

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)

    # Print instructions for viewing metrics
    print(f"\nTraining completed! View metrics with:")
    print(f"tensorboard --logdir={log_dir}")

    # Run a test episode
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
    print(f"Final equity: {env.equity:.4f}")


if __name__ == "__main__":
    main("drl_liquidity_sweep/config/default_config.yaml")
