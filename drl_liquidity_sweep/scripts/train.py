import os
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv
from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.utils.callbacks import TradingMetricsCallback, ActionHistogram, CheckpointEveryStep


class PrintStats(BaseCallback):
    def _on_step(self):
        if self.n_calls % 10000 == 0:
            acts = np.array(self.training_env.envs[0].debug_actions[-10000:])
            print("action dist", np.bincount(acts, minlength=3))
        return True


def train(config_path: str):
    """Train the agent using the specified configuration."""
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load and prepare data
    data = load_tick_csv(
        config["env"]["data_file"],
        to_seconds=config["env"]["bar_seconds"]
    )
    
    # Create environment
    env = LiquiditySweepEnv(
        data,
        commission=config["env"].get("commission", 0.0),
        lambda_dd=config["env"].get("lambda_dd", 0.0),
        max_position=config["env"].get("max_position", 1),
        max_drawdown=config["env"].get("max_drawdown", 1.0),
        reward_scale=config["env"].get("reward_scale", 1.0),
        reset_on_day_change=config["env"].get("reset_on_day_change", True)
    )
    
    # Wrap with DummyVecEnv and VecNormalize
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Create model
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"].get("n_epochs", 10),
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"].get("ent_coef", 0.0),
        verbose=1,
        tensorboard_log=config["misc"]["log_dir"]
    )
    
    # Create callbacks
    callbacks = [
        TradingMetricsCallback(),
        PrintStats(),
        ActionHistogram(freq=50_000),
        CheckpointEveryStep(save_freq=config["misc"]["save_every_steps"]),
    ]
    
    # Train model
    model.learn(
        total_timesteps=config["ppo"]["total_timesteps"],
        callback=callbacks
    )
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(os.path.join("models", "ppo_liquidity_sweep"))
    
    print("\nTraining completed! View metrics with:")
    print(f"tensorboard --logdir={config['misc']['log_dir']}")
    print(f"Final equity: {env.equity:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    train(args.config)
