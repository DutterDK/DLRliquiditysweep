import os
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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
        config["data"]["file_path"],
        to_seconds=config["data"]["bar_size"]
    )
    
    # Create environment
    env = LiquiditySweepEnv(
        data,
        commission=config["env"]["commission"],
        lambda_dd=config["env"]["lambda_dd"],
        max_position=config["env"]["max_position"],
        max_drawdown=config["env"]["max_drawdown"],
        reward_scale=config["env"]["reward_scale"],
        reset_on_day_change=config["env"]["reset_on_day_change"]
    )
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["train"]["learning_rate"],
        n_steps=config["train"]["n_steps"],
        batch_size=config["train"]["batch_size"],
        n_epochs=config["train"]["n_epochs"],
        gamma=config["train"]["gamma"],
        gae_lambda=config["train"]["gae_lambda"],
        clip_range=config["train"]["clip_range"],
        verbose=1,
        tensorboard_log=config["train"]["tensorboard_log"]
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
        total_timesteps=config["train"]["total_timesteps"],
        callback=callbacks
    )
    
    # Save model
    os.makedirs(config["train"]["model_dir"], exist_ok=True)
    model.save(os.path.join(config["train"]["model_dir"], config["train"]["model_name"]))
    
    print("\nTraining completed! View metrics with:")
    print(f"tensorboard --logdir={config['train']['tensorboard_log']}")
    print(f"Final equity: {env.equity:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    train(args.config)
