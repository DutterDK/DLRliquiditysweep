import argparse
import os
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv
from drl_liquidity_sweep.data.loader import load_tick_csv
from stable_baselines3 import PPO

def evaluate_model(env, model, plot=False):
    """Evaluate a trained model on the given environment.

    Args:
        env: The trading environment
        model: The trained model
        plot: Whether to plot the equity curve (optional)
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0
    equity = [0.0]
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        equity.append(equity[-1] + reward)
        done = terminated or truncated
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(equity)
        plt.title("Equity curve")
        plt.show()
    return total_reward

def main(cfg_path="config/default_config.yaml",
         model_path="models/ppo_liquidity_sweep.zip"):
    """Backward-compat thin wrapper expected by tests/test_evaluate.py"""
    from pathlib import Path
    import yaml
    from drl_liquidity_sweep.data.loader import load_tick_csv
    from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv
    from stable_baselines3 import PPO

    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df = load_tick_csv(cfg["env"]["data_file"], to_seconds=cfg["env"]["bar_seconds"])
    env = LiquiditySweepEnv(
        df,
        lambda_dd=cfg["env"]["lambda_dd"],
        commission=cfg["env"]["commission"],
    )
    model = PPO.load(model_path, env=env)
    evaluate_model(env, model, plot=True)

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
