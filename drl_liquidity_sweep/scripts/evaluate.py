import yaml
import pandas as pd
import matplotlib.pyplot as plt
from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv
from stable_baselines3 import PPO
from pathlib import Path


def main(
    cfg_path="config/default_config.yaml", model_path="models/ppo_liquidity_sweep.zip"
):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df = load_tick_csv(cfg["env"]["data_file"], to_seconds=cfg["env"]["bar_seconds"])
    env = LiquiditySweepEnv(
        df, lambda_dd=cfg["env"]["lambda_dd"], commission=cfg["env"]["commission"]
    )
    model = PPO.load(model_path, env=env)
    obs, _ = env.reset()
    equity = [0.0]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(int(action))
        equity.append(equity[-1] + reward)
    plt.plot(equity)
    plt.title("Equity curve")
    plt.show()


if __name__ == "__main__":
    main()
