import yaml, pathlib, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv

CFG = yaml.safe_load(pathlib.Path("config/exp_2025-long.yaml").read_text())
df  = load_tick_csv(CFG["env"]["data_file"], to_seconds=CFG["env"]["bar_seconds"])

def make_env():
    allowed = ["lambda_dd", "commission", "reset_on_day_change", "max_position", "max_drawdown", "reward_scale"]
    env_kwargs = {k: v for k, v in CFG["env"].items() if k in allowed}
    return LiquiditySweepEnv(df, **env_kwargs)

venv = DummyVecEnv([make_env])
venv = VecNormalize(venv, training=False, norm_obs=True, norm_reward=False)

model = PPO.load("models/ppo_5000000.zip", env=venv)  # loads stats into venv
venv.save("models/vecnorm.pkl")                       # overwrite old file
print("✔ vecnorm.pkl written, shape", venv.obs_rms.mean.shape) 