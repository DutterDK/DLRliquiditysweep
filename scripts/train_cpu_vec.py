import os, torch
import numpy as np
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv

torch.set_num_threads(32)            # parent uses 32 BLAS threads
torch.set_num_interop_threads(4)

CFG = yaml.safe_load(Path("config/exp_2025-long.yaml").read_text())
N_ENVS = 32                       # one per core
TOTAL_BATCH = 4096                # keep as before
per_env_steps = TOTAL_BATCH // N_ENVS   # 128 when N_ENVS=32
TOTAL_STEPS = 5_000_000

# 1. Load + resample once in the parent
df = load_tick_csv(CFG["env"]["data_file"], to_seconds=CFG["env"]["bar_seconds"])
train_mask = (df.index >= CFG["env"]["train_start"]) & (df.index <= CFG["env"]["train_end"])
df_train = df.loc[train_mask]

# Slice the DataFrame once (copy-on-write)
slices = np.array_split(df_train, N_ENVS)

def make_env(df_slice):
    def _init():
        env = LiquiditySweepEnv(
            df_slice,
            lambda_dd=CFG["env"]["lambda_dd"],
            commission=CFG["env"]["commission"],
        )
        return env
    return _init

# Use 'spawn' on Windows, 'fork' on Linux
start_method = "spawn" if os.name == "nt" else "fork"
venv = SubprocVecEnv([make_env(s) for s in slices], start_method=start_method)
venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

model = PPO(
    "MlpPolicy",
    venv,
    n_steps        = per_env_steps,
    batch_size     = TOTAL_BATCH,
    learning_rate  = 1e-4,
    gamma          = 0.997,
    gae_lambda     = 0.95,
    ent_coef       = 0.001,
    clip_range     = 0.2,
    verbose        = 1,
    tensorboard_log=CFG["misc"]["log_dir"]
)

model.learn(TOTAL_STEPS)
model.save("models/ppo_5m_cpu_vec")
venv.save("models/vecnorm_cpu_vec.pkl")
print("Training complete. Model and VecNormalize state saved.") 