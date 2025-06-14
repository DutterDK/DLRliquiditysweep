"""
Vectorised PPO training: 16 parallel LiquiditySweepEnv workers
Uses CPU only; adjust N_ENVS if you have more or fewer cores.
"""

import os, yaml, numpy as np, torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv

# ----------  CONFIG  ----------
N_ENVS          = 16                     # change to 32 on Threadripper if RAM allows
TOTAL_TIMESTEPS = 5_000_000
TOTAL_BATCH     = 4_096           # PPO batch size
# ------------------------------

def make_env(df_slice):
    def _thunk():
        # keep worker BLAS single-threaded
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"]     = "1"
        return LiquiditySweepEnv(df_slice)
    return _thunk

def main():
    torch.set_num_threads(os.cpu_count() // 2)
    torch.set_num_interop_threads(4)

    # 2. Load YAML config
    cfg = yaml.safe_load(Path("config/exp_2025-long.yaml").read_text())

    # 3. Load & resample tick data
    df = load_tick_csv(cfg["env"]["data_file"], to_seconds=cfg["env"]["bar_seconds"])

    # 4. Split DataFrame across workers (copy-on-write)
    slices = np.array_split(df, N_ENVS)

    # 5. Build vectorised env
    venv = SubprocVecEnv(
        [make_env(s) for s in slices],
        start_method="spawn"          # Windows-safe
    )
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    # 6. PPO hyper-parameters
    per_env_steps = TOTAL_BATCH // N_ENVS      # e.g. 256 with 16 envs
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps        = per_env_steps,
        batch_size     = TOTAL_BATCH,
        learning_rate  = cfg["ppo"]["learning_rate"],
        gamma          = cfg["ppo"]["gamma"],
        gae_lambda     = cfg["ppo"]["gae_lambda"],
        clip_range     = cfg["ppo"]["clip_range"],
        ent_coef       = cfg["ppo"]["ent_coef"],
        verbose        = 1,
        tensorboard_log= cfg["misc"]["log_dir"],
    )

    # 7. Train
    model.learn(TOTAL_TIMESTEPS)
    model.save("models/ppo_5m_vec")
    venv.save("models/vecnorm_vec.pkl")
    print("✔ 5 M-step training finished")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()          # safe on Windows & ignored on *nix
    main() 