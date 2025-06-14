import yaml, numpy as np, torch, os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv

CFG = yaml.safe_load(Path("config/exp_2025-long.yaml").read_text())
N_ENVS = 56                         # one per core (adjust if RAM hits limit)
N_STEPS = 4096 // N_ENVS           # so total batch ≈ 4096
TOTAL_STEPS = 5_000_000

# 1. Load + resample once in the parent
print("Loading and resampling data...")
df = load_tick_csv(CFG["env"]["data_file"], to_seconds=CFG["env"]["bar_seconds"])
train_mask = (df.index >= CFG["env"]["train_start"]) & (df.index <= CFG["env"]["train_end"])
df_train = df.loc[train_mask]

print(f"Training on {len(df_train)} bars with {N_ENVS} parallel environments.")

def make_env(start_idx):
    def _init():
        env = LiquiditySweepEnv(
            df_train.iloc[start_idx:].append(df_train.iloc[:start_idx]),
            lambda_dd=CFG["env"]["lambda_dd"],
            commission=CFG["env"]["commission"],
        )
        return env
    return _init

# 2. Build the vector env pool
starts = np.linspace(0, len(df_train) - 1, N_ENVS, dtype=int)
vec_env = SubprocVecEnv([make_env(s) for s in starts])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

# 3. PPO on CPU
print("Configuring PyTorch for multi-threaded CPU training...")
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "56")))

print("Starting PPO training...")
model = PPO(
    "MlpPolicy", vec_env,
    n_steps=N_STEPS,
    batch_size=N_STEPS * N_ENVS,
    learning_rate=1e-4,
    gamma=0.997,
    gae_lambda=0.95,
    ent_coef=0.001,
    clip_range=0.2,
    verbose=1,
    tensorboard_log=CFG["misc"]["log_dir"]
)

model.learn(TOTAL_STEPS)
model.save("models/ppo_5m_cpu_vec")
vec_env.save("models/vecnorm_cpu_vec.pkl")
print("Training complete. Model and VecNormalize state saved.") 