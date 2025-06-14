from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv
from stable_baselines3 import PPO
import pandas as pd, pathlib, warnings
warnings.filterwarnings("ignore")
df = pd.DataFrame(
    {"bid":[1]*300,"ask":[1.0002]*300,"mid":[1.0001]*300,
     "spread":[0.0002]*300,"volume":[1]*300},
    index=pd.date_range("2025-01-01", periods=300, freq="1s")
)
env = LiquiditySweepEnv(df)
model = PPO("MlpPolicy", env, n_steps=256, batch_size=256, learning_rate=3e-4, verbose=0)
model.learn(10_000)
pathlib.Path("tests/assets").mkdir(exist_ok=True)
model.save("tests/assets/model_14f") 