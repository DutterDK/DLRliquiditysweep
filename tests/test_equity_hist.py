import pandas as pd
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def test_equity_history_updates():
    df = pd.DataFrame(
        {
            "bid": [1, 1],
            "ask": [1.0002, 1.0002],
            "mid": [1.0001, 1.0001],
            "spread": [0.0002, 0.0002],
            "volume": [1, 1],
        }
    )
    env = LiquiditySweepEnv(df)
    env.reset()
    env.step(0)
    assert len(env.equity_history) == 2
