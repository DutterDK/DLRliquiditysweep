import pandas as pd
import pytest
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv


def _df():
    return pd.DataFrame(
        {
            "mid": [1.0000, 1.0006],
            "bid": [0.9999, 1.0005],
            "ask": [1.0001, 1.0007],
            "spread": [0.0002, 0.0002],
            "volume": [1, 1],
        }
    )


def test_commission_deducted():
    env = LiquiditySweepEnv(_df(), commission=0.00005)
    env.reset()
    _, _, _, _, _ = env.step(1)  # open long @ 1.0001
    obs, r, _, _, _ = env.step(2)  # close long @ 1.0005 → gross 0.0004
    assert r == pytest.approx(0.00035)
