import pytest
import pandas as pd
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv

def test_commission_applied():
    data = pd.DataFrame({
        'mid': [1.0001, 1.0004],
        'bid': [1.0000, 1.0003],
        'ask': [1.0002, 1.0005],
        'volume': [1, 1],
    })
    env = LiquiditySweepEnv(data, commission=0.00005)
    obs, _ = env.reset()
    # Open long
    obs, reward, done, truncated, info = env.step(1)
    # Close long
    obs, reward, done, truncated, info = env.step(2)
    # Realized P/L: entry at ask[0]=1.0002, exit at bid[1]=1.0003
    # Profit = 1.0003 - 1.0002 = 0.0001, minus commission = 0.00005
    assert reward == pytest.approx(0.00005) 