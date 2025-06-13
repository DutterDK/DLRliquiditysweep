import pytest
import pandas as pd
from drl_liquidity_sweep.data.loader import load_tick_csv

def test_bar_size():
    # Create a dummy CSV with 60 rows (0-59 seconds)
    csv = "ticks.csv"
    with open(csv, "w") as f:
        f.write("time,bid,ask,volume\n")
        for i in range(60):
            f.write(f"2025-01-01 00:00:{i:02d},1.0000,1.0002,1\n")
    # Load with to_seconds=1
    df1 = load_tick_csv(csv, to_seconds=1)
    # Load with to_seconds=5
    df5 = load_tick_csv(csv, to_seconds=5)
    # Assert len(df1) ≈ 5× len(df5)
    assert len(df1) == pytest.approx(5 * len(df5), rel=0.1) 