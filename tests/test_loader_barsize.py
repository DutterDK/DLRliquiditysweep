import pytest
from pathlib import Path
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


def test_bar_size_resample(tmp_path):
    csv = tmp_path / "ticks.csv"

    rows = [
        # three DISTINCT seconds so resampling can create 3×1-s bars
        "2025-01-01 00:00:00.000,1.0000,1.0002,1",
        "2025-01-01 00:00:01.000,1.0001,1.0003,1",
        "2025-01-01 00:00:02.000,1.0002,1.0004,1",
    ]
    csv.write_text("time,bid,ask,volume\n" + "\n".join(rows))

    df_1s = load_tick_csv(csv, to_seconds=1)
    df_5s = load_tick_csv(csv, to_seconds=5)

    assert len(df_1s) == 3         # 3 seconds → 3 bars
    assert len(df_5s) == 1         # 5-second resample collapses to 1 bar
