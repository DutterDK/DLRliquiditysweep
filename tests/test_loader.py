from pathlib import Path
from drl_liquidity_sweep.data.loader import load_tick_csv
from datetime import datetime


def test_load_tick_csv(tmp_path: Path):
    csv = tmp_path / "ticks.csv"
    csv.write_text(
        "time,bid,ask,volume\n"
        "2025-01-01 00:00:00.000,1.0000,1.0002,1\n"
        "2025-01-01 00:00:00.500,1.0001,1.0003,2\n"
    )
    df = load_tick_csv(csv, to_seconds=1)
    assert len(df) == 1
    row = df.iloc[0]
    assert abs(row.mid - 1.00015) < 1e-6
    assert abs(row.spread - 0.0002) < 1e-6
    assert str(df.index.tz) == "UTC"
