"""Test data loading utilities."""
import pytest
from pathlib import Path
from drl_liquidity_sweep.data.loader import load_tick_csv


def test_load_tick_csv(tmp_path: Path):
    """Test loading tick data from CSV."""
    csv = tmp_path / "ticks.csv"
    csv.write_text(
        "time,bid,ask,volume\n"
        "2025-01-01 00:00:00.000,1.0000,1.0002,1\n"
        "2025-01-01 00:00:00.500,1.0001,1.0003,2\n"
    )
    df = load_tick_csv(csv)
    assert list(df.columns) == ["bid", "ask", "volume", "mid", "spread"]
    assert len(df) == 2
    assert df.index.name == "time"
