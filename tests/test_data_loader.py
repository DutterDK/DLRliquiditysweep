"""Tests for the data loader module."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from drl_liquidity_sweep.data.loader import load_tick_csv


@pytest.fixture
def sample_tick_data():
    """Create sample tick data for testing."""
    data = {
        "time": [
            "2024-01-01 00:00:00.000",
            "2024-01-01 00:00:01.000",
            "2024-01-01 00:00:02.000",
        ],
        "bid": [1.0000, 1.0001, 1.0002],
        "ask": [1.0002, 1.0003, 1.0004],
        "volume": [1.0, 2.0, 1.0],
    }
    return pd.DataFrame(data)


def test_load_tick_csv_basic(sample_tick_data):
    """Test basic tick data loading."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        sample_tick_data.to_csv(tmp.name, index=False)
        df = load_tick_csv(tmp.name)
        os.unlink(tmp.name)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz == "UTC"
    assert all(col in df.columns for col in ["bid", "ask", "mid", "spread", "volume"])
    assert len(df) == 3


def test_load_tick_csv_resampling(sample_tick_data):
    """Test tick data loading with resampling."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        sample_tick_data.to_csv(tmp.name, index=False)
        df = load_tick_csv(tmp.name, to_seconds=2)
        os.unlink(tmp.name)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # 3 ticks resampled to 2-second bars


def test_load_tick_csv_missing_columns():
    """Test loading data with missing required columns."""
    data = {
        "time": ["2024-01-01 00:00:00.000"],
        "bid": [1.0000],
    }
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        pd.DataFrame(data).to_csv(tmp.name, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            load_tick_csv(tmp.name)
        os.unlink(tmp.name)


def test_load_tick_csv_invalid_file():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_tick_csv("nonexistent.csv")


def test_load_tick_csv_mid_spread_calculation(tmp_path: Path):
    """Test mid price and spread calculations with resampling."""
    csv = tmp_path / "ticks.csv"
    csv.write_text(
        "time,bid,ask,volume\n"
        "2025-01-01 00:00:00.000,1.0000,1.0002,1\n"
        "2025-01-01 00:00:00.500,1.0001,1.0003,2\n"
    )
    df = load_tick_csv(csv, to_seconds=1)
    assert len(df) == 1
    row = df.iloc[0]
    assert abs(row.mid - 1.00015) < 1e-9
    assert abs(row.spread - 0.0002) < 1e-9 