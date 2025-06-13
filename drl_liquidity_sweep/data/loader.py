"""Data loading utilities."""
import pandas as pd
from pathlib import Path


def load_tick_csv(filepath: Path | str, to_seconds: int | None = None) -> pd.DataFrame:
    """Load tick data from CSV file.

    Args:
        filepath: Path to CSV file
        to_seconds: Bar size in seconds. If None, return tick data.

    Returns:
        DataFrame with tick data or resampled bars
    """
    df = pd.read_csv(
        filepath,
        parse_dates=["timestamp"],
        date_parser=pd.to_datetime,
    )
    df["mid"] = (df["bid"] + df["ask"]) / 2
    if to_seconds is not None:
        df = df.resample(f"{to_seconds}S", on="timestamp").agg({
            "bid": "first",
            "ask": "first",
            "volume": "sum",
            "mid": "last"
        }).reset_index()
    return df
