"""Data loading utilities."""
import pandas as pd
from pathlib import Path
from typing import Optional


def load_tick_csv(path: str, to_seconds: Optional[int] = None) -> pd.DataFrame:
    """Load tick data from CSV or Parquet and optionally resample to bars.

    Args:
        path: Path to CSV/Parquet file with columns: time,bid,ask,volume
        to_seconds: If set, resample to bars of this many seconds

    Returns:
        DataFrame with processed tick data
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
    else:
        df = pd.read_csv(path, parse_dates=["time"])
    
    df.set_index("time", inplace=True)
    
    # Add mid price and spread if not present
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    if "spread" not in df.columns:
        df["spread"] = df["ask"] - df["bid"]
    
    if to_seconds is not None:
        # Resample to bars
        df = df.resample(f"{to_seconds}S", label="left", closed="left").agg({
            "bid": "last",
            "ask": "last",
            "volume": "sum",
            "mid": "last",
            "spread": "mean"
        })
        # Drop any bars with no data
        df = df.dropna()
    
    return df
