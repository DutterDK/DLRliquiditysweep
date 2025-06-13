"""Data loading utilities for MT5 tick data."""
from __future__ import annotations
from pathlib import Path
from typing import Union
import pandas as pd

def load_tick_csv(path: Union[str, Path], *, to_seconds: int = 1) -> pd.DataFrame:
    """
    Load an MT5 tick CSV with columns: time,bid,ask,volume.
    Returns a DataFrame indexed by UTC Timestamp and columns:
    ['bid', 'ask', 'mid', 'spread', 'volume'].
    If `to_seconds>0`, resample to that second interval.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the CSV file
    to_seconds : int, optional
        Resample interval in seconds, by default 1

    Returns
    -------
    pd.DataFrame
        Processed tick data

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
        date_parser=lambda col: pd.to_datetime(col, utc=True),
    )
    req = {"bid", "ask", "volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    df.set_index("time", inplace=True)
    df.index = df.index.tz_localize("UTC")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    if to_seconds > 0:
        rule = f"{to_seconds}s"
        agg = {
            "bid": "last",
            "ask": "last",
            "mid": "mean",
            "spread": "mean",
            "volume": "sum",
        }
        df = (
            df.resample(rule, label="left", closed="left")
            .agg(agg)
            .dropna(how="any")
            .astype("float64")
        )
    return df.sort_index()

