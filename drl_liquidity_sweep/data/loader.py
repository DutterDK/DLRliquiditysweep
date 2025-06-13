"""Data loading utilities."""
import pandas as pd
from pathlib import Path


def load_tick_csv(filepath: Path | str, to_seconds: bool = False) -> pd.DataFrame:
    """Load tick data from CSV file.

    Args:
        filepath: Path to CSV file
        to_seconds: Whether to convert timestamps to seconds

    Returns:
        DataFrame with tick data
    """
    df = pd.read_csv(
        filepath,
        parse_dates=["time"],
        date_parser=pd.to_datetime,
    )
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df
