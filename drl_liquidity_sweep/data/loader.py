"""Data loading utilities."""
import pandas as pd
from pathlib import Path


def load_tick_csv(
    filepath: Path | str, to_seconds: bool = False
) -> pd.DataFrame:
    """Load tick data from CSV file.

    Parameters
    ----------
    filepath : Path | str
        Path to CSV file
    to_seconds : bool, optional
        Convert time to seconds, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, bid, ask, volume
    """
    df = pd.read_csv(
        filepath,
        parse_dates=['time'],
        date_parser=lambda x: pd.to_datetime(
            x, unit='s' if to_seconds else None
        )
    )
    # Calculate mid price
    df['mid'] = (df['bid'] + df['ask']) / 2
    return df
