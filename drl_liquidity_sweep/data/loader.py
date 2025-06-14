"""Data loading utilities."""
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def _resample_polars(df: pl.DataFrame, to_seconds: int) -> pd.DataFrame:
    every = f"{to_seconds}s"

    df = df.with_columns([
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        pl.col("time").cast(pl.Datetime),          # ensure proper dtype
    ])

    out = (
        df.group_by_dynamic(index_column="time", every=every, closed="left")
          .agg([
              pl.col("bid").last().alias("bid"),
              pl.col("ask").last().alias("ask"),
              pl.col("mid").last().alias("mid"),
              pl.col("spread").mean().alias("spread"),
              pl.col("volume").sum().alias("volume"),
          ])
          .sort("time")
    )
    return out.to_pandas()


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

    if to_seconds is None:
        # just ensure derived cols exist and index set
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread"] = df["ask"] - df["bid"]
        return df.set_index("time")

    if HAS_POLARS:
        # multi-core branch
        pdf = _resample_polars(pl.from_pandas(df), to_seconds)
    else:
        # single-core pandas fallback
        df = df.set_index("time")
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread"] = df["ask"] - df["bid"]
        pdf = (
            df.resample(f"{to_seconds}S", label="left", closed="left")
              .agg({"bid":"last","ask":"last","mid":"last",
                    "spread":"mean","volume":"sum"})
              .dropna()
              .reset_index()
        )
    return pdf.set_index("time")
