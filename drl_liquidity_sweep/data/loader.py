"""Data loading utilities."""
import pandas as pd


def load_tick_csv(csv_path: str, to_seconds: int = 1) -> pd.DataFrame:
    """Load tick data from CSV and resample to specified bar size."""
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["time"],
            date_format="%Y-%m-%d %H:%M:%S.%f",
        )
        df.set_index("time", inplace=True)
    except TypeError:
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if to_seconds > 1:
        df = df.resample(f"{to_seconds}S", label="left", closed="left").agg(
            {"bid": "first", "ask": "first", "volume": "sum"}
        ).dropna()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    return df
