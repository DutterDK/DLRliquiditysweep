"""
Download N months of tick data from MT5, resample to bar_seconds,
and write one Parquet per month + a combined Parquet.

Requires MetaTrader5 terminal running & logged in.
"""

from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "EURUSD"
MONTHS = 12
BAR_SECONDS = 1
OUT_DIR = Path("data/eurusd_1s")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_month(year: int, month: int) -> pd.DataFrame:
    first = datetime(year, month, 1)
    last = (first + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
    ticks = mt5.copy_ticks_range(SYMBOL, first, last, mt5.COPY_TICKS_ALL)
    df = pd.DataFrame(ticks)[["time", "bid", "ask", "volume"]]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    rule = f"{BAR_SECONDS}s"
    agg = {
        "bid": "last",
        "ask": "last",
        "mid": "last",
        "spread": "mean",
        "volume": "sum",
    }
    bars = df.resample(rule, label="left", closed="left").agg(agg).dropna()
    return bars.reset_index()


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    today = datetime.utcnow()
    year, month = today.year, today.month
    all_months = []
    for i in range(MONTHS):
        y = year if month - i > 0 else year - 1
        m = (month - i - 1) % 12 + 1
        bars = fetch_month(y, m)
        fname = OUT_DIR / f"{SYMBOL}_{y}_{m:02d}.parquet"
        bars.to_parquet(fname, index=False)
        print(f"Saved {len(bars):,} bars → {fname}")
        all_months.append(bars)
    mt5.shutdown()
    full = pd.concat(reversed(all_months), ignore_index=True)
    full.to_parquet(OUT_DIR / f"{SYMBOL}_ALL.parquet", index=False)
    print(f"Final merged bars: {len(full):,}")


if __name__ == "__main__":
    main()
