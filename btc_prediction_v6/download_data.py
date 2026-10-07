"""Download Binance spot 1h klines for the V6 universe into data/{SYMBOL}_1h.csv.

  python download_data.py                 # all default symbols, from 2017-08
  python download_data.py --symbols ETHUSDT SOLUSDT

Re-running resumes from the last saved bar.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "LTCUSDT", "LINKUSDT", "TRXUSDT", "DOGEUSDT", "SOLUSDT",
]
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def download(symbol, start):
    path = DATA_DIR / f"{symbol}_1h.csv"
    old = None
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["open_time"])
        cursor = int(old["open_time"].max().timestamp() * 1000) + 1
    end_ms = int(time.time() * 1000)
    rows = []
    while cursor < end_ms:
        for attempt in range(5):
            try:
                r = requests.get("https://api.binance.com/api/v3/klines", params={
                    "symbol": symbol, "interval": "1h", "startTime": cursor, "limit": 1000,
                }, timeout=30)
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"{symbol}: download failed")
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        if len(rows) % 20000 < 1000:
            print(f"  {symbol}: {len(rows):,} new bars", flush=True)
        time.sleep(0.05)
    new = pd.DataFrame(rows, columns=COLS)
    new = new[new["close_time"].astype("int64") < end_ms]  # completed bars only
    new["open_time"] = pd.to_datetime(new["open_time"], unit="ms", utc=True)
    df = pd.concat([old, new]) if old is not None else new
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df.to_csv(path, index=False)
    print(f"{symbol}: {len(df):,} bars {df['open_time'].min()} -> {df['open_time'].max()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", default="2017-08-01")
    a = p.parse_args()
    for s in a.symbols:
        download(s, a.start)
