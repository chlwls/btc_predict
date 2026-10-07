"""Forward paper trading of the frozen BTC strategy.

Run once a day after 01:00 UTC (10:00 KST). Missed days are caught up
automatically (up to ~120 days). Nothing is ever traded for real.

  python paper_trade.py            # update the paper account
  python paper_trade.py --verify   # check frozen code == V5 backtest
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import strategy_frozen as S

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "paper_state.json"
LOG = ROOT / "paper_log.csv"
FETCH_BARS = 3000


def strategy_hash():
    return hashlib.sha256((ROOT / "strategy_frozen.py").read_bytes()).hexdigest()[:12]


def fetch_recent(symbol="BTCUSDT", bars=FETCH_BARS):
    url = "https://api.binance.com/api/v3/klines"
    end_ms = int(time.time() * 1000)
    cursor = end_ms - bars * 3600 * 1000
    rows = []
    while cursor < end_ms:
        r = requests.get(url, params={"symbol": symbol, "interval": "1h",
                                      "startTime": cursor, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        time.sleep(0.05)
    df = pd.DataFrame(rows).iloc[:, [0, 4, 6]]
    df.columns = ["open_time", "close", "close_time"]
    df = df[df["close_time"].astype("int64") < end_ms]  # completed bars only
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    s = df.drop_duplicates("open_time").set_index("open_time")["close"].astype(float)
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq="1h")).ffill()


def update():
    close = fetch_recent()
    target = S.target_exposure(close)
    decisions = close.index[close.index.hour == S.DECISION_HOUR_UTC]
    h = strategy_hash()

    if STATE.exists():
        st = json.loads(STATE.read_text())
        if st["strategy_hash"] != h:
            print(f"WARNING: strategy_frozen.py changed ({st['strategy_hash']} -> {h}). "
                  "This is no longer the frozen strategy.")
    else:
        first = decisions[-1]
        st = {"started": str(first), "strategy_hash": h, "position": 0.0,
              "equity": 1.0, "bh_equity": 1.0, "last_decision": None,
              "last_price": float(close[first])}
        print(f"New paper account starting at {first} (price {close[first]:,.2f})")

    last = pd.Timestamp(st["last_decision"]) if st["last_decision"] else None
    if last is not None and last < close.index[0] + pd.Timedelta(hours=max(S.LOOKBACKS)):
        print("WARNING: last run is older than the fetched window; gap cannot be replayed exactly.")

    new_rows = []
    for d in decisions:
        if last is not None and d <= last:
            continue
        if last is None and d != decisions[-1]:
            continue
        price = float(close[d])
        ret = price / st["last_price"] - 1
        st["equity"] *= 1 + st["position"] * ret
        st["bh_equity"] *= 1 + ret
        tgt = float(target[d])
        new_pos = S.next_position(st["position"], tgt)
        trade = new_pos - st["position"]
        st["equity"] *= 1 - S.COST_PER_SIDE * abs(trade)
        action = "HOLD" if trade == 0 else ("BUY" if trade > 0 else "SELL")
        new_rows.append({
            "decision_time_utc": str(d), "btc_close": price, "target": round(tgt, 4),
            "position_before": round(st["position"], 4), "position_after": round(new_pos, 4),
            "action": action, "paper_equity": round(st["equity"], 6),
            "buy_hold_equity": round(st["bh_equity"], 6), "strategy_hash": h,
            "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        st["position"], st["last_price"], st["last_decision"] = new_pos, price, str(d)

    if new_rows:
        out = pd.DataFrame(new_rows)
        out.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    STATE.write_text(json.dumps(st, indent=2))

    print(f"BTC close (last 00:00 UTC bar) : {st['last_price']:,.2f}")
    print(f"momentum votes                : {S.momentum_votes(close)}")
    print(f"target exposure (latest bar)  : {float(target.iloc[-1]):.2f}")
    print(f"PAPER POSITION                : {st['position']:.0%} BTC")
    for r in new_rows:
        print(f"  {r['decision_time_utc']}  {r['action']:4s}  "
              f"{r['position_before']:.2f} -> {r['position_after']:.2f}")
    print(f"paper equity {st['equity']:.4f} | buy&hold {st['bh_equity']:.4f} | since {st['started']}")


def verify():
    """Frozen code must reproduce V5 TREND_ENS_SMOOTH on the 9-year data."""
    import sys
    sys.path.insert(0, str(ROOT.parent / "btc_prediction_v5"))
    sys.path.insert(0, str(ROOT.parent / "btc_prediction_v4"))
    import btc_v4
    from btc_v5 import rebalance, ENS_LOOKBACKS
    btc_v4.FALLBACK_RAW = ROOT.parent / "btc_prediction_v4" / "data" / "btc_usdt_1h_raw.csv"
    df = btc_v4.load_data(0, False)
    ens = btc_v4.trend_signal(df, ENS_LOOKBACKS)
    vs = btc_v4.vol_scale(df, 0.40)
    ref = rebalance((ens * vs).ewm(span=72, adjust=False).mean(), df.index.hour == 0, 0.20)

    tgt = S.target_exposure(df["close"])
    pos, cur = np.zeros(len(df)), 0.0
    for i, (t, hr) in enumerate(zip(tgt.to_numpy(), df.index.hour)):
        if hr == S.DECISION_HOUR_UTC:
            cur = S.next_position(cur, t)
        pos[i] = cur
    diff = np.abs(pos - ref.to_numpy()).max()
    print(f"max |frozen - V5| position difference = {diff:.2e}")
    # Short-window check: fetching only FETCH_BARS must give the same target.
    tail = S.target_exposure(df["close"].iloc[-FETCH_BARS:])
    d2 = (tail - tgt.iloc[-FETCH_BARS:]).abs().iloc[-24:].max()
    print(f"max target difference using only last {FETCH_BARS} bars = {d2:.2e}")
    print("OK" if diff < 1e-9 and d2 < 1e-3 else "MISMATCH")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    verify() if a.verify else update()
