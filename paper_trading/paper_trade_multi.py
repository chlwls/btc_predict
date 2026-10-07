"""Forward paper trading of the frozen 10-coin portfolio (V6 MULTI_TREND).

Run once a day after 01:00 UTC (10:00 KST), like paper_trade.py.

  python paper_trade_multi.py                 # update the paper account
  python paper_trade_multi.py --verify DIR    # check frozen rule == btc_v6 on DIR/*_1h.csv
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio_frozen as P
import strategy_frozen as S
from paper_trade import fetch_recent

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "paper_state_multi.json"
LOG = ROOT / "paper_log_multi.csv"
LOG_COLUMNS = ["decision_time_utc", "symbol", "price", "target", "weight_before",
               "weight_after", "action", "paper_equity", "btc_bh_equity",
               "ew_bh_equity", "frozen_hash", "logged_at_utc"]


def frozen_hash():
    h = hashlib.sha256()
    for f in ["strategy_frozen.py", "portfolio_frozen.py"]:
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()[:12]


def fetch_all():
    closes = {}
    for sym in P.UNIVERSE:
        try:
            closes[sym] = fetch_recent(sym)
        except Exception as e:  # delisted or API error: coin is treated as invalid
            print(f"WARNING: {sym} fetch failed ({e}); treated as no signal")
    return pd.DataFrame(closes)


def update():
    panel = fetch_all()
    targets = pd.DataFrame({s: S.target_exposure(panel[s].dropna()) for s in panel})
    targets = targets.reindex(panel.index)
    decisions = panel.index[panel.index.hour == S.DECISION_HOUR_UTC]
    h = frozen_hash()

    if STATE.exists():
        st = json.loads(STATE.read_text())
        if st["frozen_hash"] != h:
            print(f"WARNING: frozen rule changed ({st['frozen_hash']} -> {h}).")
    else:
        first = decisions[-1]
        st = {"started": str(first), "frozen_hash": h, "units": {}, "weights": {},
              "equity": 1.0, "btc_bh_equity": 1.0, "ew_bh_equity": 1.0,
              "last_decision": None, "last_prices": {}}
        print(f"New multi-coin paper account starting at {first}")

    last = pd.Timestamp(st["last_decision"]) if st["last_decision"] else None
    rows = []
    for d in decisions:
        if (last is not None and d <= last) or (last is None and d != decisions[-1]):
            continue
        prices = {s: float(panel.at[d, s]) for s in panel if not np.isnan(panel.at[d, s])}
        rets = {s: prices[s] / st["last_prices"][s] - 1
                for s in prices if s in st["last_prices"]}
        if st["last_decision"]:
            st["equity"] *= 1 + sum(st["weights"].get(s, 0.0) * r for s, r in rets.items())
            st["btc_bh_equity"] *= 1 + rets.get("BTCUSDT", 0.0)
            if rets:
                st["ew_bh_equity"] *= 1 + np.mean(list(rets.values()))

        tg = {s: float(targets.at[d, s]) for s in targets}
        units, weights = P.decide(st["units"], tg)
        trade_cost = sum(P.cost(s) * abs(weights[s] - st["weights"].get(s, 0.0)) for s in P.UNIVERSE)
        st["equity"] *= 1 - trade_cost
        for s in P.UNIVERSE:
            before, after = st["weights"].get(s, 0.0), weights[s]
            if before != after:
                rows.append({"decision_time_utc": str(d), "symbol": s,
                             "price": prices.get(s), "target": round(tg.get(s, np.nan), 4),
                             "weight_before": round(before, 4), "weight_after": round(after, 4),
                             "action": "BUY" if after > before else "SELL"})
        rows.append({"decision_time_utc": str(d), "symbol": "_PORTFOLIO",
                     "weight_after": round(sum(weights.values()), 4),
                     "paper_equity": round(st["equity"], 6),
                     "btc_bh_equity": round(st["btc_bh_equity"], 6),
                     "ew_bh_equity": round(st["ew_bh_equity"], 6),
                     "frozen_hash": h,
                     "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        st.update(units=units, weights=weights, last_prices=prices, last_decision=str(d))

    if rows:
        pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    STATE.write_text(json.dumps(st, indent=2))

    print(f"last decision: {st['last_decision']}")
    for s in P.UNIVERSE:
        w = st["weights"].get(s, 0.0)
        print(f"  {s:9s} {w:6.1%}")
    print(f"  {'CASH':9s} {1 - sum(st['weights'].values()):6.1%}")
    print(f"paper equity {st['equity']:.4f} | BTC buy&hold {st['btc_bh_equity']:.4f} "
          f"| equal-weight buy&hold {st['ew_bh_equity']:.4f} | since {st['started']}")


def verify(data_dir):
    """Frozen rule must reproduce btc_v6.portfolio_positions at every decision."""
    import sys
    sys.path.insert(0, str(ROOT.parent / "btc_prediction_v6"))
    import btc_v6
    btc_v6.DATA_DIR = Path(data_dir)
    panel = btc_v6.load_closes()
    panel = panel[[s for s in P.UNIVERSE if s in panel]]
    ref = btc_v6.portfolio_positions(panel)
    targets = pd.DataFrame({s: S.target_exposure(panel[s].dropna()) for s in panel}).reindex(panel.index)
    units, worst = {}, 0.0
    for d in panel.index[panel.index.hour == S.DECISION_HOUR_UTC]:
        units, w = P.decide(units, {s: float(targets.at[d, s]) for s in panel})
        worst = max(worst, max(abs(w[s] - ref.at[d, s]) for s in panel))
    print(f"coins={list(panel.columns)}")
    print(f"max |frozen - V6| weight difference = {worst:.2e}")
    print("OK" if worst < 1e-9 else "MISMATCH")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", metavar="DATA_DIR")
    a = p.parse_args()
    verify(a.verify) if a.verify else update()
