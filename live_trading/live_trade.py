"""Upbit auto-trading of the frozen V6 MULTI_TREND portfolio.

Default is DRY RUN: computes and prints the orders, places nothing.
Real orders only with --live.

  python live_trade.py                 # dry run (virtual 500,000 KRW if no keys)
  python live_trade.py --live          # real orders
  python live_trade.py --live --force  # re-run today's decision

Signals: Binance USDT 1h data (the data the strategy was validated on).
Execution: Upbit KRW market orders. Coins not listed on Upbit KRW stay in cash.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "paper_trading"))
import portfolio_frozen as P  # noqa: E402
import strategy_frozen as S  # noqa: E402
from paper_trade_multi import fetch_all, frozen_hash  # noqa: E402
from upbit_client import UpbitClient, DryRunClient  # noqa: E402

CONFIG = ROOT / "config.json"
KEYS = ROOT / "upbit_keys.env"
STATE = ROOT / "live_state.json"
LOG = ROOT / "live_log.csv"

DEFAULT_CONFIG = {
    "capital_cap_krw": 500_000,   # the bot never manages more than this
    "kill_drawdown": 0.36,        # halt if managed value falls 36% below its peak (1.5x backtest MDD)
    "rebalance_tolerance": 0.10,  # act on a coin only if off target by >= 10% of one slot
    "min_order_krw": 5_000,       # Upbit minimum order
    "fee_buffer": 0.002,          # keep 0.2% of KRW aside for fees
}
LOG_COLUMNS = ["time_utc", "decision_time_utc", "mode", "market", "side", "krw", "volume",
               "price", "target_weight", "result"]


def log(msg):
    print(f"[LIVE] {msg}", flush=True)


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG.exists():
        cfg.update(json.loads(CONFIG.read_text(encoding="utf-8")))
    return cfg


def load_keys():
    ak, sk = os.environ.get("UPBIT_ACCESS_KEY"), os.environ.get("UPBIT_SECRET_KEY")
    if KEYS.exists():
        for line in KEYS.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == "UPBIT_ACCESS_KEY":
                    ak = v.strip()
                elif k.strip() == "UPBIT_SECRET_KEY":
                    sk = v.strip()
    return ak, sk


def market_of(symbol):
    return "KRW-" + symbol.replace("USDT", "")


def latest_decision(panel):
    targets = pd.DataFrame({s: S.target_exposure(panel[s].dropna()) for s in panel}).reindex(panel.index)
    decisions = panel.index[panel.index.hour == S.DECISION_HOUR_UTC]
    d = decisions[-1]
    age_h = (panel.index[-1] - d) / pd.Timedelta("1h")
    return d, {s: float(targets.at[d, s]) for s in targets}, age_h


def plan_orders(weights, balances, prices, upbit_markets, cfg):
    """Pure function: returns (orders, summary). Sells come before buys."""
    tradable = {s: w for s, w in weights.items() if market_of(s) in upbit_markets}
    values = {s: balances.get(s.replace("USDT", ""), 0.0) * prices.get(market_of(s), 0.0)
              for s in P.UNIVERSE if market_of(s) in prices}
    krw = balances.get("KRW", 0.0)
    managed = krw + sum(values.values())
    capital = min(managed, cfg["capital_cap_krw"])
    slot = capital / len(P.UNIVERSE)
    tol = max(cfg["min_order_krw"], cfg["rebalance_tolerance"] * slot)

    sells, buys = [], []
    for s in P.UNIVERSE:
        m = market_of(s)
        if m not in prices:
            continue
        target = tradable.get(s, 0.0) * capital
        cur = values.get(s, 0.0)
        diff = target - cur
        vol_held = balances.get(s.replace("USDT", ""), 0.0)
        if target == 0 and cur >= cfg["min_order_krw"]:
            sells.append({"market": m, "side": "ask", "volume": vol_held, "krw": cur,
                          "price": prices[m], "target_weight": 0.0})
        elif diff <= -tol:
            vol = min(vol_held, -diff / prices[m])
            sells.append({"market": m, "side": "ask", "volume": vol, "krw": -diff,
                          "price": prices[m], "target_weight": tradable.get(s, 0.0)})
        elif diff >= tol:
            buys.append({"market": m, "side": "bid", "krw": diff, "volume": diff / prices[m],
                         "price": prices[m], "target_weight": tradable.get(s, 0.0)})

    # Buys are funded by current KRW plus what the sells release, never more.
    budget = krw + sum(o["krw"] for o in sells)
    budget = min(budget, capital) * (1 - cfg["fee_buffer"])
    want = sum(o["krw"] for o in buys)
    scale = min(1.0, budget / want) if want > 0 else 1.0
    final_buys = []
    for o in buys:
        o["krw"] = int(o["krw"] * scale)
        o["volume"] = o["krw"] / o["price"]
        if o["krw"] >= cfg["min_order_krw"]:
            final_buys.append(o)
    summary = {"krw": krw, "coin_value": sum(values.values()), "managed": managed,
               "capital": capital, "slot": slot, "tolerance": tol,
               "not_on_upbit": [s for s in P.UNIVERSE if market_of(s) not in upbit_markets]}
    return sells + final_buys, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place real orders")
    ap.add_argument("--force", action="store_true", help="act again on an already handled decision")
    a = ap.parse_args()
    cfg = load_config()
    mode = "LIVE" if a.live else "DRY_RUN"

    ak, sk = load_keys()
    if a.live and not (ak and sk):
        sys.exit("--live needs UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY in upbit_keys.env")
    real = UpbitClient(ak, sk) if (ak and sk) else None
    client = real if a.live else DryRunClient(real, cfg["capital_cap_krw"])
    log(f"mode={mode} cap={cfg['capital_cap_krw']:,} KRW keys={'yes' if real else 'no'}")

    state = json.loads(STATE.read_text()) if STATE.exists() else \
        {"units": {}, "last_decision": None, "peak": 0.0, "halted": False, "frozen_hash": frozen_hash()}
    if state.get("frozen_hash") != frozen_hash():
        log("WARNING: frozen strategy files changed since this live account started.")
    if state.get("halted"):
        log("HALTED by kill switch. Inspect, then delete 'halted' in live_state.json to resume.")
        return

    panel = fetch_all()
    d, targets, age_h = latest_decision(panel)
    if age_h > 6:
        log(f"Signal data is stale ({age_h:.0f}h after decision bar). No orders.")
        return
    if state["last_decision"] == str(d) and not a.force:
        log(f"Decision {d} already handled. Nothing to do (use --force to re-run).")
        return

    units, weights = P.decide(state["units"], targets)
    markets = client.krw_markets()
    want_markets = [market_of(s) for s in P.UNIVERSE if market_of(s) in markets]
    prices = client.prices(want_markets)
    balances = client.balances()
    orders, summ = plan_orders(weights, balances, prices, markets, cfg)

    log(f"decision {d} | managed {summ['managed']:,.0f} KRW | capital {summ['capital']:,.0f} "
        f"| slot {summ['slot']:,.0f} | tolerance {summ['tolerance']:,.0f}")
    if summ["not_on_upbit"]:
        log(f"not on Upbit KRW (kept as cash): {summ['not_on_upbit']}")
    for s in P.UNIVERSE:
        flag = "" if market_of(s) in markets else "  (cash)"
        log(f"  {s:9s} target {weights[s]:6.1%}{flag}")

    # Kill switch on the managed value.
    state["peak"] = max(state.get("peak", 0.0), summ["capital"])
    if state["peak"] > 0 and summ["capital"] < state["peak"] * (1 - cfg["kill_drawdown"]):
        if a.live:
            state["halted"] = True
            STATE.write_text(json.dumps(state, indent=2))
        log(f"KILL SWITCH: capital {summ['capital']:,.0f} < {1 - cfg['kill_drawdown']:.0%} of peak "
            f"{state['peak']:,.0f}. No orders. Trading halted.")
        return

    if not orders:
        log("No orders needed.")
    new_log = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if new_log:
            w.writeheader()
        for o in orders:
            try:
                if o["side"] == "ask":
                    res = client.sell_market(o["market"], o["volume"])
                else:
                    res = client.buy_market_krw(o["market"], o["krw"])
                result = "dry_run" if res.get("dry_run") else f"ok uuid={res.get('uuid')}"
            except Exception as e:  # keep going; one failed order must not block the rest
                result = f"ERROR {e}"
            log(f"  {o['side']:3s} {o['market']:10s} {o['krw']:>10,.0f} KRW  -> {result}")
            w.writerow({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "decision_time_utc": str(d), "mode": mode, "market": o["market"],
                        "side": o["side"], "krw": round(o["krw"]), "volume": f"{o['volume']:.8f}",
                        "price": o["price"], "target_weight": round(o["target_weight"], 4),
                        "result": result})

    if a.live:  # dry runs never advance the live state
        state.update(units=units, last_decision=str(d), frozen_hash=frozen_hash())
        STATE.write_text(json.dumps(state, indent=2))
    log("done")


if __name__ == "__main__":
    main()
