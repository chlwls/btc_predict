"""V6: the frozen trend rule applied to a multi-coin universe.

Pre-registered before any altcoin data was seen (see README_v6.md):
  * Per coin: exactly the frozen paper-trading rule (paper_trading/strategy_frozen.py).
  * PRIMARY = MULTI_TREND: each available coin gets 1/N of capital, scaled by its
    own trend target; N = coins with a valid signal at that decision time.
    Total exposure <= 100%, no leverage, long only, spot.
  * Costs per side: BTC/ETH 0.15%, other coins 0.20%.
  * Same dev/holdout split and gates as V5, judged against BTC buy&hold.
  * Known bias: the universe is today's survivors, which flatters every
    multi-coin result (including the equal-weight benchmark).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "paper_trading"))
sys.path.insert(0, str(ROOT.parent / "btc_prediction_v4"))
sys.path.insert(0, str(ROOT.parent / "btc_prediction_v5"))
import strategy_frozen as S  # noqa: E402
from btc_v4 import metrics, bootstrap_sharpe, daily  # noqa: E402
from btc_v5 import GATE, losing_quarters  # noqa: E402

DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)
BTC_FALLBACK = ROOT.parent / "btc_prediction_v4" / "data" / "btc_usdt_1h_raw.csv"
LOW_COST = {"BTCUSDT": 0.0015, "ETHUSDT": 0.0015}
DEFAULT_COST = 0.0020
WARMUP_DAYS, HOLDOUT_DAYS = 365, 180


def log(msg):
    print(f"[V6] {msg}", flush=True)


def load_closes():
    files = {p.name.replace("_1h.csv", ""): p for p in sorted(DATA_DIR.glob("*_1h.csv"))}
    if "BTCUSDT" not in files and BTC_FALLBACK.exists():
        files["BTCUSDT"] = BTC_FALLBACK
    closes = {}
    for sym, path in files.items():
        d = pd.read_csv(path, usecols=["open_time", "close"], parse_dates=["open_time"])
        s = d.drop_duplicates("open_time").set_index("open_time")["close"].astype(float).sort_index()
        closes[sym] = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="1h")).ffill()
    panel = pd.DataFrame(closes).sort_index()
    return panel.reindex(pd.date_range(panel.index[0], panel.index[-1], freq="1h"))


def portfolio_positions(panel, use_trend=True):
    """Per-coin weights (fraction of total capital), changed only at 00:00 UTC."""
    decision = panel.index.hour == S.DECISION_HOUR_UTC
    targets = {}
    for sym in panel:
        c = panel[sym]
        alive = c.notna()
        t = S.target_exposure(c.dropna()).reindex(panel.index)
        if not use_trend:  # equal-weight buy & hold benchmark
            t = pd.Series(1.0, index=panel.index).where(c.pct_change(max(S.LOOKBACKS)).notna())
        targets[sym] = t.where(alive)
    T = pd.DataFrame(targets)

    weights = np.zeros(T.shape)
    cur = np.zeros(T.shape[1])  # per-coin exposure in units of its 1/N slot
    tv = T.to_numpy()
    n_slots = 1
    for i in range(len(T)):
        if decision[i]:
            row = tv[i]
            valid = ~np.isnan(row)
            n_slots = max(int(valid.sum()), 1)
            for j in range(len(cur)):
                if not valid[j]:
                    cur[j] = 0.0  # not yet listed / no history / delisted
                else:
                    cur[j] = S.next_position(cur[j], row[j]) if use_trend else 1.0
        weights[i] = cur / n_slots
    return pd.DataFrame(weights, index=T.index, columns=T.columns)


def backtest(panel, W, start, end):
    m = (panel.index >= start) & (panel.index < end)
    R = panel.pct_change()[m].fillna(0.0)
    W = W[m]
    prev = W.shift(1).fillna(0.0)
    costs = pd.Series({c: LOW_COST.get(c, DEFAULT_COST) for c in W.columns})
    pnl = (prev * R).sum(axis=1) - ((W - prev).abs() * costs).sum(axis=1)
    return pnl, W.sum(axis=1)


def main():
    panel = load_closes()
    syms = list(panel.columns)
    log(f"universe ({len(syms)}): " + ", ".join(
        f"{s}[{panel[s].first_valid_index().date()}]" for s in syms))
    if len(syms) == 1:
        log("Only BTC found. Run download_data.py first; results below are a smoke test.")

    btc_start = panel["BTCUSDT"].first_valid_index()
    dev_start = (btc_start + pd.Timedelta(days=WARMUP_DAYS)).normalize() + pd.Timedelta(days=1)
    end = panel.index[-1] + pd.Timedelta("1h")
    hold_start = (end - pd.Timedelta(days=HOLDOUT_DAYS)).normalize()
    log(f"dev {dev_start.date()} -> {hold_start.date()} | holdout -> {panel.index[-1].date()}")

    btc_only = panel[["BTCUSDT"]]
    strategies = {
        "BTC_BUY_HOLD": pd.DataFrame({"BTCUSDT": 1.0}, index=panel.index),
        "EW_BUY_HOLD": portfolio_positions(panel, use_trend=False),
        "BTC_TREND": portfolio_positions(btc_only),
        "MULTI_TREND": portfolio_positions(panel),
    }

    rows, res = [], {}
    for per, (s, e) in {"dev": (dev_start, hold_start), "holdout": (hold_start, end)}.items():
        res[per] = {}
        for name, W in strategies.items():
            p = panel[W.columns]
            pnl, expo = backtest(p, W, s, e)
            m = metrics(pnl, expo)
            m["sharpe_p05"], m["sharpe_p50"], m["sharpe_p95"] = bootstrap_sharpe(pnl)
            m["losing_quarters"], m["quarters"] = losing_quarters(pnl)
            res[per][name] = m
            rows.append({"period": per, "strategy": name, **m})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "summary_v6.csv", index=False)

    bh = res["dev"]["BTC_BUY_HOLD"]
    score = pd.DataFrame({n: {k: bool(fn(m, bh)) for k, fn in GATE.items()}
                          for n, m in res["dev"].items()}).T
    score["PASS"] = score.all(axis=1)
    score.to_csv(RESULT_DIR / "scorecard_v6.csv")

    # Diagnostics: each coin's own trend vs its own buy&hold (dev), and correlations.
    per_coin = []
    for sym in syms:
        c = panel[[sym]]
        first = max(dev_start, c[sym].first_valid_index() + pd.Timedelta(hours=max(S.LOOKBACKS)))
        if first >= hold_start:
            continue
        for label, W in [("buy_hold", pd.DataFrame({sym: 1.0}, index=panel.index).where(c.notna(), 0.0)),
                         ("trend", portfolio_positions(c))]:
            pnl, expo = backtest(c, W, first, hold_start)
            m = metrics(pnl, expo)
            per_coin.append({"symbol": sym, "kind": label, "from": first.date(),
                             "sharpe": m["sharpe"], "cagr": m["cagr"], "max_drawdown": m["max_drawdown"]})
    per_coin = pd.DataFrame(per_coin)
    per_coin.to_csv(RESULT_DIR / "per_coin_v6.csv", index=False)
    dr = panel.resample("1D").last().pct_change()
    corr = dr[(dr.index >= dev_start) & (dr.index < hold_start)].corr()
    corr.to_csv(RESULT_DIR / "correlation_v6.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, W in strategies.items():
        pnl, _ = backtest(panel[W.columns], W, dev_start, end)
        ax.plot(pnl.index, (1 + pnl).cumprod(), label=name, lw=1.6 if name == "MULTI_TREND" else 1.0)
    ax.axvline(hold_start, color="grey", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title("V6 - equity net of costs (log scale); dashed = holdout start")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "equity_v6.png", dpi=130)
    plt.close(fig)

    W = strategies["MULTI_TREND"].iloc[-1]
    latest = {"timestamp": str(panel.index[-1]),
              "weights": {k: round(float(v), 4) for k, v in W.items() if v > 0},
              "cash": round(1 - float(W.sum()), 4)}
    (RESULT_DIR / "latest_weights_v6.json").write_text(json.dumps(latest, indent=2))

    pd.set_option("display.width", 220)
    show = ["strategy", "total_return", "cagr", "sharpe", "sharpe_p05", "max_drawdown",
            "calmar", "avg_exposure", "turnover_per_year", "losing_quarters", "quarters"]
    for per in ["dev", "holdout"]:
        log(f"===== {per.upper()} =====")
        print(summary[summary.period == per][show].round(3).to_string(index=False))
    log("===== SCORECARD (dev, vs BTC buy&hold) =====")
    print(score.to_string())
    log("===== PER COIN (dev, from first valid signal) =====")
    if len(per_coin):
        print(per_coin.pivot(index="symbol", columns="kind", values="sharpe").round(2).to_string())
    log(f"mean pairwise daily correlation = {corr.where(~np.eye(len(corr), dtype=bool)).stack().mean():.2f}")
    log("===== LATEST WEIGHTS =====")
    print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    main()
