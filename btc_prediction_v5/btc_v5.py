"""BTC V5: lower-turnover trend following, judged by a pre-registered scorecard.

Decided before running (see README_v5.md):
  * ML direction model dropped (V1-V4: no edge after costs).
  * PRIMARY = TREND_ENS: equal-weight momentum votes over 72h/1w/2w/1m/2m,
    vol-targeted to 40%, daily rebalance, 20%p deadband. The lookbacks are
    NOT chosen from the V4 grid; they span it.
  * Variants are reported for sensitivity only; the verdict is on PRIMARY.
  * Gate #5 changed from "positive quarters >= 60%" to
    "fewer losing quarters than buy&hold" (user decision after V4 results).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "btc_prediction_v4"))
from btc_v4 import (  # noqa: E402
    load_data as _load_v4, trend_signal, vol_scale, run_backtest, metrics,
    bootstrap_sharpe, daily, realized_vol,
)

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DATA = ROOT.parent / "btc_prediction_v4" / "data" / "btc_usdt_1h_raw.csv"

PRIMARY = "TREND_ENS"
ENS_LOOKBACKS = [72, 168, 336, 720, 1440]


def log(msg):
    print(f"[V5] {msg}", flush=True)


def rebalance(target, decision_mask, deadband):
    """Change position only at decision times and only if the move > deadband."""
    t = target.to_numpy()
    dm = np.asarray(decision_mask)
    out = np.zeros_like(t)
    cur = 0.0
    for i in range(len(t)):
        if dm[i] and not np.isnan(t[i]):
            if abs(t[i] - cur) > deadband or (t[i] == 0 and cur != 0):
                cur = t[i]
        out[i] = cur
    return pd.Series(out, index=target.index)


def losing_quarters(pnl):
    q = (1 + daily(pnl)).resample("QE").prod() - 1
    return int((q < 0).sum()), len(q)


GATE = {
    "sharpe >= 1.0": lambda m, bh: m["sharpe"] >= 1.0,
    "sharpe > buy&hold": lambda m, bh: m["sharpe"] > bh["sharpe"],
    "|MDD| <= 0.75 * buy&hold": lambda m, bh: abs(m["max_drawdown"]) <= 0.75 * abs(bh["max_drawdown"]),
    "bootstrap 5% sharpe > 0": lambda m, bh: m["sharpe_p05"] > 0,
    "losing quarters < buy&hold": lambda m, bh: m["losing_quarters"] < bh["losing_quarters"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-vol", type=float, default=0.40)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--warmup-days", type=int, default=365)
    p.add_argument("--holdout-days", type=int, default=180)
    args = p.parse_args()
    cost = args.fee + args.slippage

    import btc_v4
    btc_v4.FALLBACK_RAW = DATA
    df = _load_v4(years=0, refresh=False)
    log(f"rows={len(df):,} {df.index[0]} -> {df.index[-1]}")

    dev_start = (df.index[0] + pd.Timedelta(days=args.warmup_days)).normalize() + pd.Timedelta(days=1)
    end = df.index[-1] + pd.Timedelta("1h")
    hold_start = (end - pd.Timedelta(days=args.holdout_days)).normalize()
    log(f"dev     : {dev_start.date()} -> {hold_start.date()}")
    log(f"holdout : {hold_start.date()} -> {df.index[-1].date()}")

    vs = vol_scale(df, args.target_vol)
    daily_mask = df.index.hour == 0
    weekly_mask = daily_mask & (df.index.dayofweek == 0)
    ens = trend_signal(df, ENS_LOOKBACKS)

    positions = {
        "BUY_HOLD": pd.Series(1.0, index=df.index),
        "BH_VOLTARGET": rebalance(vs, daily_mask, 0.10),
        "TREND_V4": rebalance(trend_signal(df, [168, 336, 720]) * vs, daily_mask, 0.10),
        "TREND_ENS": rebalance(ens * vs, daily_mask, 0.20),
        "TREND_ENS_SMOOTH": rebalance((ens * vs).ewm(span=72, adjust=False).mean(), daily_mask, 0.20),
        "TREND_ENS_WEEKLY": rebalance(ens * vs, weekly_mask, 0.20),
    }

    periods = {"dev": (dev_start, hold_start), "holdout": (hold_start, end)}
    rows, results, pnls = [], {}, {}
    for per, (s, e) in periods.items():
        results[per] = {}
        for name, pos in positions.items():
            pnl, eq, pp = run_backtest(df, pos, s, e, cost)
            m = metrics(pnl, pp)
            m["sharpe_p05"], m["sharpe_p50"], m["sharpe_p95"] = bootstrap_sharpe(pnl)
            m["losing_quarters"], m["quarters"] = losing_quarters(pnl)
            results[per][name] = m
            rows.append({"period": per, "strategy": name, **m})
            pnls[(per, name)] = pnl
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "summary_v5.csv", index=False)

    bh = results["dev"]["BUY_HOLD"]
    score = pd.DataFrame({
        n: {**{k: bool(fn(m, bh)) for k, fn in GATE.items()}}
        for n, m in results["dev"].items()
    }).T
    score["PASS"] = score.all(axis=1)
    score.to_csv(RESULT_DIR / "scorecard_v5.csv")

    # Calendar-year returns (dev + holdout combined) for the primary vs buy&hold.
    years = {}
    for name in ["BUY_HOLD", PRIMARY]:
        pnl, _, _ = run_backtest(df, positions[name], dev_start, end, cost)
        years[name] = (1 + daily(pnl)).resample("YE").prod() - 1
    yearly = pd.DataFrame(years)
    yearly.index = yearly.index.year
    yearly.to_csv(RESULT_DIR / "yearly_v5.csv")

    # Cost stress for the primary (dev).
    stress = []
    for mult in [0, 1, 2, 3]:
        pnl, _, pp = run_backtest(df, positions[PRIMARY], dev_start, hold_start, cost * mult)
        m = metrics(pnl, pp)
        stress.append({"cost_multiplier": mult, "cost_per_side": cost * mult,
                       "sharpe": m["sharpe"], "cagr": m["cagr"], "max_drawdown": m["max_drawdown"]})
    stress = pd.DataFrame(stress)
    stress.to_csv(RESULT_DIR / "cost_stress_v5.csv", index=False)

    # Plot dev+holdout equity.
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in ["BUY_HOLD", "BH_VOLTARGET", "TREND_V4", PRIMARY]:
        pnl, eq, _ = run_backtest(df, positions[name], dev_start, end, cost)
        ax.plot(eq.index, eq.values, label=name, lw=1.6 if name == PRIMARY else 1.0)
    ax.axvline(hold_start, color="grey", ls="--", lw=1)
    ax.text(hold_start, ax.get_ylim()[1], " holdout", va="top", color="grey")
    ax.set_yscale("log")
    ax.set_title("BTC V5 - equity net of costs (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "equity_v5.png", dpi=130)
    plt.close(fig)

    latest = {
        "timestamp": str(df.index[-1]),
        "close": float(df["close"].iloc[-1]),
        "strategy": PRIMARY,
        "momentum_votes": {f"{n}h": bool(df["close"].pct_change(n).iloc[-1] > 0) for n in ENS_LOOKBACKS},
        "trend_signal": float(ens.iloc[-1]),
        "realized_vol_168h": float(realized_vol(df).iloc[-1]),
        "vol_scale": float(vs.iloc[-1]),
        "target_exposure_now": float(ens.iloc[-1] * vs.iloc[-1]),
        "current_position": float(positions[PRIMARY].iloc[-1]),
        "note": "Position changes only at 00:00 UTC and only if |target - current| > 0.20.",
        "dev_pass": bool(score.loc[PRIMARY, "PASS"]),
    }
    with open(RESULT_DIR / "latest_signal_v5.json", "w") as fh:
        json.dump(latest, fh, indent=2)

    pd.set_option("display.width", 220)
    show = ["strategy", "total_return", "cagr", "sharpe", "sharpe_p05", "max_drawdown", "calmar",
            "avg_exposure", "turnover_per_year", "losing_quarters", "quarters"]
    for per in periods:
        log("")
        log(f"===== {per.upper()} =====")
        print(summary[summary.period == per][show].round(3).to_string(index=False))
    log("")
    log("===== SCORECARD (dev) =====")
    print(score.to_string())
    log("")
    log(f"===== YEARLY RETURNS ({PRIMARY} vs BUY_HOLD) =====")
    print(yearly.round(3).to_string())
    log("")
    log(f"===== COST STRESS ({PRIMARY}, dev) =====")
    print(stress.round(4).to_string(index=False))
    log("")
    log("===== LATEST =====")
    print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    main()
