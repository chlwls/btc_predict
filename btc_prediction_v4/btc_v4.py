"""BTC V4: position-based walk-forward research.

V1-V3 tried to predict short-horizon direction and found AUC ~0.5.
V4 changes the question from "will it go up in 12h?" to
"how much exposure should I hold right now?", and judges every strategy
by the same out-of-sample scorecard, net of costs.

Periods
  warmup   : first `--min-train-days` (features + first ML training set)
  dev      : walk-forward out-of-sample period used to compare versions
  holdout  : last `--holdout-days`, reported separately, never used to choose
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
FALLBACK_RAW = ROOT.parent / "btc_prediction_v3" / "data" / "btc_usdt_1h_raw.csv"
HOURS_PER_YEAR = 24 * 365

for d in [DATA_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[V4] {msg}", flush=True)


# ---------------------------------------------------------------- data

def fetch_binance_1h(years):
    url = "https://api.binance.com/api/v3/klines"
    end_ms = int(time.time() * 1000)
    cursor = end_ms - int(years * 365.25 * 24 * 3600 * 1000)
    rows = []
    log(f"Downloading Binance BTCUSDT 1h, {years} years...")
    while cursor < end_ms:
        r = requests.get(url, params={
            "symbol": "BTCUSDT", "interval": "1h",
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        time.sleep(0.05)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def load_data(years, refresh):
    path = DATA_DIR / "btc_usdt_1h_raw.csv"
    if refresh or not (path.exists() or FALLBACK_RAW.exists()):
        df = fetch_binance_1h(years)
        df.to_csv(path, index=False)
    else:
        src = path if path.exists() else FALLBACK_RAW
        log(f"Using cached data: {src}")
        df = pd.read_csv(src, parse_dates=["open_time"])
    df = df.drop_duplicates("open_time").set_index("open_time").sort_index()
    df = df[["open", "high", "low", "close", "volume", "taker_buy_base"]].astype(float)
    # Fill missing hours so that shift(n) always means n hours.
    full = pd.date_range(df.index[0], df.index[-1], freq="1h")
    df = df.reindex(full)
    df["close"] = df["close"].ffill()
    for c in ["open", "high", "low"]:
        df[c] = df[c].fillna(df["close"])
    df[["volume", "taker_buy_base"]] = df[["volume", "taker_buy_base"]].fillna(0)
    return df


# ---------------------------------------------------------------- features

def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(df):
    """All features use data up to and including the close of bar t."""
    c = df["close"]
    r1 = c.pct_change()
    f = pd.DataFrame(index=df.index)

    for n in [24, 72, 168, 336, 720, 1440]:
        f[f"ret_{n}"] = c.pct_change(n)
    for n in [24, 168, 720]:
        f[f"vol_{n}"] = r1.rolling(n).std() * np.sqrt(HOURS_PER_YEAR)
    f["vol_ratio_24_720"] = f["vol_24"] / f["vol_720"]
    f["vol_ratio_168_720"] = f["vol_168"] / f["vol_720"]
    for n in [168, 336, 720, 1440]:
        f[f"dist_sma_{n}"] = c / c.rolling(n).mean() - 1
    for n in [168, 720, 1440]:
        f[f"dd_{n}"] = c / c.rolling(n).max() - 1
        f[f"up_{n}"] = c / c.rolling(n).min() - 1
    # Risk-adjusted momentum (t-stat like).
    for n in [168, 720]:
        f[f"sharpe_{n}"] = r1.rolling(n).mean() / r1.rolling(n).std() * np.sqrt(n)
    f["rsi_24h"] = rsi(c, 24)
    f["rsi_168h"] = rsi(c, 168)
    daily_close = c.resample("1D").last()
    f["rsi_d14"] = rsi(daily_close, 14).shift(1).reindex(df.index, method="ffill")
    vol = df["volume"]
    f["volu_ratio_24_720"] = vol.rolling(24).sum() / (vol.rolling(720).sum() / 30)
    f["volu_ratio_168_720"] = vol.rolling(168).sum() / (vol.rolling(720).sum() / 720 * 168)
    taker = df["taker_buy_base"] / vol.replace(0, np.nan)
    f["taker_buy_24"] = taker.rolling(24).mean()
    f["taker_buy_168"] = taker.rolling(168).mean()
    f["range_24"] = df["high"].rolling(24).max() / df["low"].rolling(24).min() - 1
    f["dow"] = df.index.dayofweek
    return f.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- exposures

def realized_vol(df, n=168):
    return df["close"].pct_change().rolling(n).std() * np.sqrt(HOURS_PER_YEAR)


def vol_scale(df, target_vol, cap=1.0):
    if not target_vol:
        return pd.Series(1.0, index=df.index)
    return (target_vol / realized_vol(df)).clip(upper=cap)


def trend_signal(df, lookbacks):
    """Fraction of lookbacks with positive momentum, in [0, 1]."""
    c = df["close"]
    votes = [(c.pct_change(n) > 0).astype(float) for n in lookbacks]
    sig = sum(votes) / len(votes)
    return sig.where(c.pct_change(max(lookbacks)).notna())


def ml_walk_forward(df, feats, horizon, eval_start, retrain_days, seed=42):
    """Expanding-window LightGBM, retrained every `retrain_days`.

    Label for row t uses close[t + horizon], so a model trained at cutoff T
    only sees rows with t + horizon <= T (purged).
    """
    c = df["close"]
    fwd = c.shift(-horizon) / c - 1
    y = (fwd > 0).astype(float).where(fwd.notna())
    X = feats
    prob = pd.Series(np.nan, index=df.index)
    cutoffs = pd.date_range(eval_start, df.index[-1], freq=f"{retrain_days}D")
    importances = []

    for k, cut in enumerate(cutoffs):
        nxt = cutoffs[k + 1] if k + 1 < len(cutoffs) else df.index[-1] + pd.Timedelta("1h")
        label_ok = df.index <= cut - pd.Timedelta(hours=horizon)
        sub = label_ok & (df.index.hour % 4 == 0)  # thin overlapping labels
        tr = X[sub].join(y[sub].rename("y")).dropna()
        if len(tr) < 1000:
            continue
        model = make_classifier(seed)
        model.fit(tr[X.columns], tr["y"])
        importances.append(pd.Series(model.feature_importances_, index=X.columns))
        blk = (df.index >= cut) & (df.index < nxt)
        xb = X[blk].dropna()
        if len(xb):
            prob.loc[xb.index] = model.predict_proba(xb)[:, 1]

    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False) \
        if importances else pd.Series(dtype=float)
    return prob, imp


def make_classifier(seed=42):
    # Deliberately small and heavily regularised: the signal is weak.
    return LGBMClassifier(
        n_estimators=200, learning_rate=0.02, num_leaves=7, max_depth=3,
        min_child_samples=200, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.6, reg_lambda=5.0, verbosity=-1, random_state=seed,
    )


def ml_exposure(prob, width=0.10):
    """p=0.5 -> 0, p=0.5+width -> 1 (long only)."""
    return ((prob - 0.5) / width).clip(0, 1)


# ---------------------------------------------------------------- backtest

def rebalance(target, rebalance_hours, deadband):
    """Act only every `rebalance_hours` (UTC), and only if the change > deadband."""
    t = target.to_numpy()
    hours = target.index.hour.to_numpy()
    out = np.zeros_like(t)
    cur = 0.0
    for i in range(len(t)):
        if hours[i] % rebalance_hours == 0 and not np.isnan(t[i]):
            if abs(t[i] - cur) > deadband or (t[i] == 0 and cur != 0):
                cur = t[i]
        out[i] = cur
    return pd.Series(out, index=target.index)


def run_backtest(df, position, start, end, cost):
    """Position decided at close of bar t earns the return of bar t+1."""
    m = (df.index >= start) & (df.index < end)
    ret = df["close"].pct_change()[m]
    pos = position[m].copy()
    prev = pos.shift(1).fillna(0.0)  # flat when the period starts
    pnl = prev * ret - cost * (pos - prev).abs()
    equity = (1 + pnl.fillna(0)).cumprod()
    return pnl.fillna(0), equity, pos


def daily(pnl):
    return (1 + pnl).resample("1D").prod() - 1


def metrics(pnl, pos):
    d = daily(pnl)
    eq = (1 + d).cumprod()
    years = len(d) / 365
    total = eq.iloc[-1] - 1
    sharpe = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else 0.0
    downside = d[d < 0].std()
    mdd = (eq / eq.cummax() - 1).min()
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    q = (1 + d).resample("QE").prod() - 1
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": d.std() * np.sqrt(365),
        "sharpe": sharpe,
        "sortino": d.mean() / downside * np.sqrt(365) if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "avg_exposure": pos.mean(),
        "turnover_per_year": pos.diff().abs().sum() / years if years > 0 else np.nan,
        "pct_quarters_positive": (q > 0).mean(),
    }


def bootstrap_sharpe(pnl, n_boot=2000, block=10, seed=0):
    """Circular block bootstrap of daily returns -> 5th/50th/95th pct Sharpe."""
    d = daily(pnl).to_numpy()
    n = len(d)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    sample = d[idx.reshape(n_boot, -1)[:, :n]]
    sd = sample.std(axis=1)
    sh = np.where(sd > 0, sample.mean(axis=1) / sd * np.sqrt(365), 0)
    return np.percentile(sh, [5, 50, 95])


# ---------------------------------------------------------------- scorecard

GATE = {
    "sharpe >= 1.0": lambda m, bh: m["sharpe"] >= 1.0,
    "sharpe > buy&hold": lambda m, bh: m["sharpe"] > bh["sharpe"],
    "|MDD| <= 0.75 * buy&hold": lambda m, bh: abs(m["max_drawdown"]) <= 0.75 * abs(bh["max_drawdown"]),
    "bootstrap 5% sharpe > 0": lambda m, bh: m["sharpe_p05"] > 0,
    "positive quarters >= 60%": lambda m, bh: m["pct_quarters_positive"] >= 0.6,
}


def scorecard(rows, bh_name="BUY_HOLD"):
    bh = rows[bh_name]
    out = {}
    for name, m in rows.items():
        checks = {k: bool(fn(m, bh)) for k, fn in GATE.items()}
        out[name] = {**checks, "PASS": all(checks.values())}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=float, default=3)
    p.add_argument("--refresh", action="store_true", help="re-download data")
    p.add_argument("--horizon", type=int, default=24, help="ML label horizon (hours)")
    p.add_argument("--rebalance", type=int, default=24, help="rebalance every N hours")
    p.add_argument("--deadband", type=float, default=0.10)
    p.add_argument("--target-vol", type=float, default=0.40)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--min-train-days", type=int, default=365)
    p.add_argument("--retrain-days", type=int, default=30)
    p.add_argument("--holdout-days", type=int, default=180)
    args = p.parse_args()
    cost = args.fee + args.slippage

    df = load_data(args.years, args.refresh)
    feats = make_features(df)
    log(f"rows={len(df):,} {df.index[0]} -> {df.index[-1]}, features={feats.shape[1]}")

    eval_start = df.index[0] + pd.Timedelta(days=args.min_train_days)
    eval_start = eval_start.normalize() + pd.Timedelta(days=1)
    end = df.index[-1] + pd.Timedelta("1h")
    holdout_start = (end - pd.Timedelta(days=args.holdout_days)).normalize()
    log(f"dev     : {eval_start.date()} -> {holdout_start.date()}")
    log(f"holdout : {holdout_start.date()} -> {df.index[-1].date()}")

    # ---- target exposures (before rebalancing rules)
    vs = vol_scale(df, args.target_vol)
    trend = trend_signal(df, [168, 336, 720])
    log("Walk-forward ML training...")
    prob, importance = ml_walk_forward(df, feats, args.horizon, eval_start, args.retrain_days)
    ml = ml_exposure(prob)

    targets = {
        "BUY_HOLD": pd.Series(1.0, index=df.index),
        "BH_VOLTARGET": vs,
        "TREND": trend * vs,
        "ML": ml * vs,
        "TREND_x_ML": (0.5 * trend + 0.5 * ml) * vs,
    }

    positions = {k: (v if k == "BUY_HOLD" else rebalance(v, args.rebalance, args.deadband))
                 for k, v in targets.items()}

    periods = {"dev": (eval_start, holdout_start), "holdout": (holdout_start, end)}
    summary = []
    per_period = {}
    curves = {}
    for per, (s, e) in periods.items():
        rows = {}
        for name, pos in positions.items():
            pnl, eq, pp = run_backtest(df, pos, s, e, cost)
            m = metrics(pnl, pp)
            m["sharpe_p05"], m["sharpe_p50"], m["sharpe_p95"] = bootstrap_sharpe(pnl)
            rows[name] = m
            summary.append({"period": per, "strategy": name, **m})
            if per == "dev":
                curves[name] = eq
        per_period[per] = rows

    summary = pd.DataFrame(summary)
    summary.to_csv(RESULT_DIR / "summary_v4.csv", index=False)
    score = scorecard(per_period["dev"])
    score.to_csv(RESULT_DIR / "scorecard_v4.csv")

    # ---- parameter robustness for TREND (dev only)
    grid = []
    for lbs in [[72], [168], [336], [720], [1440], [72, 168, 336], [168, 336, 720], [336, 720, 1440]]:
        for tv in [None, 0.3, 0.4, 0.6]:
            tgt = trend_signal(df, lbs) * vol_scale(df, tv)
            pos = rebalance(tgt, args.rebalance, args.deadband)
            pnl, _, pp = run_backtest(df, pos, eval_start, holdout_start, cost)
            m = metrics(pnl, pp)
            grid.append({"lookbacks": str(lbs), "target_vol": tv, **m})
    grid = pd.DataFrame(grid)
    grid.to_csv(RESULT_DIR / "trend_grid_v4.csv", index=False)

    # ---- ML diagnostics
    c = df["close"]
    fwd = c.shift(-args.horizon) / c - 1
    m_dev = (df.index >= eval_start) & (df.index < holdout_start) & prob.notna() & fwd.notna()
    from sklearn.metrics import roc_auc_score
    ml_auc = roc_auc_score((fwd[m_dev] > 0).astype(int), prob[m_dev])
    importance.head(20).to_csv(RESULT_DIR / "ml_importance_v4.csv")

    # ---- plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, label=name, lw=1.2)
    ax.set_yscale("log")
    ax.set_title("BTC V4 - dev period equity (net of costs, log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "equity_v4.png", dpi=130)
    plt.close(fig)

    # ---- latest signal: final ML model trained on all labelled data
    label_ok = (df.index <= df.index[-1] - pd.Timedelta(hours=args.horizon)) & (df.index.hour % 4 == 0)
    y = (fwd > 0).astype(float).where(fwd.notna())
    tr = feats[label_ok].join(y[label_ok].rename("y")).dropna()
    final = make_classifier().fit(tr[feats.columns], tr["y"])
    last_x = feats.iloc[[-1]]
    p_last = float(final.predict_proba(last_x)[:, 1][0])
    latest = {
        "timestamp": str(df.index[-1]),
        "close": float(c.iloc[-1]),
        "realized_vol_168h": float(realized_vol(df).iloc[-1]),
        "vol_scale": float(vs.iloc[-1]),
        "trend_signal": float(trend.iloc[-1]),
        "ml_prob_up_24h": p_last,
        "target_exposure": {
            "BH_VOLTARGET": float(vs.iloc[-1]),
            "TREND": float(trend.iloc[-1] * vs.iloc[-1]),
            "ML": float(ml_exposure(pd.Series([p_last])).iloc[0] * vs.iloc[-1]),
            "TREND_x_ML": float((0.5 * trend.iloc[-1] + 0.5 * ml_exposure(pd.Series([p_last])).iloc[0]) * vs.iloc[-1]),
        },
        "passing_strategies": score.index[score["PASS"]].tolist(),
    }
    with open(RESULT_DIR / "latest_signal_v4.json", "w") as fh:
        json.dump(latest, fh, indent=2)

    # ---- report
    pd.set_option("display.width", 200)
    show = ["strategy", "total_return", "cagr", "sharpe", "sharpe_p05", "max_drawdown",
            "calmar", "avg_exposure", "turnover_per_year", "pct_quarters_positive"]
    for per in periods:
        log("")
        log(f"===== {per.upper()} =====")
        print(summary[summary.period == per][show].round(3).to_string(index=False))
    log("")
    log(f"ML walk-forward AUC (dev, {args.horizon}h direction) = {ml_auc:.3f}")
    log("ML top features: " + ", ".join(importance.head(8).index))
    log("")
    log("===== SCORECARD (dev) =====")
    print(score.to_string())
    log("")
    log("===== TREND ROBUSTNESS (dev) =====")
    g = grid.pivot(index="lookbacks", columns="target_vol", values="sharpe")
    print(g.round(2).to_string())
    log(f"median sharpe over grid = {grid.sharpe.median():.2f} "
        f"(buy&hold {per_period['dev']['BUY_HOLD']['sharpe']:.2f})")
    log("")
    log("===== LATEST =====")
    print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    main()
