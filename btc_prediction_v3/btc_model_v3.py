import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

for d in [DATA_DIR, RESULT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[V3] {msg}", flush=True)


def fetch_binance_klines(symbol="BTCUSDT", interval="1h", years=3):
    path = DATA_DIR / f"btc_usdt_{interval}_raw.csv"

    # Reuse existing sufficiently large dataset.
    if path.exists():
        try:
            old = pd.read_csv(path, parse_dates=["open_time"])
            if len(old) > 5000:
                log(f"Using existing raw data: {path.name}, rows={len(old):,}")
                return old
        except Exception:
            pass

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(years * 365.25 * 24 * 60 * 60 * 1000)

    url = "https://api.binance.com/api/v3/klines"
    rows = []
    cursor = start_ms

    log(f"Downloading Binance {symbol} {interval} data...")
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()

        if not batch:
            break

        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if len(rows) % 10000 < 1000:
            log(f"downloaded={len(rows):,}")

        time.sleep(0.05)

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    numeric = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def rsi(s, n=14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def add_tf_features(base, tf, prefix):
    x = base.resample(tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    x[f"{prefix}_ret1"] = x["close"].pct_change()
    x[f"{prefix}_ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x[f"{prefix}_ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x[f"{prefix}_ema200"] = x["close"].ewm(span=200, adjust=False).mean()
    x[f"{prefix}_ema20_slope"] = x[f"{prefix}_ema20"].pct_change(5)
    x[f"{prefix}_ema50_slope"] = x[f"{prefix}_ema50"].pct_change(5)
    x[f"{prefix}_rsi14"] = rsi(x["close"], 14)
    x[f"{prefix}_atr_pct"] = atr(x) / x["close"]
    x[f"{prefix}_range20"] = x["high"].rolling(20).max() / x["low"].rolling(20).min() - 1
    x[f"{prefix}_trend"] = (
        (x["close"] > x[f"{prefix}_ema50"]).astype(int)
        + (x[f"{prefix}_ema50"] > x[f"{prefix}_ema200"]).astype(int)
    )

    keep = [c for c in x.columns if c.startswith(prefix + "_")]
    # Shift one completed higher-timeframe candle forward only.
    out = x[keep].shift(1)
    return out


def make_features(raw, horizon=12):
    df = raw.copy().set_index("open_time").sort_index()

    # Base 1h features.
    df["ret1"] = df["close"].pct_change()
    for n in [2, 3, 6, 12, 24, 48, 72, 168]:
        df[f"ret_{n}"] = df["close"].pct_change(n)

    for n in [8, 20, 50, 100, 200]:
        df[f"sma_{n}"] = df["close"].rolling(n).mean()
        df[f"dist_sma_{n}"] = df["close"] / df[f"sma_{n}"] - 1

    for n in [8, 20, 50, 200]:
        df[f"ema_{n}"] = df["close"].ewm(span=n, adjust=False).mean()
        df[f"dist_ema_{n}"] = df["close"] / df[f"ema_{n}"] - 1

    df["ema8_20_cross"] = df["ema_8"] / df["ema_20"] - 1
    df["ema20_50_cross"] = df["ema_20"] / df["ema_50"] - 1
    df["ema50_200_cross"] = df["ema_50"] / df["ema_200"] - 1

    df["rsi14"] = rsi(df["close"], 14)
    df["rsi7"] = rsi(df["close"], 7)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["atr14"] = atr(df)
    df["atr_pct"] = df["atr14"] / df["close"]

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    df["volume_z20"] = (
        (df["volume"] - df["volume"].rolling(20).mean())
        / df["volume"].rolling(20).std()
    )

    df["candle_body"] = (df["close"] - df["open"]) / df["open"]
    df["candle_range"] = (df["high"] - df["low"]) / df["open"]
    df["upper_wick"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / df["open"]
    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / df["open"]

    df["hh20"] = df["high"].rolling(20).max().shift(1)
    df["ll20"] = df["low"].rolling(20).min().shift(1)
    df["breakout_up"] = (df["close"] > df["hh20"]).astype(int)
    df["breakout_down"] = (df["close"] < df["ll20"]).astype(int)

    df["volatility_24"] = df["ret1"].rolling(24).std()
    df["volatility_72"] = df["ret1"].rolling(72).std()
    df["volatility_168"] = df["ret1"].rolling(168).std()

    # Higher timeframes. Resampling labels the completed higher-TF candle.
    for tf, prefix in [("4h", "h4"), ("1D", "d1")]:
        higher = add_tf_features(df[["open", "high", "low", "close", "volume"]], tf, prefix)
        df = df.join(higher.reindex(df.index, method="ffill"))

    # Regime features.
    df["trend_up"] = (
        (df["d1_trend"] >= 2) &
        (df["h4_trend"] >= 1)
    ).astype(int)
    df["trend_down"] = (
        (df["d1_trend"] == 0) &
        (df["h4_trend"] <= 1)
    ).astype(int)

    vol_rank = df["volatility_24"].rolling(720, min_periods=100).rank(pct=True)
    df["vol_rank"] = vol_rank

    df["regime"] = np.select(
        [
            df["trend_up"].eq(1) & (df["vol_rank"] >= 0.7),
            df["trend_up"].eq(1),
            df["trend_down"].eq(1) & (df["vol_rank"] >= 0.7),
            df["trend_down"].eq(1),
        ],
        ["TREND_UP_HIGH_VOL", "TREND_UP", "TREND_DOWN_HIGH_VOL", "TREND_DOWN"],
        default=np.where(df["vol_rank"] >= 0.7, "RANGE_HIGH_VOL", "RANGE_LOW_VOL")
    )

    # Targets. All future information is used only as target.
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["future_log_return"] = np.log(df["close"].shift(-horizon) / df["close"])
    df["target"] = (df["future_return"] > 0.005).astype(int)

    # One-hot regime.
    regime_dummies = pd.get_dummies(df["regime"], prefix="regime", dtype=int)
    df = pd.concat([df, regime_dummies], axis=1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(DATA_DIR / "btc_features_v3.csv")
    return df


def get_feature_columns(df):
    excluded = {
        "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
        "future_return", "future_log_return", "target", "regime"
    }
    cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def split_data(df, horizon):
    # Labels look `horizon` bars ahead, so drop the last `horizon` rows of
    # each earlier block; otherwise their targets overlap the next block.
    n = len(df)
    train_end = int(n * 0.65)
    valid_end = int(n * 0.80)
    return (
        df.iloc[:train_end - horizon],
        df.iloc[train_end:valid_end - horizon],
        df.iloc[valid_end:],
    )


def make_models():
    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=4
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=350, learning_rate=0.03, num_leaves=15,
            max_depth=-1, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, verbosity=-1, random_state=42
        )
    }


def threshold_search(y, p, future_returns):
    best = None
    for t in np.arange(0.40, 0.71, 0.02):
        sig = (p >= t).astype(int)
        count = int(sig.sum())
        if count < 10:
            continue
        gross = future_returns[sig == 1]
        score = gross.mean() if len(gross) else -999
        # Prefer positive expectancy and sufficient sample size.
        key = score
        if best is None or key > best[0]:
            best = (key, float(t), count)
    return best if best else (0.0, 0.50, 0)


def evaluate_classifier(model, X, y):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return {
        "auc": auc,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "prob": p
    }


def backtest_nonoverlap(test_df, prob, threshold, horizon, fee, slippage):
    d = test_df.reset_index().copy()
    d["prob"] = prob
    d["signal"] = (d["prob"] >= threshold).astype(int)

    cash = 1.0
    equity_rows = []
    trades = []
    i = 0

    while i < len(d) - horizon:
        row = d.iloc[i]
        if row["signal"] == 1:
            entry = float(row["close"])
            exit_row = d.iloc[i + horizon]
            exit_price = float(exit_row["close"])

            raw_ret = exit_price / entry - 1
            net_ret = raw_ret - 2 * (fee + slippage)
            cash *= (1 + net_ret)

            trades.append({
                "entry_time": row["open_time"],
                "exit_time": exit_row["open_time"],
                "entry_price": entry,
                "exit_price": exit_price,
                "raw_return": raw_ret,
                "net_return": net_ret,
                "probability": row["prob"],
                "regime": row.get("regime", "")
            })

            for j in range(i, min(i + horizon + 1, len(d))):
                equity_rows.append({
                    "open_time": d.iloc[j]["open_time"],
                    "equity": cash / (1 + net_ret) * (
                        1 + (d.iloc[j]["close"] / entry - 1)
                    )
                })
            i += horizon
        else:
            equity_rows.append({
                "open_time": row["open_time"],
                "equity": cash
            })
            i += 1

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        equity_df = pd.DataFrame(equity_rows).drop_duplicates("open_time").sort_values("open_time")
    else:
        equity_df = pd.DataFrame([{
            "open_time": d.iloc[0]["open_time"],
            "equity": 1.0
        }])

    if len(equity_df):
        peak = equity_df["equity"].cummax()
        dd = equity_df["equity"] / peak - 1
        mdd = float(dd.min())
    else:
        mdd = 0.0

    strategy_return = float(cash - 1)
    buy_hold = float(d["close"].iloc[-1] / d["close"].iloc[0] - 1)
    win_rate = float((trades_df["net_return"] > 0).mean()) if len(trades_df) else 0.0

    stats = {
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold,
        "max_drawdown": mdd,
        "trades": int(len(trades_df)),
        "win_rate": win_rate,
    }
    return stats, trades_df, equity_df


def pattern_statistics(df, horizon):
    x = df.copy()
    fr = x["future_return"]

    patterns = {
        "RSI_OVERSOLD": x["rsi14"] < 30,
        "RSI_OVERBOUGHT": x["rsi14"] > 70,
        "EMA_BULL_ALIGNMENT": (x["ema_8"] > x["ema_20"]) & (x["ema_20"] > x["ema_50"]),
        "EMA_BEAR_ALIGNMENT": (x["ema_8"] < x["ema_20"]) & (x["ema_20"] < x["ema_50"]),
        "BREAKOUT_UP": x["breakout_up"] == 1,
        "BREAKOUT_DOWN": x["breakout_down"] == 1,
        "VOLUME_SPIKE": x["vol_ratio"] > 2,
        "LARGE_BULL_CANDLE": x["candle_body"] > 0.01,
        "LARGE_BEAR_CANDLE": x["candle_body"] < -0.01,
        "TREND_ALIGNMENT_UP": x["trend_up"] == 1,
        "TREND_ALIGNMENT_DOWN": x["trend_down"] == 1,
        "LOW_VOL_COMPRESSION": x["bb_width"] < x["bb_width"].rolling(240, min_periods=50).quantile(0.2),
    }

    rows = []
    for name, mask in patterns.items():
        z = fr[mask].dropna()
        if len(z) == 0:
            continue
        rows.append({
            "pattern": name,
            "sample_count": int(len(z)),
            "mean_forward_return": float(z.mean()),
            "median_forward_return": float(z.median()),
            "win_rate": float((z > 0).mean()),
            "positive_0.5pct_rate": float((z > 0.005).mean()),
            "negative_rate": float((z < 0).mean()),
            "expectancy_after_0.3pct_cost": float((z - 0.003).mean()),
            "horizon_hours": horizon
        })

    out = pd.DataFrame(rows).sort_values("expectancy_after_0.3pct_cost", ascending=False)
    out.to_csv(RESULT_DIR / "pattern_statistics_v3.csv", index=False)
    return out


def train_regressor(Xtr, ytr, Xte, yte):
    model = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = math.sqrt(mean_squared_error(yte, pred))
    sign_acc = float((np.sign(pred) == np.sign(yte)).mean())
    return model, pred, mae, rmse, sign_acc


def walk_forward(df, features, horizon, fee, slippage):
    # Four non-overlapping future windows.
    n = len(df)
    start = int(n * 0.50)
    window = max(1000, int(n * 0.10))
    rows = []

    for k in range(4):
        train_end = start + k * window
        test_start = train_end
        test_end = min(test_start + window, n)
        if test_end <= test_start or train_end < 3000:
            continue

        train = df.iloc[:train_end - horizon].dropna(subset=features + ["target", "future_return"])
        test = df.iloc[test_start:test_end].dropna(subset=features + ["target", "future_return"])

        if len(test) < 100:
            continue

        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, verbosity=-1, random_state=42
        )
        model.fit(train[features], train["target"])
        p = model.predict_proba(test[features])[:, 1]
        auc = roc_auc_score(test["target"], p)

        # Fixed threshold from validation concept, not tuned on this future window.
        bt, _, _ = backtest_nonoverlap(
            test, p, 0.50, horizon, fee, slippage
        )

        rows.append({
            "window": k + 1,
            "train_end": str(train.index[-1]),
            "test_start": str(test.index[0]),
            "test_end": str(test.index[-1]),
            "auc": auc,
            **bt
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "walk_forward_v3.csv", index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=3)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    args = parser.parse_args()

    log("V3 started")
    raw = fetch_binance_klines(years=args.years)
    full = make_features(raw, args.horizon)

    df = full.dropna(subset=["future_return", "future_log_return"]).copy()
    features = get_feature_columns(df)
    df = df.dropna(subset=features + ["target", "future_return"]).copy()

    train, valid, test = split_data(df, args.horizon)
    log(f"rows={len(df):,}")
    log(f"train={len(train):,}, valid={len(valid):,}, test={len(test):,}")
    log(f"feature_count={len(features)}")
    log(f"train_positive_rate={train.target.mean():.4f}")
    log(f"valid_positive_rate={valid.target.mean():.4f}")
    log(f"test_positive_rate={test.target.mean():.4f}")

    # Pattern research is performed on training data only for discovery.
    pattern_statistics(train, args.horizon)

    model_rows = []
    trained = {}

    for name, model in make_models().items():
        log(f"Training {name}...")
        model.fit(train[features], train["target"])

        v = evaluate_classifier(model, valid[features], valid["target"])
        t = evaluate_classifier(model, test[features], test["target"])

        threshold_info = threshold_search(
            valid["target"].to_numpy(),
            v["prob"],
            valid["future_return"].to_numpy()
        )
        threshold = threshold_info[1]

        bt, trades, equity = backtest_nonoverlap(
            test, t["prob"], threshold, args.horizon, args.fee, args.slippage
        )

        model_rows.append({
            "model": name,
            "valid_auc": v["auc"],
            "test_auc": t["auc"],
            "valid_accuracy": v["accuracy"],
            "test_accuracy": t["accuracy"],
            "valid_precision": v["precision"],
            "test_precision": t["precision"],
            "valid_recall": v["recall"],
            "test_recall": t["recall"],
            "threshold": threshold,
            **bt
        })

        trained[name] = {
            "model": model,
            "test_prob": t["prob"],
            "threshold": threshold
        }

    comparison = pd.DataFrame(model_rows).sort_values(
        ["valid_auc", "test_auc"], ascending=False
    )
    comparison.to_csv(RESULT_DIR / "model_comparison_v3.csv", index=False)

    winner_name = comparison.iloc[0]["model"]
    winner = trained[winner_name]
    log(f"winner={winner_name}, threshold={winner['threshold']:.2f}")

    # Regression: future log-return.
    reg_train = train.dropna(subset=["future_log_return"])
    reg_test = test.dropna(subset=["future_log_return"])
    reg_model, reg_pred, mae, rmse, sign_acc = train_regressor(
        reg_train[features],
        reg_train["future_log_return"],
        reg_test[features],
        reg_test["future_log_return"]
    )
    joblib.dump(reg_model, MODEL_DIR / "best_regressor_v3.joblib")

    # Final winner artifacts.
    joblib.dump(winner["model"], MODEL_DIR / "best_classifier_v3.joblib")

    test_pred = test.copy()
    test_pred["probability_up"] = winner["test_prob"]
    test_pred["prediction"] = (
        test_pred["probability_up"] >= winner["threshold"]
    ).astype(int)
    test_pred["regression_log_return"] = reg_pred[:len(test_pred)]
    test_pred.to_csv(RESULT_DIR / "test_predictions_v3.csv", index=False)

    bt, trades, equity = backtest_nonoverlap(
        test, winner["test_prob"], winner["threshold"],
        args.horizon, args.fee, args.slippage
    )
    trades.to_csv(RESULT_DIR / "backtest_trades_v3.csv", index=False)
    equity.to_csv(RESULT_DIR / "equity_curve_v3.csv", index=False)

    if len(equity):
        plt.figure(figsize=(12, 5))
        plt.plot(equity["open_time"], equity["equity"])
        plt.title(f"BTC V3 Equity Curve - {winner_name}")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "equity_curve_v3.png", dpi=150)
        plt.close()

    wf = walk_forward(df, features, args.horizon, args.fee, args.slippage)

    # Use the newest completed bar, not the last labelled row (horizon bars older).
    latest_X = full[features].dropna().iloc[[-1]].astype(float)
    latest = full.loc[latest_X.index[0]]
    latest_prob = float(winner["model"].predict_proba(latest_X)[:, 1][0])
    latest_reg = float(reg_model.predict(latest_X)[0])

    latest_info = {
        "timestamp": str(latest_X.index[0]),
        "close": float(latest["close"]),
        "probability_up": latest_prob,
        "probability_down": 1 - latest_prob,
        "decision": "LONG" if latest_prob >= winner["threshold"] else "NO-LONG",
        "threshold": float(winner["threshold"]),
        "winner_model": winner_name,
        "regime": str(latest["regime"]),
        "horizon_hours": args.horizon,
        "fee_per_side": args.fee,
        "slippage_per_side": args.slippage,
        "regression_log_return": latest_reg,
        "regression_note": "Regression model predicts future log return; latest regression value is not used for the decision."
    }
    with open(RESULT_DIR / "latest_prediction_v3.json", "w", encoding="utf-8") as f:
        json.dump(latest_info, f, ensure_ascii=False, indent=2)

    log("")
    log("===== V3 TEST RESULT =====")
    log(comparison.to_string(index=False))
    log("")
    log(f"REGRESSION MAE={mae:.6f}, RMSE={rmse:.6f}, sign_accuracy={sign_acc:.4f}")
    log(f"WINNER={winner_name}")
    log(f"BACKTEST return={bt['strategy_return']:.4f}")
    log(f"BUY&HOLD return={bt['buy_hold_return']:.4f}")
    log(f"MAX_DRAWDOWN={bt['max_drawdown']:.4f}")
    log(f"TRADES={bt['trades']}, WIN_RATE={bt['win_rate']:.4f}")
    if not wf.empty:
        log("")
        log("===== WALK FORWARD =====")
        log(wf.to_string(index=False))
    log("")
    log("===== LATEST =====")
    log(json.dumps(latest_info, ensure_ascii=False, indent=2))
    log("")
    log("Saved results to results/ and models/")


if __name__ == "__main__":
    main()
