"""FROZEN strategy for forward paper trading. Do not edit.

Frozen on 2026-10-07 from V5 TREND_ENS_SMOOTH:
  target = smooth_72h( momentum_vote(72h, 1w, 2w, 1m, 2m) * min(1, 40% / vol_168h) )
  act once a day on the 00:00 UTC hourly bar (decided at its close, 01:00 UTC),
  and only if |target - position| > 0.20 (or target hits 0).

paper_trade.py records a hash of this file; any change shows up in the log.
"""

import numpy as np
import pandas as pd

LOOKBACKS = [72, 168, 336, 720, 1440]
TARGET_VOL = 0.40
VOL_WINDOW = 168
SMOOTH_SPAN = 72
DEADBAND = 0.20
COST_PER_SIDE = 0.0015  # fee 0.10% + slippage 0.05%
DECISION_HOUR_UTC = 0
HOURS_PER_YEAR = 24 * 365


def target_exposure(close: pd.Series) -> pd.Series:
    """`close` must be a gap-free hourly series (UTC open_time index)."""
    votes = [(close.pct_change(n) > 0).astype(float) for n in LOOKBACKS]
    signal = (sum(votes) / len(votes)).where(close.pct_change(max(LOOKBACKS)).notna())
    vol = close.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(HOURS_PER_YEAR)
    vol_scale = (TARGET_VOL / vol).clip(upper=1.0)
    return (signal * vol_scale).ewm(span=SMOOTH_SPAN, adjust=False).mean()


def next_position(current: float, target: float) -> float:
    if np.isnan(target):
        return current
    if abs(target - current) > DEADBAND or (target == 0 and current != 0):
        return float(target)
    return current


def momentum_votes(close: pd.Series) -> dict:
    return {f"{n}h": bool(close.pct_change(n).iloc[-1] > 0) for n in LOOKBACKS}
