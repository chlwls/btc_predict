"""FROZEN multi-coin portfolio rule (V6 MULTI_TREND) for paper trading. Do not edit.

Frozen on 2026-10-07. Per coin the rule is strategy_frozen.py, unchanged.
At each daily decision:
  * coins with a valid signal (listed, >= 1440h of history) get one slot each
  * each coin's exposure inside its slot follows strategy_frozen.next_position
  * weight = slot exposure / number of valid coins  (total <= 100%, rest is cash)
  * a coin without a valid signal is sold
"""

import numpy as np

import strategy_frozen as S

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "LTCUSDT", "LINKUSDT", "TRXUSDT", "DOGEUSDT", "SOLUSDT",
]
LOW_COST = {"BTCUSDT": 0.0015, "ETHUSDT": 0.0015}
DEFAULT_COST = 0.0020


def cost(symbol):
    return LOW_COST.get(symbol, DEFAULT_COST)


def decide(units, targets):
    """units/targets: dict symbol -> float (target NaN = no valid signal).

    Returns (new_units, weights).
    """
    valid = [s for s in UNIVERSE if s in targets and not np.isnan(targets[s])]
    n = max(len(valid), 1)
    new_units = {}
    for s in UNIVERSE:
        if s in valid:
            new_units[s] = S.next_position(units.get(s, 0.0), targets[s])
        else:
            new_units[s] = 0.0
    weights = {s: u / n for s, u in new_units.items()}
    return new_units, weights
