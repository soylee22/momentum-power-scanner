"""Mark Minervini Stage 2 Trend Template — 8 binary gates.

Source: Trade Like a Stock Market Wizard, ch. on the Trend Template.

All eight gates must be true for a stock to be classified as Stage 2.

  1. Price > 150d SMA AND Price > 200d SMA
  2. 150d SMA > 200d SMA
  3. 200d SMA trending up for at least 1 month
  4. 50d SMA > 150d SMA AND 50d SMA > 200d SMA
  5. Price > 50d SMA
  6. Price >= 30% above 52-week low
  7. Price within 25% of 52-week high
  8. RS rating >= 70 (computed externally and joined in)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendFeatures:
    """Per-ticker feature snapshot used by gates and the digest."""
    ticker: str
    price: float
    sma50: float
    sma150: float
    sma200: float
    sma200_21d_ago: float
    high_52w: float
    low_52w: float
    return_3m: float
    return_6m: float
    return_9m: float
    return_12m: float


def features_from_prices(ticker: str, prices: pd.DataFrame) -> TrendFeatures | None:
    """Compute all features needed by the gates from a single ticker's daily price series.

    Returns None if there are fewer than 220 daily bars or any close is
    non-positive (data error).
    """
    if prices is None or len(prices) < 220:
        return None
    close = prices["close"].astype(float).values
    if (close <= 0).any():
        return None

    sma50 = float(np.mean(close[-50:]))
    sma150 = float(np.mean(close[-150:]))
    sma200 = float(np.mean(close[-200:]))
    sma200_21 = float(np.mean(close[-221:-21]))

    win_252 = close[-252:] if len(close) >= 252 else close
    high_52w = float(np.max(win_252))
    low_52w = float(np.min(win_252))

    price = float(close[-1])

    def _ret(n: int) -> float:
        if len(close) <= n:
            return float("nan")
        return float(close[-1] / close[-1 - n] - 1.0)

    return TrendFeatures(
        ticker=ticker,
        price=price,
        sma50=sma50,
        sma150=sma150,
        sma200=sma200,
        sma200_21d_ago=sma200_21,
        high_52w=high_52w,
        low_52w=low_52w,
        return_3m=_ret(63),
        return_6m=_ret(126),
        return_9m=_ret(189),
        return_12m=_ret(252),
    )


def evaluate_gates(f: TrendFeatures, rs_rating: float) -> dict[str, bool]:
    """Apply the 8 gates and return a dict of pass/fail booleans."""
    return {
        "g1_price_above_150_200": f.price > f.sma150 and f.price > f.sma200,
        "g2_sma150_above_sma200": f.sma150 > f.sma200,
        "g3_sma200_trending_up": f.sma200 > f.sma200_21d_ago,
        "g4_sma50_above_150_200": f.sma50 > f.sma150 and f.sma50 > f.sma200,
        "g5_price_above_sma50": f.price > f.sma50,
        "g6_30pct_above_52w_low": f.price >= 1.30 * f.low_52w,
        "g7_within_25pct_of_52w_high": f.price >= 0.75 * f.high_52w,
        "g8_rs_rating_ge_70": rs_rating >= 70.0,
    }


def all_pass(gates: dict[str, bool]) -> bool:
    return all(gates.values())


def gates_passed_count(gates: dict[str, bool]) -> int:
    return int(sum(gates.values()))
