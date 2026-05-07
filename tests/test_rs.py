"""Tests for the RS rating + composite ranker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from momentum.composite import rank_composite
from momentum.minervini import TrendFeatures
from momentum.rs_rating import compute_weighted_perf, rs_rating_table


def _feat(ticker, returns):
    r3, r6, r9, r12 = returns
    return TrendFeatures(
        ticker=ticker, price=100.0, sma50=90, sma150=80, sma200=70,
        sma200_21d_ago=68, high_52w=105, low_52w=50,
        return_3m=r3, return_6m=r6, return_9m=r9, return_12m=r12,
    )


def test_weighted_perf_recent_quarter_dominates():
    # Same trailing 12m return (+50%), but A loaded into latest quarter, B uniform
    a = _feat("A", (0.45, 0.50, 0.50, 0.50))
    b = _feat("B", (0.10, 0.20, 0.35, 0.50))
    pa, pb = compute_weighted_perf(a), compute_weighted_perf(b)
    assert pa > pb, (pa, pb)


def test_rs_rating_monotonic_in_universe():
    feats = [_feat(f"T{i}", (0.01 * i, 0.01 * i, 0.01 * i, 0.01 * i)) for i in range(100)]
    df = rs_rating_table(feats)
    s = df.sort_values("ticker", key=lambda c: c.str.replace("T", "").astype(int))["rs_rating"].values
    # higher index -> higher return -> higher RS
    assert (np.diff(s) >= 0).all(), s


def test_composite_orders_by_three_factor_mean():
    survivors = pd.DataFrame([
        # Strong all-round
        dict(ticker="STRONG", rs_rating=99, price=100, high_52w=102, return_12m=1.20),
        # High RS but well off the high
        dict(ticker="OFF_HIGH", rs_rating=90, price=80, high_52w=100, return_12m=0.40),
        # Near high, low RS, low 1y
        dict(ticker="LATE", rs_rating=72, price=99, high_52w=100, return_12m=0.20),
        # Mid all-round
        dict(ticker="MID", rs_rating=85, price=90, high_52w=100, return_12m=0.55),
    ])
    out = rank_composite(survivors)
    assert out.iloc[0]["ticker"] == "STRONG"
    # OFF_HIGH should not lead because of dist_from_high penalty
    assert out.iloc[0]["composite"] >= out.iloc[1]["composite"]
