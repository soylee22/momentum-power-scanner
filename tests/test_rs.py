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


def test_composite_orders_by_four_factor_mean():
    survivors = pd.DataFrame([
        # Strong all-round, consistent path
        dict(ticker="STRONG", rs_rating=99, price=100, high_52w=102, return_12m=1.20, k_ratio=25.0),
        # High RS but well off the high
        dict(ticker="OFF_HIGH", rs_rating=90, price=80, high_52w=100, return_12m=0.40, k_ratio=12.0),
        # Near high, low RS, low 1y
        dict(ticker="LATE", rs_rating=72, price=99, high_52w=100, return_12m=0.20, k_ratio=8.0),
        # Mid all-round
        dict(ticker="MID", rs_rating=85, price=90, high_52w=100, return_12m=0.55, k_ratio=15.0),
    ])
    out = rank_composite(survivors)
    assert out.iloc[0]["ticker"] == "STRONG"
    # OFF_HIGH should not lead because of dist_from_high penalty
    assert out.iloc[0]["composite"] >= out.iloc[1]["composite"]
    assert "rank_k" in out.columns


def test_k_ratio_penalises_one_off_spike():
    """Same total gain, smoother path should rank above a one-spike path."""
    from momentum.minervini import k_ratio_from_closes

    n = 252
    t = np.arange(n, dtype=float)
    # Steady grind: +100% over the year, linear in log space
    smooth = 100.0 * np.exp(np.log(2.0) * t / (n - 1))
    # One-off spike: flat, then doubles in the last 5 days
    spike = np.full(n, 100.0)
    spike[-5:] = np.linspace(100.0, 200.0, 5)

    k_smooth = k_ratio_from_closes(smooth)
    k_spike = k_ratio_from_closes(spike)
    assert k_smooth > k_spike * 2, (k_smooth, k_spike)

    survivors = pd.DataFrame([
        dict(ticker="SMOOTH", rs_rating=90, price=200, high_52w=200, return_12m=1.0, k_ratio=k_smooth),
        dict(ticker="SPIKE", rs_rating=90, price=200, high_52w=200, return_12m=1.0, k_ratio=k_spike),
    ])
    out = rank_composite(survivors)
    assert out.iloc[0]["ticker"] == "SMOOTH"
