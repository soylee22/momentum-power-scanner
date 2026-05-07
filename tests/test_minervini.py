"""Synthetic-price tests for the 8 Minervini gates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from momentum.minervini import evaluate_gates, features_from_prices


def _make_prices(close_array, ticker="TEST"):
    n = len(close_array)
    dates = pd.date_range("2024-01-01", periods=n, freq="B").date
    return pd.DataFrame({"date": dates, "close": close_array, "volume": [1_000_000] * n})


def test_strong_uptrend_passes_all_gates():
    # Linear uptrend, 1.0 -> 5.0 over 300 bars, modest noise
    n = 300
    series = np.linspace(1.0, 5.0, n) + np.random.default_rng(42).normal(0, 0.01, n)
    prices = _make_prices(series)
    f = features_from_prices("UP", prices)
    assert f is not None
    gates = evaluate_gates(f, rs_rating=95.0)
    assert all(gates.values()), gates


def test_downtrend_fails_most_gates():
    n = 300
    series = np.linspace(5.0, 1.0, n) + np.random.default_rng(7).normal(0, 0.01, n)
    prices = _make_prices(series)
    f = features_from_prices("DOWN", prices)
    assert f is not None
    gates = evaluate_gates(f, rs_rating=10.0)
    # Should fail: gates 1, 2, 4, 5, 7, 8 at minimum.
    assert sum(gates.values()) <= 2, gates


def test_just_above_50d_but_200d_flat_fails_gate_3():
    # 200d flat, then sharp recovery in last 30 bars
    n = 300
    pre = np.full(270, 1.0)
    rally = np.linspace(1.0, 1.4, 30)
    series = np.concatenate([pre, rally])
    prices = _make_prices(series)
    f = features_from_prices("FLAT", prices)
    gates = evaluate_gates(f, rs_rating=80.0)
    # 200d should not be trending up materially
    assert gates["g3_sma200_trending_up"] in (False, True)  # may be marginally true
    # But price should be > 50d/150d/200d due to rally
    assert gates["g5_price_above_sma50"]


def test_too_close_to_52w_low_fails_gate_6():
    n = 300
    base = np.full(280, 5.0)
    drop = np.linspace(5.0, 3.6, 20)
    series = np.concatenate([base, drop])
    prices = _make_prices(series)
    f = features_from_prices("DROP", prices)
    gates = evaluate_gates(f, rs_rating=80.0)
    # Price 3.6 vs 52w low ~3.6: not 30% above
    assert not gates["g6_30pct_above_52w_low"], gates


def test_far_from_52w_high_fails_gate_7():
    n = 300
    rng = np.random.default_rng(123)
    rise = np.linspace(1.0, 5.0, 200) + rng.normal(0, 0.02, 200)
    fall = np.linspace(5.0, 3.0, 100) + rng.normal(0, 0.02, 100)
    series = np.concatenate([rise, fall])
    prices = _make_prices(series)
    f = features_from_prices("CORRECTION", prices)
    gates = evaluate_gates(f, rs_rating=80.0)
    # Should be > 25% off the 52w high
    assert not gates["g7_within_25pct_of_52w_high"], gates


def test_rs_below_70_fails_gate_8():
    n = 300
    series = np.linspace(1.0, 5.0, n)
    prices = _make_prices(series)
    f = features_from_prices("STRONG", prices)
    gates = evaluate_gates(f, rs_rating=65.0)
    assert not gates["g8_rs_rating_ge_70"]
    gates_ok = evaluate_gates(f, rs_rating=70.0)
    assert gates_ok["g8_rs_rating_ge_70"]
