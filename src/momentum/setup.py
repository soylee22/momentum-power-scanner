"""Minervini-style 'setup' detection — VCP + pivot break.

This is the entry-trigger layer that the v1 trend-template scanner
deliberately omitted. A Stage 2 stock can be Stage 2 for months while
extended from any reasonable buy-point; v1 just took the top 10 by
composite each month regardless. Minervini's actual playbook waits for
a tight base (VCP) and a breakout above its high (pivot).

Implementation is approximate. True VCP recognition is pattern-based
(successively tighter pullbacks within a base) and hard to encode
robustly across hundreds of names. We use three numerical proxies:

  1. Within 5% of 52-week high — i.e. at the top of whatever base
     has formed, not extended off the lows.
  2. ATR(20) / ATR(60) <= 0.85 — short-term volatility CONTRACTING
     relative to medium-term; the "tightening" signature.
  3. Volume(20) / Volume(60) <= 0.85 — volume drying up; classic
     base-completion behaviour.

A 'pivot break' is the buyable trigger: today's close above the prior
10-day max on volume >= 1.3x the 20d average. This filters extended
chases — only names that just broke out qualify for entry.

Both flags are computed point-in-time so the backtest can ask
'was this name a buy on date X?'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_setup_panel(
    prices_close: pd.DataFrame,
    prices_volume: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """For each ticker, return a date-indexed DataFrame with columns:

        atr_20, atr_60, vol_20, vol_60,
        atr_contraction (20/60), vol_contraction (20/60),
        near_high (dist_from_52w_high <= 0.05),
        vcp_flag (near_high AND atr_contraction <= 0.85 AND vol_contraction <= 0.85),
        pivot_break (close > prior 10d max AND volume >= 1.3 * 20d avg),
        setup_buy (vcp_flag AND pivot_break).
    """
    out: dict[str, pd.DataFrame] = {}
    for tk in prices_close.columns:
        s = prices_close[tk].dropna()
        if len(s) < 252:
            continue
        v = prices_volume[tk].reindex(s.index)

        df = pd.DataFrame(index=s.index)
        df["close"] = s
        df["volume"] = v

        # True Range approximation using close-only (we don't have high/low here):
        # TR ≈ |close_t − close_{t-1}|. Crude but consistent across names and
        # captures volatility contraction reliably.
        tr = s.diff().abs()
        df["atr_20"] = tr.rolling(20).mean()
        df["atr_60"] = tr.rolling(60).mean()
        df["atr_contraction"] = df["atr_20"] / df["atr_60"]

        df["vol_20"] = v.rolling(20).mean()
        df["vol_60"] = v.rolling(60).mean()
        df["vol_contraction"] = df["vol_20"] / df["vol_60"]

        df["high_52w"] = s.rolling(252, min_periods=252).max()
        df["dist_from_high"] = (df["high_52w"] - s) / df["high_52w"]
        df["near_high"] = df["dist_from_high"] <= 0.05

        df["vcp_flag"] = (
            df["near_high"]
            & (df["atr_contraction"] <= 0.85)
            & (df["vol_contraction"] <= 0.85)
        )

        prior_10d_max = s.shift(1).rolling(10).max()
        df["prior_10d_max"] = prior_10d_max
        vol_20 = df["vol_20"]
        df["pivot_break"] = (
            (s > prior_10d_max)
            & (v >= 1.3 * vol_20)
            & vol_20.notna()
        )

        df["setup_buy"] = df["vcp_flag"] & df["pivot_break"]

        out[tk] = df.dropna(subset=["atr_60", "high_52w"])
    return out


def asof_setup_lookup(
    setups: dict[str, pd.DataFrame],
    d: pd.Timestamp,
) -> pd.DataFrame:
    """Return one row per ticker with the setup state on date d (or the
    most recent prior trading day)."""
    rows = []
    for tk, df in setups.items():
        valid = df.index[df.index <= d]
        if len(valid) == 0:
            continue
        r = df.loc[valid[-1]]
        rows.append({
            "ticker": tk,
            "near_high": bool(r["near_high"]),
            "vcp_flag": bool(r["vcp_flag"]),
            "pivot_break": bool(r["pivot_break"]),
            "setup_buy": bool(r["setup_buy"]),
            "atr_contraction": float(r["atr_contraction"]) if not np.isnan(r["atr_contraction"]) else np.nan,
            "vol_contraction": float(r["vol_contraction"]) if not np.isnan(r["vol_contraction"]) else np.nan,
            "dist_from_high": float(r["dist_from_high"]) if not np.isnan(r["dist_from_high"]) else np.nan,
        })
    return pd.DataFrame(rows)
