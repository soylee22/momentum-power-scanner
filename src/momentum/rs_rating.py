"""IBD-style Relative Strength rating.

Quarter-weighted return composite, percentile-ranked across the universe
to a 1-99 scale. Stocks with insufficient history get NaN.

  weighted_perf = 0.4 * r3m + 0.2 * r6m + 0.2 * r9m + 0.2 * r12m

Returns are computed in local currency (USD/GBP) so FX noise does not
pollute the momentum signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .minervini import TrendFeatures


def compute_weighted_perf(f: TrendFeatures) -> float:
    parts = [f.return_3m, f.return_6m, f.return_9m, f.return_12m]
    if any(np.isnan(p) for p in parts):
        return float("nan")
    return 0.4 * f.return_3m + 0.2 * f.return_6m + 0.2 * f.return_9m + 0.2 * f.return_12m


def rs_rating_table(features: list[TrendFeatures]) -> pd.DataFrame:
    """Compute weighted perf and 1-99 RS rating for each ticker.

    Percentile rank across the universe; NaN-perf rows get NaN rating
    (and so will fail gate 8).
    """
    rows = []
    for f in features:
        rows.append({
            "ticker": f.ticker,
            "weighted_perf": compute_weighted_perf(f),
        })
    df = pd.DataFrame(rows)
    valid = df["weighted_perf"].notna()
    pct = df.loc[valid, "weighted_perf"].rank(pct=True, method="average")
    df.loc[valid, "rs_rating"] = (1.0 + 98.0 * pct).round(1)
    return df
