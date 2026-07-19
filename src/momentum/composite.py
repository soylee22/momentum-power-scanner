"""Composite momentum ranker over Stage 2 survivors.

Mean of three percentile ranks within the survivor pool:

  - RS rating  (IBD-style, already 1-99 across full universe)
  - 52-week-high proximity  (closer = higher; we rank -1 * dist_from_high)
  - 1-year total return     (in local currency)

Headline top-N by composite (default 100 in the scanner).
"""
from __future__ import annotations

import pandas as pd


def rank_composite(survivors: pd.DataFrame) -> pd.DataFrame:
    """Augment the survivor DataFrame with composite ranks.

    Expected columns: ticker, rs_rating, price, high_52w, return_12m.
    Adds: dist_from_high, rank_rs, rank_prox, rank_1yr, composite,
    rank_overall.
    """
    df = survivors.copy()
    df["dist_from_high"] = (df["high_52w"] - df["price"]) / df["high_52w"]

    df["rank_rs"] = df["rs_rating"].rank(pct=True, method="average")
    df["rank_prox"] = (-df["dist_from_high"]).rank(pct=True, method="average")
    df["rank_1yr"] = df["return_12m"].rank(pct=True, method="average")

    df["composite"] = (df["rank_rs"] + df["rank_prox"] + df["rank_1yr"]) / 3.0
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank_overall"] = df.index + 1
    return df


def top_n(survivors: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    return rank_composite(survivors).head(n).copy()
