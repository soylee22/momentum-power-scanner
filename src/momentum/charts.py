"""Per-ticker matplotlib charts in the Palantir dark aesthetic.

For each top-10 name, render a 14-month price chart with:
  - Price line (bone)
  - 50d, 150d, 200d SMAs (mute, hairline, hairline)
  - 52-week high/low horizontal bands (hairline_dk dotted)
  - Today's data point ringed in emerald
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fetch import load_prices
from .palantir import PL, mpl_dark_rcparams

ROOT = Path(__file__).resolve().parent.parent.parent
CHART_DIR = ROOT / "outputs" / "charts"


def _draw_one(ticker: str, prices: pd.DataFrame, name: str, idx_label: str, dest: Path) -> None:
    plt.rcParams.update(mpl_dark_rcparams())
    fig = plt.figure(figsize=(12, 6.6), dpi=160, facecolor=PL.ink)
    ax = fig.add_axes([0.075, 0.13, 0.90, 0.65])  # header above, legend below

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma150"] = df["close"].rolling(150).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df = df.iloc[-252:].reset_index(drop=True)  # last year on display

    ax.plot(df["date"], df["close"], color=PL.bone, linewidth=1.6, label="Close")
    ax.plot(df["date"], df["sma50"], color=PL.mute, linewidth=1.0, alpha=0.85, label="50d SMA")
    ax.plot(df["date"], df["sma150"], color=PL.hairline_lt, linewidth=0.9, alpha=0.65,
            linestyle="--", label="150d SMA")
    ax.plot(df["date"], df["sma200"], color=PL.hairline_lt, linewidth=0.9, alpha=0.5,
            linestyle=":", label="200d SMA")

    # 52-week bands
    high_52w = float(df["close"].max())
    low_52w = float(df["close"].min())
    ax.axhline(high_52w, color=PL.hairline_dk, linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axhline(low_52w, color=PL.hairline_dk, linewidth=0.7, linestyle=":", alpha=0.6)
    ax.text(df["date"].iloc[0], high_52w, "  52w HIGH", color=PL.mute, fontsize=8,
            va="bottom", ha="left")
    ax.text(df["date"].iloc[0], low_52w, "  52w LOW", color=PL.mute, fontsize=8,
            va="top", ha="left")

    # Hero point
    last_x, last_y = df["date"].iloc[-1], df["close"].iloc[-1]
    ax.scatter([last_x], [last_y], s=200, facecolors="none",
               edgecolors=PL.accent_green, linewidths=1.5, zorder=5)
    ax.scatter([last_x], [last_y], s=28, color=PL.accent_green, zorder=6)

    # Header band above chart (chip on left, hero number on right)
    fig.text(
        0.075, 0.93,
        f"MOMENTUM POWER     |     {idx_label}",
        color=PL.mute,
        fontsize=9,
        weight="medium",
    )
    fig.text(
        0.075, 0.86,
        f"{ticker}",
        color=PL.bone,
        fontsize=24,
        weight="regular",
    )
    # Name as subtitle next to ticker, vertically aligned to ticker baseline
    fig.text(
        0.155, 0.86,
        f" ·  {name}",
        color=PL.mute,
        fontsize=14,
    )
    # Hero number top-right
    fig.text(
        0.965, 0.87,
        f"{last_y:,.2f}",
        color=PL.accent_green,
        fontsize=22,
        weight="semibold",
        ha="right",
    )

    # Axes cosmetics
    ax.spines["bottom"].set_color(PL.hairline_dk)
    ax.tick_params(axis="x", colors=PL.mute, labelsize=9)
    ax.tick_params(axis="y", colors=PL.mute, labelsize=9)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        frameon=False, fontsize=9, labelcolor=PL.mute, ncol=4,
    )
    ax.margins(x=0.01)

    fig.savefig(dest, facecolor=PL.ink, pad_inches=0.0)
    plt.close(fig)


def render_top_charts(top: pd.DataFrame, full_table: pd.DataFrame) -> None:
    if top is None or top.empty:
        return
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    # Clean stale charts so the dashboard never lists exited tickers.
    for f in CHART_DIR.glob("*.png"):
        f.unlink()

    # Use cached prices if present, otherwise fetch fresh for the headline list.
    tickers = top["ticker"].tolist()
    prices = load_prices(tickers)

    for _, r in top.iterrows():
        tk = r["ticker"]
        if tk not in prices:
            continue
        idx_label = str(r.get("index") or r.get("country") or "")
        _draw_one(
            tk,
            prices[tk],
            str(r.get("name", "")),
            idx_label,
            CHART_DIR / f"{tk.replace('.', '_')}.png",
        )
