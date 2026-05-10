"""Fetch ~10 years of daily Close+Volume for the current S&P 500 + FTSE 100
universe, into data/prices_backtest.parquet.

Independent from the live cache at data/prices.parquet so the live scan
isn't disturbed. Idempotent: re-running tops up to the latest close.

Survivorship-bias note: we use the *current* index membership, which
systematically excludes tickers that got delisted or removed. The backtest
results will therefore over-estimate realistic forward returns. We're
deciding this is acceptable for a quick-and-honest first pass.
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from momentum.universe import load_universe  # noqa: E402

OUT = ROOT / "data" / "prices_backtest.parquet"
START = dt.date(2015, 1, 1)
BATCH = 60

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def _load_cached() -> pd.DataFrame:
    if OUT.exists():
        df = pd.read_parquet(OUT)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    return pd.DataFrame(columns=["ticker", "date", "close", "volume"])


def _save(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"])
    df.to_parquet(OUT, index=False)


def _download_batch(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "date", "close", "volume"])
    raw = yf.download(
        tickers=" ".join(tickers),
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    rows = []
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            if tk not in raw.columns.get_level_values(0):
                continue
            sub = raw[tk][["Close", "Volume"]].dropna()
            for d, r in sub.iterrows():
                rows.append({"ticker": tk, "date": d.date(), "close": float(r["Close"]), "volume": float(r["Volume"])})
    else:
        sub = raw[["Close", "Volume"]].dropna()
        tk = tickers[0]
        for d, r in sub.iterrows():
            rows.append({"ticker": tk, "date": d.date(), "close": float(r["Close"]), "volume": float(r["Volume"])})
    return pd.DataFrame(rows)


def main() -> None:
    universe = load_universe()
    tickers = sorted(universe["ticker"].astype(str).unique().tolist())
    print(f"Universe: {len(tickers)} tickers")

    cached = _load_cached()
    last_by_ticker = cached.groupby("ticker")["date"].max().to_dict() if not cached.empty else {}
    today = dt.date.today()

    all_new: list[pd.DataFrame] = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        # Per-batch start = min last-cached date + 1, or START if any ticker fresh
        starts = [last_by_ticker.get(tk) for tk in batch]
        fresh = [tk for tk, s in zip(batch, starts) if s is None]
        if fresh:
            start = START
        else:
            start = max(starts) + dt.timedelta(days=1)
            if start >= today:
                continue
        print(f"  batch {i//BATCH + 1}/{(len(tickers) + BATCH - 1)//BATCH}: {len(batch)} tickers from {start}")
        try:
            df = _download_batch(batch, start, today)
            if not df.empty:
                all_new.append(df)
        except Exception as exc:
            print(f"    batch failed: {exc!r}")

    if all_new:
        merged = pd.concat([cached, *all_new], ignore_index=True)
    else:
        merged = cached
    _save(merged)

    final = pd.read_parquet(OUT)
    print(f"Saved {len(final):,} rows, {final['ticker'].nunique()} tickers, range {final['date'].min()} -> {final['date'].max()}")


if __name__ == "__main__":
    main()
