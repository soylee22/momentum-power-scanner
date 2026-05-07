"""Incrementally-cached yfinance downloader.

Single growing parquet at `data/prices.parquet` (long format, columns:
ticker, date, close, volume). On each run we top it up: for every
requested ticker, find the last cached date (or fall back to ~14 months
ago for fresh tickers), download only bars after that date, and append.

This means:
  - First run for ~600 tickers takes a few minutes
  - Every subsequent run is fast (only days since last run downloaded)
  - History compounds across runs, useful for later backtests
  - Old tickers that drop out of the universe are kept (idempotent reads)
"""
from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent.parent
PRICES_PATH = ROOT / "data" / "prices.parquet"
INITIAL_LOOKBACK_DAYS = 425  # ~14 months for a fresh ticker
MIN_BARS = 220               # need >= 200d MA + buffer

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def _read_cache() -> pd.DataFrame:
    if not PRICES_PATH.exists():
        return pd.DataFrame(columns=["ticker", "date", "close", "volume"])
    df = pd.read_parquet(PRICES_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _write_cache(df: pd.DataFrame) -> None:
    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(PRICES_PATH, index=False)


def _download_batch(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download daily Adj Close + Volume for a batch in one call."""
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
    if raw.empty:
        return pd.DataFrame(columns=["ticker", "date", "close", "volume"])

    rows = []
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            if tk not in raw.columns.get_level_values(0):
                continue
            sub = raw[tk]
            if "Close" not in sub.columns or sub["Close"].dropna().empty:
                continue
            sub = sub[["Close", "Volume"]].dropna(subset=["Close"]).reset_index()
            sub.columns = ["date", "close", "volume"]
            sub["ticker"] = tk
            rows.append(sub)
    else:
        sub = raw[["Close", "Volume"]].dropna(subset=["Close"]).reset_index()
        sub.columns = ["date", "close", "volume"]
        sub["ticker"] = tickers[0]
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "close", "volume"])

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out[["ticker", "date", "close", "volume"]]


def update_cache(
    tickers: Iterable[str],
    asof: dt.date | None = None,
    batch_size: int = 75,
    initial_lookback_days: int = INITIAL_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Top up the price cache for the given tickers and return the full cache.

    For each ticker:
      - if not in cache: download initial_lookback_days of history
      - if last_date < asof: download from last_date+1 to asof+1 (yfinance end is exclusive)
      - if up to date: skip
    """
    asof = asof or dt.date.today()
    end = asof + dt.timedelta(days=1)

    tickers = list(dict.fromkeys(tickers))  # de-dup, preserve order
    cache = _read_cache()

    last_dates = (
        cache.groupby("ticker")["date"].max()
        if not cache.empty else pd.Series(dtype="object")
    )

    # Group tickers by required start-date so each batch shares a window
    fresh_start = asof - dt.timedelta(days=initial_lookback_days)
    by_start: dict[dt.date, list[str]] = {}
    skipped = 0
    for tk in tickers:
        last = last_dates.get(tk)
        if last is None:
            start = fresh_start
        elif last >= asof:
            skipped += 1
            continue  # already up to date
        else:
            start = last + dt.timedelta(days=1)
        by_start.setdefault(start, []).append(tk)

    if skipped:
        print(f"  cache up-to-date for {skipped}/{len(tickers)} tickers")

    new_chunks: list[pd.DataFrame] = []
    for start, group in by_start.items():
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            try:
                sub = _download_batch(batch, start, end)
            except Exception as exc:
                print(f"  fetch batch failed (start={start}): {exc!r}")
                continue
            if not sub.empty:
                new_chunks.append(sub)
        if group:
            print(f"  topped up {len(group)} tickers from {start.isoformat()}")

    if new_chunks:
        new_df = pd.concat(new_chunks, ignore_index=True)
        cache = pd.concat([cache, new_df], ignore_index=True)
        _write_cache(cache)

    return cache


def load_prices(
    tickers: Iterable[str],
    asof: dt.date | None = None,
    batch_size: int = 75,
) -> dict[str, pd.DataFrame]:
    """Top up the cache and return per-ticker DataFrames usable by features.

    Tickers with fewer than MIN_BARS rows are silently dropped (e.g.
    very recent IPOs).
    """
    cache = update_cache(tickers, asof=asof, batch_size=batch_size)
    if cache.empty:
        return {}

    tickers_set = set(tickers)
    out: dict[str, pd.DataFrame] = {}
    for tk, grp in cache.groupby("ticker", sort=False):
        if tk not in tickers_set:
            continue
        g = grp.sort_values("date").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        out[tk] = g
    return out
