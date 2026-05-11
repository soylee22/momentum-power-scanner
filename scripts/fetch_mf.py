"""Fetch the MarketFighter UCITS-ETF universe + SPY (cash-overlay signal)
into data/prices_mf.parquet. ~10 years of daily Close (most LSE iShares
UCITS launched 2014-2016).
"""
from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "prices_mf.parquet"

# Factor basket (8) — LSE-listed UCITS proxies for MSCI USA/Europe x M/Q/V/Size
FACTOR_TICKERS = [
    "IUMO.L",  # USA Momentum
    "IUQA.L",  # USA Quality
    "IUVL.L",  # USA Value
    "CUS1.L",  # USA Small Cap (S&P 600 proxy — closest UCITS available)
    "IEMO.L",  # Europe Momentum
    "IEQU.L",  # Europe Quality
    "IEVL.L",  # Europe Value
    "WSML.L",  # World Small Cap (substitute — Europe-only Small Cap UCITS not on yfinance)
]

# Sector basket (7) — iShares S&P 500 sector UCITS
SECTOR_TICKERS = [
    "IUIT.L",  # IT
    "IHCU.L",  # Health Care
    "IUES.L",  # Energy
    "IUCD.L",  # Consumer Discretionary
    "IUCS.L",  # Consumer Staples
    "IUIS.L",  # Industrials
    "IUMS.L",  # Materials
]

# Cash-overlay signal
OVERLAY = ["SPY"]

ALL = FACTOR_TICKERS + SECTOR_TICKERS + OVERLAY
START = dt.date(2014, 1, 1)

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def main() -> None:
    print(f"Fetching {len(ALL)} tickers from {START}...")
    rows = []
    for tk in ALL:
        try:
            raw = yf.download(tk, start=START.isoformat(), auto_adjust=True,
                              progress=False, threads=False)
        except Exception as exc:
            print(f"  FAIL {tk}: {exc!r}")
            continue
        if raw.empty:
            print(f"  EMPTY {tk}")
            continue
        s = raw["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        v = raw["Volume"] if "Volume" in raw.columns else None
        if isinstance(v, pd.DataFrame):
            v = v.iloc[:, 0]
        for d, c in s.items():
            rows.append({
                "ticker": tk,
                "date": d.date(),
                "close": float(c),
                "volume": float(v.loc[d]) if v is not None and d in v.index else 0.0,
            })
        print(f"  ok {tk}: {len(s)} bars from {s.index[0].date()} to {s.index[-1].date()}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")
    print(df.groupby("ticker").agg(bars=("date", "size"),
                                    start=("date", "min"),
                                    end=("date", "max")))


if __name__ == "__main__":
    main()
