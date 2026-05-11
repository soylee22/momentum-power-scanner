"""S&P 500 + FTSE 100 constituent fetcher with weekly disk cache.

Reads Wikipedia (single source we know is structured + reliable + free).
Cached as parquet at data/cache/universe.parquet, refreshed if older than 7 days.
"""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = ROOT / "data" / "universe.parquet"
CACHE_TTL_DAYS = 7

_HEADERS = {
    "User-Agent": "momentum-power-scanner/0.1 (personal research; +github.com/soylee22)"
}

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FTSE100_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"


def _read_wiki_tables(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def _fetch_sp500() -> pd.DataFrame:
    tables = _read_wiki_tables(SP500_URL)
    df = tables[0].copy()
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "industry",
    })
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B for yfinance
    df["exchange"] = "NYSE/NASDAQ"
    df["country"] = "US"
    df["currency"] = "USD"
    df["index"] = "S&P 500"
    return df[["ticker", "name", "sector", "industry", "exchange", "country", "currency", "index"]]


def _flatten_columns(t: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns down to single strings."""
    t = t.copy()
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [
            " ".join([str(c) for c in col if str(c) != "nan"]).strip()
            for col in t.columns.values
        ]
    else:
        t.columns = [str(c) for c in t.columns]
    return t


def _fetch_ftse100() -> pd.DataFrame:
    tables = _read_wiki_tables(FTSE100_URL)
    # The constituents table shape on Wikipedia varies; pick the one with a "Ticker" column.
    df = None
    for raw in tables:
        t = _flatten_columns(raw)
        cols_lower = [c.lower() for c in t.columns]
        if any("ticker" in c or "epic" in c for c in cols_lower) and any(
            "company" in c or "name" in c for c in cols_lower
        ):
            df = t
            break
    if df is None:
        raise RuntimeError("FTSE 100 constituents table not found on Wikipedia page")

    # Normalise columns (Wikipedia sometimes uses "EPIC", sometimes "Ticker")
    rename = {}
    seen_ticker = seen_name = seen_sector = False
    for c in df.columns:
        cl = c.lower()
        if not seen_ticker and ("ticker" in cl or "epic" in cl or cl == "code"):
            rename[c] = "ticker"
            seen_ticker = True
        elif not seen_name and ("company" in cl or "constituent" in cl or "name" in cl):
            rename[c] = "name"
            seen_name = True
        elif not seen_sector and "sector" in cl:
            rename[c] = "sector"
            seen_sector = True
    df = df.rename(columns=rename)

    if "sector" not in df.columns:
        df["sector"] = ""
    df["industry"] = df["sector"]

    df = df[["ticker", "name", "sector", "industry"]].copy()
    # FTSE tickers on yfinance need .L; Wikipedia drops the suffix.
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False) + ".L"
    df["exchange"] = "LSE"
    df["country"] = "UK"
    df["currency"] = "GBP"  # Most FTSE 100 stocks quote in GBp (pence) but yfinance returns pence; we treat returns as % so this is fine.
    df["index"] = "FTSE 100"
    return df


def _extras() -> pd.DataFrame:
    """Curated alts that aren't in the equity indices but trade on the LSE
    and are eligible for the Stage 2 trend filter the same way any other
    instrument with a daily price series is. Physical-backed ETCs only.

    Lee's request: gold + silver in the scanner so commodities can show
    up alongside equities when they're trending.
    """
    rows = [
        # Precious metals — physical-backed iShares ETCs, GBP-denominated.
        # All three available on Trading 212 (Lee's broker).
        ("SGLN.L", "iShares Physical Gold",       "Precious Metals",      "Commodity ETF"),
        ("SSLN.L", "iShares Physical Silver",     "Precious Metals",      "Commodity ETF"),
        ("SPLT.L", "iShares Physical Platinum",   "Precious Metals",      "Commodity ETF"),
        # Thematic / sub-sector — let the scanner surface these when they're trending.
        ("SMH.L",  "VanEck Semiconductor",        "Semiconductors",       "Thematic ETF"),
        ("DFND.L", "Future Generations Defence",  "Aerospace & Defence",  "Thematic ETF"),
        # Missing GICS sectors — full sector coverage so themes can compete with stocks.
        ("IUFS.L", "iShares S&P 500 Financials",  "Financials",           "Sector ETF"),
        ("IUCM.L", "iShares S&P 500 Comm Services","Communication Services","Sector ETF"),
        ("IUUS.L", "iShares S&P 500 Utilities",   "Utilities",            "Sector ETF"),
        ("IUSP.L", "iShares S&P 500 Real Estate", "Real Estate",          "Sector ETF"),
        # MarketFighter factor basket — let the live scanner show today's pick.
        ("IUMO.L", "iShares Edge MSCI USA Momentum",  "Factor",  "Factor ETF"),
        ("IUQA.L", "iShares Edge MSCI USA Quality",   "Factor",  "Factor ETF"),
        ("IUVL.L", "iShares Edge MSCI USA Value",     "Factor",  "Factor ETF"),
        ("CUS1.L", "iShares S&P 600 Small Cap",       "Factor",  "Factor ETF"),
        ("IEMO.L", "iShares Edge MSCI Europe Momentum","Factor", "Factor ETF"),
        ("IEQU.L", "iShares Edge MSCI Europe Quality","Factor",  "Factor ETF"),
        ("IEVL.L", "iShares Edge MSCI Europe Value",  "Factor",  "Factor ETF"),
        ("WSML.L", "iShares MSCI World Small Cap",    "Factor",  "Factor ETF"),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "name", "sector", "index"])
    df["industry"] = df["sector"]
    df["exchange"] = "LSE"
    df["country"] = "UK"
    df["currency"] = "GBP"
    return df[["ticker", "name", "sector", "industry", "exchange", "country", "currency", "index"]]


def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    age = dt.datetime.now() - dt.datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
    return age < dt.timedelta(days=CACHE_TTL_DAYS)


def load_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Return the combined S&P 500 + FTSE 100 + alts universe.

    Cached for 7 days.
    """
    if not force_refresh and _cache_is_fresh():
        return pd.read_parquet(CACHE_PATH)

    sp = _fetch_sp500()
    ft = _fetch_ftse100()
    alt = _extras()
    combined = pd.concat([sp, ft, alt], ignore_index=True).drop_duplicates("ticker")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(CACHE_PATH, index=False)
    return combined


def main() -> None:
    df = load_universe(force_refresh=True)
    print(f"Loaded {len(df)} tickers ({(df['index'] == 'S&P 500').sum()} S&P, "
          f"{(df['index'] == 'FTSE 100').sum()} FTSE)")
    print(df.head())


if __name__ == "__main__":
    main()
