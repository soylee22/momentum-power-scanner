"""Equity universe: every US + UK stock Yahoo lists at ≥ $1bn market cap.

Source: yfinance EquityQuery screener (Yahoo Finance), paginated.
  - US: region=us, exchanges NMS/NYQ/ASE/NGM/NCM
  - UK: region=gb, exchange LSE
  - Hard floor: marketCap ≥ $1bn (post-filter; Yahoo's own floor is leaky)
  - Drop preferreds / warrants / notes and other non-common junk
  - Curated LSE alts (metals, factors) kept outside the cap filter

Cached as parquet at data/universe.parquet, refreshed if older than 7 days.
"""
from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path

import pandas as pd
from yfinance import EquityQuery, screen

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = ROOT / "data" / "universe.parquet"
CACHE_TTL_DAYS = 7

MIN_MARKET_CAP_USD = 1_000_000_000  # $1bn hard floor
_SCREEN_PAGE = 250  # Yahoo hard max per request
_SCREEN_PAUSE_S = 0.12

# Primary US equity venues (skip OTC / pinks / weird boards).
_US_EXCHANGES = ("NMS", "NYQ", "ASE", "NGM", "NCM")
_UK_EXCHANGES = ("LSE",)

_EXCLUDE_NAME_RE = re.compile(
    r"(?i)\b("
    r"preferred|preference|warrant|warrants|right|rights|"
    r"unit(?!ed)|units|"
    r"note|notes|debenture|debentures|bond|bonds|"
    r"etf|etn|etp"
    r")\b"
)
# Preferred depositary / coupon paper that slips past the simple word list.
_EXCLUDE_NAME_RE2 = re.compile(
    r"(?i)("
    r"depositary shares each representing|"
    r"fixed[- ]rate senior notes|"
    r"subordinated debentures|"
    r"cumulative redeemable|"
    r"%\s*(series|fixed|senior|subordinated|cum)"
    r")"
)
# US preferred tickers: FOO-PA, FOO-PB, FOO-P. Not FOO-A (class A common).
_US_PREFERRED_TICKER_RE = re.compile(r"-[P][A-Z]?$")
# LSE international / order-book lines often start with a digit (e.g. 0DJN.L).
_UK_INTL_TICKER_RE = re.compile(r"^\d")


def _is_common_equity_name(name: str) -> bool:
    if not name or not str(name).strip():
        return False
    n = str(name)
    if _EXCLUDE_NAME_RE.search(n):
        return False
    if _EXCLUDE_NAME_RE2.search(n):
        return False
    return True


def _screen_page(query: EquityQuery, offset: int, size: int = _SCREEN_PAGE) -> dict:
    return screen(
        query,
        offset=offset,
        size=size,
        sortField="intradaymarketcap",
        sortAsc=False,
    )


def _equity_query(region: str, exchanges: tuple[str, ...]) -> EquityQuery:
    ops = [
        EquityQuery("gte", ["intradaymarketcap", MIN_MARKET_CAP_USD]),
        EquityQuery("eq", ["region", region]),
    ]
    if len(exchanges) == 1:
        ops.append(EquityQuery("eq", ["exchange", exchanges[0]]))
    else:
        ops.append(EquityQuery("is-in", ["exchange", *exchanges]))
    return EquityQuery("and", ops)


def _fetch_yahoo_region(
    region: str,
    exchanges: tuple[str, ...],
    *,
    country: str,
    currency: str,
    index_label: str,
) -> pd.DataFrame:
    """Paginate Yahoo's equity screener for one region and normalise rows."""
    query = _equity_query(region, exchanges)
    offset = 0
    raw: list[dict] = []
    total: int | None = None

    while True:
        resp = _screen_page(query, offset=offset)
        if total is None:
            total = int(resp.get("total") or 0)
        quotes = resp.get("quotes") or []
        if not quotes:
            break
        raw.extend(quotes)
        offset += len(quotes)
        print(f"  Yahoo {region.upper()}: {len(raw)}/{total or '?'}")
        if total and offset >= total:
            break
        if len(quotes) < _SCREEN_PAGE:
            break
        time.sleep(_SCREEN_PAUSE_S)

    rows: list[dict] = []
    for q in raw:
        symbol = str(q.get("symbol") or "").strip()
        if not symbol:
            continue
        if (q.get("quoteType") or "EQUITY").upper() != "EQUITY":
            continue

        mc = q.get("marketCap")
        try:
            market_cap = float(mc) if mc is not None else 0.0
        except (TypeError, ValueError):
            market_cap = 0.0
        # Yahoo's gte filter leaks at the tail — enforce the floor ourselves.
        if market_cap < MIN_MARKET_CAP_USD:
            continue

        name = (
            str(q.get("longName") or q.get("shortName") or q.get("displayName") or "")
            .strip()
        )
        if not _is_common_equity_name(name):
            continue
        if not _is_common_equity_name(str(q.get("shortName") or "")):
            continue

        base = symbol[:-2] if symbol.endswith(".L") else symbol
        if country == "US" and _US_PREFERRED_TICKER_RE.search(base):
            continue
        if country == "UK" and _UK_INTL_TICKER_RE.search(symbol):
            continue

        sector = str(q.get("sector") or "").strip() or "Unknown"
        industry = str(q.get("industry") or "").strip() or sector
        exchange = str(q.get("fullExchangeName") or q.get("exchange") or "").strip()

        rows.append({
            "ticker": symbol,
            "name": name,
            "sector": sector,
            "industry": industry,
            "exchange": exchange or ("LSE" if country == "UK" else "NYSE/NASDAQ"),
            "country": country,
            "currency": str(q.get("currency") or currency),
            "index": index_label,
            "market_cap": market_cap,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Yahoo {region} screener returned 0 names after filters")
    return df.drop_duplicates("ticker", keep="first").reset_index(drop=True)


def _fetch_us() -> pd.DataFrame:
    return _fetch_yahoo_region(
        "us",
        _US_EXCHANGES,
        country="US",
        currency="USD",
        index_label="US $1bn+",
    )


def _fetch_uk() -> pd.DataFrame:
    return _fetch_yahoo_region(
        "gb",
        _UK_EXCHANGES,
        country="UK",
        currency="GBP",
        index_label="UK $1bn+",
    )


def _extras() -> pd.DataFrame:
    """Curated LSE alts (outside the equity market-cap filter)."""
    rows = [
        ("SGLN.L", "iShares Physical Gold", "Precious Metals", "Commodity ETF"),
        ("SSLN.L", "iShares Physical Silver", "Precious Metals", "Commodity ETF"),
        ("SPLT.L", "iShares Physical Platinum", "Precious Metals", "Commodity ETF"),
        ("SMH.L", "VanEck Semiconductor", "Semiconductors", "Thematic ETF"),
        ("DFND.L", "Future Generations Defence", "Aerospace & Defence", "Thematic ETF"),
        ("IUFS.L", "iShares S&P 500 Financials", "Financials", "Sector ETF"),
        ("IUCM.L", "iShares S&P 500 Comm Services", "Communication Services", "Sector ETF"),
        ("IUUS.L", "iShares S&P 500 Utilities", "Utilities", "Sector ETF"),
        ("IUSP.L", "iShares S&P 500 Real Estate", "Real Estate", "Sector ETF"),
        ("IUMO.L", "iShares Edge MSCI USA Momentum", "Factor", "Factor ETF"),
        ("IUQA.L", "iShares Edge MSCI USA Quality", "Factor", "Factor ETF"),
        ("IUVL.L", "iShares Edge MSCI USA Value", "Factor", "Factor ETF"),
        ("CUS1.L", "iShares S&P 600 Small Cap", "Factor", "Factor ETF"),
        ("IEMO.L", "iShares Edge MSCI Europe Momentum", "Factor", "Factor ETF"),
        ("IEQU.L", "iShares Edge MSCI Europe Quality", "Factor", "Factor ETF"),
        ("IEVL.L", "iShares Edge MSCI Europe Value", "Factor", "Factor ETF"),
        ("WSML.L", "iShares MSCI World Small Cap", "Factor", "Factor ETF"),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "name", "sector", "index"])
    df["industry"] = df["sector"]
    df["exchange"] = "LSE"
    df["country"] = "UK"
    df["currency"] = "GBP"
    df["market_cap"] = float("nan")
    return df[
        [
            "ticker",
            "name",
            "sector",
            "industry",
            "exchange",
            "country",
            "currency",
            "index",
            "market_cap",
        ]
    ]


def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    age = dt.datetime.now() - dt.datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
    return age < dt.timedelta(days=CACHE_TTL_DAYS)


def load_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Return US + UK equities with market cap ≥ $1bn, plus curated alts.

    Cached for 7 days.
    """
    if not force_refresh and _cache_is_fresh():
        return pd.read_parquet(CACHE_PATH)

    print(f"  building universe via Yahoo screener (min cap ${MIN_MARKET_CAP_USD / 1e9:.0f}bn)...")
    us = _fetch_us()
    print(f"  US ≥ $1bn: {len(us)}")
    uk = _fetch_uk()
    print(f"  UK ≥ $1bn: {len(uk)}")
    alt = _extras()

    combined = pd.concat([us, uk, alt], ignore_index=True).drop_duplicates("ticker")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(CACHE_PATH, index=False)
    print(f"  universe total: {len(combined)} tickers")
    return combined


def main() -> None:
    df = load_universe(force_refresh=True)
    us_n = int((df["country"] == "US").sum())
    uk_eq = int(
        ((df["country"] == "UK") & ~df["index"].astype(str).str.contains("ETF")).sum()
    )
    alt_n = int(df["index"].astype(str).str.contains("ETF").sum())
    print(f"Loaded {len(df)} tickers (US {us_n}, UK equities {uk_eq}, alts {alt_n})")
    print(df["index"].value_counts().to_string())
    print(df.head())


if __name__ == "__main__":
    main()
