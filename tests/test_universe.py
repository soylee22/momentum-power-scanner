"""Universe construction helpers."""
from __future__ import annotations

from momentum.universe import (
    MIN_MARKET_CAP_USD,
    _is_common_equity_name,
    _UK_INTL_TICKER_RE,
    _US_PREFERRED_TICKER_RE,
)


def test_min_cap_is_one_billion():
    assert MIN_MARKET_CAP_USD == 1_000_000_000


def test_common_equity_keeps_stocks_and_adrs():
    assert _is_common_equity_name("Apple Inc.")
    assert _is_common_equity_name("NVIDIA Corporation")
    assert _is_common_equity_name(
        "Ambev S.A. American Depositary Shares (Each representing 1 Common Share)"
    )
    assert _is_common_equity_name("Accenture plc")


def test_common_equity_drops_preferreds_warrants_notes():
    assert not _is_common_equity_name("EIDP, Inc. Preferred Stock $4.5")
    assert not _is_common_equity_name("MetLife, Inc. Preferred Series A")
    assert not _is_common_equity_name("Some Corp Warrant")
    assert not _is_common_equity_name(
        "American Financial Group Inc. 5.875% Subordinated Debentures"
    )


def test_us_preferred_ticker_pattern():
    assert _US_PREFERRED_TICKER_RE.search("CTA-PB")
    assert _US_PREFERRED_TICKER_RE.search("MET-PA")
    assert not _US_PREFERRED_TICKER_RE.search("BRK-B")  # class B common
    assert not _US_PREFERRED_TICKER_RE.search("AAPL")


def test_uk_intl_ticker_pattern():
    assert _UK_INTL_TICKER_RE.search("0DJN.L")
    assert not _UK_INTL_TICKER_RE.search("HSBA.L")
    assert not _UK_INTL_TICKER_RE.search("BP.L")
