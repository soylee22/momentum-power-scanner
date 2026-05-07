"""Share-class dedupe tests."""
from __future__ import annotations

import pandas as pd

from momentum.scanner import _dedupe_share_classes


def test_alphabet_keeps_cheaper_class():
    df = pd.DataFrame([
        dict(ticker="GOOGL", name="Alphabet Inc. (Class A)", price=181.50),
        dict(ticker="GOOG",  name="Alphabet Inc. (Class C)", price=180.10),
        dict(ticker="MSFT",  name="Microsoft Corporation",   price=520.00),
    ])
    out = _dedupe_share_classes(df)
    assert set(out["ticker"]) == {"GOOG", "MSFT"}, out["ticker"].tolist()


def test_berkshire_keeps_b():
    df = pd.DataFrame([
        dict(ticker="BRK-A", name="Berkshire Hathaway Inc. Class A", price=720_000.00),
        dict(ticker="BRK-B", name="Berkshire Hathaway Inc. Class B", price=480.00),
    ])
    out = _dedupe_share_classes(df)
    assert out["ticker"].iloc[0] == "BRK-B"


def test_fox_keeps_cheaper():
    df = pd.DataFrame([
        dict(ticker="FOX",  name="Fox Corporation (Class B)", price=55.10),
        dict(ticker="FOXA", name="Fox Corporation (Class A)", price=58.20),
    ])
    out = _dedupe_share_classes(df)
    assert out["ticker"].iloc[0] == "FOX"


def test_no_duplicates_passes_through():
    df = pd.DataFrame([
        dict(ticker="AAPL", name="Apple Inc.", price=200.00),
        dict(ticker="MSFT", name="Microsoft Corporation", price=520.00),
        dict(ticker="NVDA", name="NVIDIA Corporation", price=180.00),
    ])
    out = _dedupe_share_classes(df)
    assert len(out) == 3
    assert set(out["ticker"]) == {"AAPL", "MSFT", "NVDA"}


def test_empty_pool_safe():
    df = pd.DataFrame(columns=["ticker", "name", "price"])
    out = _dedupe_share_classes(df)
    assert out.empty
