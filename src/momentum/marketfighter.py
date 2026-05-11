"""MarketFighter-style two-basket monthly momentum rotation.

Each month:
  1. In FACTOR basket, pick the ticker with best 12-month return → invest 50%.
  2. In SECTOR basket, pick the ticker with best 12-month return → invest 50%.
  3. Cash overlay (optional): if SPY closed BELOW its 10-month SMA at month-end,
     that leg goes to cash for the next month instead of buying the leader.

Returns the same BacktestResult shape as run_backtest for easy comparison.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, _compute_metrics


# Canonical LSE-listed UCITS universe (T212-accessible)
FACTOR_BASKET = [
    "IUMO.L",  # USA Momentum
    "IUQA.L",  # USA Quality
    "IUVL.L",  # USA Value
    "CUS1.L",  # USA Small Cap (S&P 600)
    "IEMO.L",  # Europe Momentum
    "IEQU.L",  # Europe Quality
    "IEVL.L",  # Europe Value
    "WSML.L",  # World Small Cap (Europe-only Small Cap UCITS not yfinance-accessible)
]

SECTOR_BASKET = [
    "IUIT.L",  # IT
    "IHCU.L",  # Health Care
    "IUES.L",  # Energy
    "IUCD.L",  # Cons Discretionary
    "IUCS.L",  # Cons Staples
    "IUIS.L",  # Industrials
    "IUMS.L",  # Materials
]

# Expanded basket — full 11 GICS sectors + pure-semi thematic. Lee's request.
# SMH.L only has data from Dec 2020, so it'll register NaN-signal pre-Dec-2021.
SECTOR_BASKET_EXPANDED = SECTOR_BASKET + [
    "IUFS.L",  # Financials
    "IUCM.L",  # Communication Services
    "IUUS.L",  # Utilities
    "IUSP.L",  # Real Estate
    "SMH.L",   # Semis pure-play (VanEck UCITS)
]


@dataclass
class MFConfig:
    lookback_months: int = 12
    overlay_sma_months: int | None = 10  # None = no cash overlay
    overlay_ticker: str = "SPY"
    rebalance_freq: Literal["monthly"] = "monthly"  # only monthly supported
    transaction_cost_bps: float = 5.0
    starting_capital: float = 100_000.0
    start: dt.date = dt.date(2019, 4, 1)
    end: dt.date | None = None


# Helpers ----------------------------------------------------------------------

def _wide(prices_long: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    df = prices_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="ticker", values=col, aggfunc="last").sort_index()


def _last_trading_day_of_month(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(dates, index=dates)
    return sorted(s.groupby([dates.year, dates.month]).last().tolist())


def _trailing_return(prices: pd.DataFrame, asof: pd.Timestamp, months: int) -> pd.Series:
    """For each column (ticker), return = price_today / price_n_months_ago - 1."""
    lookback_dates = prices.index[prices.index <= asof]
    if len(lookback_dates) == 0:
        return pd.Series(dtype=float)
    today = lookback_dates[-1]
    target = today - pd.DateOffset(months=months)
    earlier_dates = prices.index[prices.index <= target]
    if len(earlier_dates) == 0:
        return pd.Series(dtype=float)
    earlier = earlier_dates[-1]
    return prices.loc[today] / prices.loc[earlier] - 1.0


def _overlay_in_market(spy_wide: pd.Series, asof: pd.Timestamp, sma_months: int) -> bool:
    """SPY close > N-month SMA at asof. True = stay in market. False = go cash."""
    valid = spy_wide.loc[:asof].dropna()
    if len(valid) < 21 * sma_months:
        return True  # not enough data — default to in-market
    today = valid.iloc[-1]
    sma = valid.iloc[-21 * sma_months:].mean()
    return today > sma


# Main engine ------------------------------------------------------------------

def compute_live_picks(
    prices_long: pd.DataFrame,
    asof: dt.date | None = None,
    lookback_months: int = 12,
    factor_basket: list[str] | None = None,
    sector_basket: list[str] | None = None,
    overlay_ticker: str = "SPY",
    overlay_sma_months: int | None = 10,
) -> dict:
    """Compute today's MarketFighter pick from the live price feed.

    Returns:
      {
        "asof": date,
        "factor_pick":  {"ticker": str, "return_12m": float, ...} or None for cash,
        "sector_pick":  {"ticker": str, ...} or None for cash,
        "factor_rankings": [(ticker, return_12m), ...],
        "sector_rankings": [(ticker, return_12m), ...],
        "overlay_in_market": bool,
      }
    """
    factor = factor_basket or FACTOR_BASKET
    sector = sector_basket or SECTOR_BASKET
    prices_w = _wide(prices_long, "close")
    if asof is None:
        asof_ts = prices_w.index[-1]
    else:
        asof_ts = pd.Timestamp(asof)
        valid = prices_w.index[prices_w.index <= asof_ts]
        asof_ts = valid[-1] if len(valid) else prices_w.index[-1]

    in_market = True
    if overlay_sma_months and overlay_ticker in prices_w.columns:
        in_market = _overlay_in_market(prices_w[overlay_ticker], asof_ts, overlay_sma_months)

    def _rank(basket: list[str]) -> list[tuple[str, float]]:
        rets = _trailing_return(prices_w[[t for t in basket if t in prices_w.columns]], asof_ts, lookback_months)
        rets = rets.dropna().sort_values(ascending=False)
        return [(t, float(r)) for t, r in rets.items()]

    factor_rankings = _rank(factor)
    sector_rankings = _rank(sector)

    factor_pick = None if not in_market or not factor_rankings else {
        "ticker": factor_rankings[0][0],
        "return_12m": factor_rankings[0][1],
    }
    sector_pick = None if not in_market or not sector_rankings else {
        "ticker": sector_rankings[0][0],
        "return_12m": sector_rankings[0][1],
    }
    return {
        "asof": asof_ts.date().isoformat(),
        "in_market": in_market,
        "factor_pick": factor_pick,
        "sector_pick": sector_pick,
        "factor_rankings": factor_rankings,
        "sector_rankings": sector_rankings,
    }


def run_marketfighter(
    prices_long: pd.DataFrame,
    config: MFConfig,
    benchmarks: dict[str, pd.Series] | None = None,
    factor_basket: list[str] | None = None,
    sector_basket: list[str] | None = None,
) -> BacktestResult:
    factor = factor_basket or FACTOR_BASKET
    sector = sector_basket or SECTOR_BASKET
    prices = _wide(prices_long, "close")
    spy = prices[config.overlay_ticker] if config.overlay_ticker in prices.columns else None
    if spy is None and config.overlay_sma_months is not None:
        raise ValueError(f"overlay_ticker {config.overlay_ticker} missing from prices")

    # Universe of tickers we'll actually trade
    universe = [t for t in factor + sector if t in prices.columns]
    if len(universe) < len(factor) + len(sector):
        missing = set(factor + sector) - set(universe)
        print(f"warning: missing from prices: {missing}")

    end = config.end or prices.index.max().date()
    rebal_dates = [
        d for d in _last_trading_day_of_month(prices.index)
        if config.start <= d.date() <= end
    ]
    if not rebal_dates:
        raise ValueError("no rebalance dates")

    cost = config.transaction_cost_bps / 10_000.0
    # Two parallel sub-portfolios, half each
    leg_nav = {"factor": config.starting_capital / 2, "sector": config.starting_capital / 2}
    leg_holding = {"factor": None, "sector": None}  # currently held ticker (None = cash)
    leg_shares = {"factor": 0.0, "sector": 0.0}

    rebal_set = set(rebal_dates)
    all_days = prices.index[(prices.index.date >= config.start) & (prices.index.date <= end)]

    equity_rows = []
    holdings_rows = []
    trades_rows = []

    for d in all_days:
        # Mark-to-market each leg
        for leg in ("factor", "sector"):
            tk = leg_holding[leg]
            sh = leg_shares[leg]
            if tk is None:
                pass  # cash, leg_nav unchanged
            else:
                p = prices.loc[d, tk] if tk in prices.columns else np.nan
                if np.isnan(p):
                    col = prices[tk].loc[:d].dropna()
                    if not col.empty:
                        p = col.iloc[-1]
                if not np.isnan(p):
                    leg_nav[leg] = sh * float(p)

        if d in rebal_set:
            # Cash overlay signal (same SPY check applies to both legs)
            in_market = (config.overlay_sma_months is None
                         or _overlay_in_market(spy, d, config.overlay_sma_months))

            # Decide each leg's pick
            for leg, basket in (("factor", factor), ("sector", sector)):
                if not in_market:
                    target = None
                else:
                    rets = _trailing_return(prices[basket], d, config.lookback_months)
                    rets = rets.dropna()
                    target = rets.idxmax() if not rets.empty else None

                current = leg_holding[leg]
                if target == current:
                    continue  # no trade

                # Sell current (if any)
                if current is not None:
                    p = prices.loc[d, current]
                    if not np.isnan(p):
                        proceeds = leg_shares[leg] * float(p)
                        leg_nav[leg] = proceeds * (1 - cost)
                        trades_rows.append({"date": d.date(), "leg": leg,
                                            "ticker": current, "side": "sell",
                                            "amount": proceeds})
                    leg_shares[leg] = 0.0
                    leg_holding[leg] = None

                # Buy target (if any)
                if target is not None:
                    p = prices.loc[d, target]
                    if np.isnan(p) or p <= 0:
                        continue
                    spend = leg_nav[leg] * (1 - cost)
                    leg_shares[leg] = spend / float(p)
                    leg_holding[leg] = target
                    trades_rows.append({"date": d.date(), "leg": leg,
                                        "ticker": target, "side": "buy",
                                        "amount": spend})

            # Log holdings
            total_nav = leg_nav["factor"] + leg_nav["sector"]
            for leg in ("factor", "sector"):
                tk = leg_holding[leg] or "CASH"
                weight = leg_nav[leg] / total_nav if total_nav > 0 else np.nan
                holdings_rows.append({
                    "date": d.date(), "leg": leg, "ticker": tk,
                    "weight": weight, "in_market": in_market,
                })

        total_nav = leg_nav["factor"] + leg_nav["sector"]
        equity_rows.append({"date": d.date(), "portfolio_value": total_nav})

    equity = pd.DataFrame(equity_rows).set_index("date")
    if benchmarks:
        for label, series in benchmarks.items():
            bench = series.copy()
            bench.index = pd.to_datetime(bench.index).date
            bench = bench.reindex(equity.index, method="ffill")
            first = bench.dropna().iloc[0] if not bench.dropna().empty else np.nan
            if not np.isnan(first):
                equity[label] = bench / first * config.starting_capital

    metrics = _compute_metrics(equity, "portfolio_value")
    if benchmarks:
        for label in benchmarks:
            if label in equity.columns:
                metrics[f"{label}_metrics"] = _compute_metrics(equity, label)

    # Stash overlay info into the metrics for the report
    metrics["overlay_active"] = config.overlay_sma_months is not None
    metrics["lookback_months"] = config.lookback_months

    # Repackage as a BacktestResult — config field will be a placeholder
    bt_cfg = BacktestConfig(
        top_n=1, rebalance_freq="monthly", weighting="equal",
        drift_buffer=0, trade_mode="replace",
        transaction_cost_bps=config.transaction_cost_bps,
        starting_capital=config.starting_capital,
        start=config.start, end=config.end,
    )
    return BacktestResult(
        config=bt_cfg,
        equity_curve=equity.reset_index(),
        holdings=pd.DataFrame(holdings_rows),
        trades=pd.DataFrame(trades_rows),
        metrics=metrics,
    )
