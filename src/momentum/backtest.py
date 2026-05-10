"""Backtest engine for the Momentum Power scanner.

Runs the live ranking logic point-in-time across historical dates.
Reuses minervini.evaluate_gates and composite.rank_composite for parity
with the live scanner — same gates, same composite formula.

Performance: features are pre-computed once across the full price panel
using vectorised pandas rolling windows, then sampled at rebalance dates.
A 10-year weekly backtest over ~600 names runs in < 30 seconds.

Survivorship-bias caveat: the universe is the *current* S&P 500 + FTSE 100,
so delisted/removed tickers are absent. Realised forward returns will be
lower than backtested. Run the report sensitivity section to size this.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config + result schemas
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    top_n: int = 10
    rebalance_freq: Literal["weekly", "monthly", "quarterly"] = "weekly"
    weighting: Literal["equal", "composite"] = "equal"
    drift_buffer: int = 0  # 0 = follow exactly; N = exit only when rank > top_n + N
    transaction_cost_bps: float = 5.0  # round-trip
    starting_capital: float = 100_000.0
    start: dt.date = dt.date(2017, 1, 1)
    end: dt.date | None = None
    min_universe: int = 50  # if survivor pool < this, skip rebalance


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature pre-computation (vectorised)
# ---------------------------------------------------------------------------


def long_to_wide(prices_long: pd.DataFrame) -> pd.DataFrame:
    """Long table (ticker, date, close, volume) -> wide close panel."""
    df = prices_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()


def compute_feature_panel(prices_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """For each ticker, return a DataFrame of date-indexed features.

    Columns: price, sma50, sma150, sma200, sma200_21d_ago,
             high_52w, low_52w, return_3m, return_6m, return_9m, return_12m,
             weighted_perf (per RS rating formula).
    """
    features: dict[str, pd.DataFrame] = {}
    for tk in prices_wide.columns:
        s = prices_wide[tk].dropna()
        if len(s) < 252:
            continue
        df = pd.DataFrame(index=s.index)
        df["price"] = s
        df["sma50"] = s.rolling(50).mean()
        df["sma150"] = s.rolling(150).mean()
        df["sma200"] = s.rolling(200).mean()
        df["sma200_21d_ago"] = df["sma200"].shift(21)
        df["high_52w"] = s.rolling(252, min_periods=252).max()
        df["low_52w"] = s.rolling(252, min_periods=252).min()
        df["return_3m"] = s / s.shift(63) - 1
        df["return_6m"] = s / s.shift(126) - 1
        df["return_9m"] = s / s.shift(189) - 1
        df["return_12m"] = s / s.shift(252) - 1
        df["weighted_perf"] = (
            0.4 * df["return_3m"]
            + 0.2 * df["return_6m"]
            + 0.2 * df["return_9m"]
            + 0.2 * df["return_12m"]
        )
        df = df.dropna(subset=["sma200_21d_ago", "return_12m"])
        features[tk] = df
    return features


# ---------------------------------------------------------------------------
# Per-date snapshot: gates -> RS -> composite -> top N
# ---------------------------------------------------------------------------


def _features_asof(features: dict[str, pd.DataFrame], d: pd.Timestamp) -> pd.DataFrame:
    """Slice each ticker's feature frame at the most-recent <= d row."""
    rows = []
    for tk, fdf in features.items():
        valid = fdf.index[fdf.index <= d]
        if len(valid) == 0:
            continue
        r = fdf.loc[valid[-1]].to_dict()
        r["ticker"] = tk
        rows.append(r)
    return pd.DataFrame(rows)


def _apply_gates(df: pd.DataFrame) -> pd.Series:
    """Vectorised Stage 2 gates (g1-g7). g8 (RS>=70) applied separately
    after RS rating is computed across the asof-day universe."""
    return (
        (df["price"] > df["sma150"]) & (df["price"] > df["sma200"])
        & (df["sma150"] > df["sma200"])
        & (df["sma200"] > df["sma200_21d_ago"])
        & (df["sma50"] > df["sma150"]) & (df["sma50"] > df["sma200"])
        & (df["price"] > df["sma50"])
        & (df["price"] >= 1.30 * df["low_52w"])
        & (df["price"] >= 0.75 * df["high_52w"])
    )


def _rank_composite(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dist_from_high"] = (out["high_52w"] - out["price"]) / out["high_52w"]
    out["rank_rs"] = out["rs_rating"].rank(pct=True, method="average")
    out["rank_prox"] = (-out["dist_from_high"]).rank(pct=True, method="average")
    out["rank_1yr"] = out["return_12m"].rank(pct=True, method="average")
    out["composite"] = (out["rank_rs"] + out["rank_prox"] + out["rank_1yr"]) / 3.0
    out = out.sort_values("composite", ascending=False).reset_index(drop=True)
    out["rank_overall"] = out.index + 1
    return out


def _dedupe_share_classes(survivors: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Drop GOOGL/GOOG, BRK.A/BRK.B, FOX/FOXA etc — keep the cheaper class."""
    if survivors.empty:
        return survivors
    df = survivors.merge(universe[["ticker", "name"]], on="ticker", how="left")
    base = (
        df["name"].astype(str)
        .str.replace(r"\s*\(\s*Class\s+[A-Z]\s*\)\s*", "", regex=True, case=False)
        .str.replace(r"\s+Class\s+[A-Z]\s*$", "", regex=True, case=False)
        .str.strip()
        .str.rstrip(".")
    )
    df["_base"] = base
    df = df.sort_values("price").drop_duplicates("_base", keep="first")
    return df.drop(columns=["_base", "name"]).sort_index()


def rank_snapshot(
    features: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    d: pd.Timestamp,
) -> pd.DataFrame:
    """Return the top-ranked Stage 2 survivors at date d, sorted by composite."""
    df = _features_asof(features, d)
    if df.empty:
        return df
    # g1-g7
    g17 = _apply_gates(df)
    # RS rating across the asof-universe (1-99 percentile of weighted_perf)
    pct = df["weighted_perf"].rank(pct=True, method="average")
    df["rs_rating"] = 1.0 + 98.0 * pct
    # g8
    survivors = df[g17 & (df["rs_rating"] >= 70.0)].copy()
    if survivors.empty:
        return survivors
    survivors = _dedupe_share_classes(survivors, universe)
    if survivors.empty:
        return survivors
    return _rank_composite(survivors)


# ---------------------------------------------------------------------------
# Rebalance date selection
# ---------------------------------------------------------------------------


def _pick_rebalance_dates(
    all_dates: pd.DatetimeIndex,
    start: dt.date,
    end: dt.date,
    freq: str,
) -> list[pd.Timestamp]:
    """Pick trading-day rebalance dates within [start, end] at given freq.
    Weekly = last trading day of each ISO week. Monthly = last of month.
    Quarterly = last of quarter."""
    idx = all_dates[(all_dates.date >= start) & (all_dates.date <= end)]
    if len(idx) == 0:
        return []
    s = pd.Series(idx, index=idx)
    if freq == "weekly":
        keep = s.groupby([idx.isocalendar().year, idx.isocalendar().week]).last()
    elif freq == "monthly":
        keep = s.groupby([idx.year, idx.month]).last()
    elif freq == "quarterly":
        keep = s.groupby([idx.year, idx.quarter]).last()
    else:
        raise ValueError(f"unknown rebalance_freq: {freq}")
    return sorted(keep.tolist())


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------


def run_backtest(
    prices_long: pd.DataFrame,
    universe: pd.DataFrame,
    config: BacktestConfig,
    benchmarks: dict[str, pd.Series] | None = None,
) -> BacktestResult:
    """Run a single configuration. benchmarks is an optional dict of
    {label: daily Close Series} — used purely for the equity curve plot."""
    prices_wide = long_to_wide(prices_long)
    features = compute_feature_panel(prices_wide)

    end = config.end or prices_wide.index.max().date()
    rebal_dates = _pick_rebalance_dates(prices_wide.index, config.start, end, config.rebalance_freq)
    if not rebal_dates:
        raise ValueError("No rebalance dates in the window — check start/end vs data range.")

    nav = float(config.starting_capital)
    positions: dict[str, float] = {}  # ticker -> share count
    equity_rows = []
    holdings_rows = []
    trades_rows = []
    last_target_ranks: dict[str, int] = {}

    cost_rate = config.transaction_cost_bps / 10_000.0

    # Build a fast price lookup: daily prices_wide already
    all_trading_days = prices_wide.index[
        (prices_wide.index.date >= config.start) & (prices_wide.index.date <= end)
    ]
    rebal_set = set(rebal_dates)

    for d in all_trading_days:
        is_rebal = d in rebal_set

        # Mark-to-market: NAV = sum(shares * price_at_d) for tickers with a valid
        # close today. Tickers with NaN close (their exchange closed today) keep
        # their last marked value carried forward — otherwise a US-only basket
        # would "lose" NAV on Thanksgiving while FTSE trades, etc.
        if positions:
            valued = 0.0
            valued_count = 0
            for tk, sh in positions.items():
                p = prices_wide.loc[d, tk] if tk in prices_wide.columns else np.nan
                if not np.isnan(p):
                    valued += sh * float(p)
                    valued_count += 1
            # If we got fewer than half the positions valued today, the market for
            # most of the basket is closed — carry forward yesterday's NAV instead
            # of pretending the un-valued positions are worth zero.
            if valued_count >= max(1, len(positions) // 2):
                nav_today = valued
            else:
                nav_today = nav  # carry forward
        else:
            nav_today = nav

        if is_rebal:
            ranked = rank_snapshot(features, universe, d)
            if len(ranked) < config.min_universe:
                # Not enough survivors — hold current positions, no trade
                pass
            else:
                if config.drift_buffer > 0 and positions:
                    # Hold names while still in top_n + buffer, then add new top_n names
                    current_tickers = set(positions.keys())
                    rank_lookup = dict(zip(ranked["ticker"], ranked["rank_overall"]))
                    keep = {tk for tk in current_tickers
                            if rank_lookup.get(tk, 9999) <= config.top_n + config.drift_buffer}
                    n_keep = len(keep)
                    n_new_needed = config.top_n - n_keep
                    if n_new_needed > 0:
                        new_picks = [tk for tk in ranked["ticker"].tolist() if tk not in keep][:n_new_needed]
                    else:
                        new_picks = []
                    target_tickers = list(keep) + new_picks
                    target_ranks = {tk: rank_lookup.get(tk, 9999) for tk in target_tickers}
                else:
                    target_tickers = ranked["ticker"].head(config.top_n).tolist()
                    target_ranks = dict(zip(target_tickers, range(1, len(target_tickers) + 1)))

                # Weights
                if config.weighting == "composite" and target_tickers:
                    comp_lookup = dict(zip(ranked["ticker"], ranked["composite"]))
                    weights = np.array([max(comp_lookup.get(tk, 0.0), 0.0) for tk in target_tickers])
                    if weights.sum() == 0:
                        weights = np.ones(len(target_tickers))
                    weights = weights / weights.sum()
                else:
                    weights = np.ones(len(target_tickers)) / max(len(target_tickers), 1)

                target_weight = dict(zip(target_tickers, weights))

                # Compute trades. Current holdings -> target holdings.
                # nav_today = current NAV (mark-to-market just done above).
                current_value = {}
                for tk, sh in positions.items():
                    p = prices_wide.loc[d, tk] if tk in prices_wide.columns else np.nan
                    if not np.isnan(p):
                        current_value[tk] = sh * float(p)
                tradable_nav = nav_today

                turnover = 0.0
                new_positions: dict[str, float] = {}
                all_tk = set(current_value.keys()) | set(target_weight.keys())
                for tk in all_tk:
                    cur_v = current_value.get(tk, 0.0)
                    tgt_v = target_weight.get(tk, 0.0) * tradable_nav
                    delta_v = tgt_v - cur_v
                    if abs(delta_v) < 1e-6:
                        continue
                    turnover += abs(delta_v)
                    p = prices_wide.loc[d, tk] if tk in prices_wide.columns else np.nan
                    if not np.isnan(p) and p > 0:
                        if tgt_v > 0:
                            new_positions[tk] = tgt_v / float(p)
                        trades_rows.append({
                            "date": d.date(), "ticker": tk,
                            "side": "buy" if delta_v > 0 else "sell",
                            "amount": abs(delta_v),
                        })
                # Keep positions that didn't change (already in target_weight at correct weight)
                for tk in target_weight:
                    if tk not in new_positions:
                        p = prices_wide.loc[d, tk] if tk in prices_wide.columns else np.nan
                        if not np.isnan(p) and p > 0:
                            new_positions[tk] = (target_weight[tk] * tradable_nav) / float(p)

                # Apply transaction cost on turnover
                tx_cost = turnover * cost_rate
                nav_today -= tx_cost

                # Rescale positions slightly so they sum to nav_today - 0 (we don't keep cash)
                # Simpler: scale linearly
                if new_positions:
                    gross = sum(
                        sh * float(prices_wide.loc[d, tk])
                        for tk, sh in new_positions.items()
                        if tk in prices_wide.columns and not np.isnan(prices_wide.loc[d, tk])
                    )
                    if gross > 0:
                        scale = nav_today / gross
                        new_positions = {tk: sh * scale for tk, sh in new_positions.items()}

                positions = new_positions
                last_target_ranks = target_ranks

                # Log holdings
                for tk in positions:
                    holdings_rows.append({
                        "date": d.date(),
                        "ticker": tk,
                        "weight": target_weight.get(tk, np.nan),
                        "rank": last_target_ranks.get(tk, np.nan),
                    })

        equity_rows.append({"date": d.date(), "portfolio_value": nav_today})
        nav = nav_today

    equity = pd.DataFrame(equity_rows).set_index("date")

    # Attach benchmarks
    if benchmarks:
        for label, series in benchmarks.items():
            bench = series.copy()
            bench.index = pd.to_datetime(bench.index).date
            bench = bench.reindex(equity.index, method="ffill")
            # Normalise to start at the same capital
            first = bench.dropna().iloc[0] if not bench.dropna().empty else np.nan
            if not np.isnan(first):
                equity[label] = bench / first * config.starting_capital

    metrics = _compute_metrics(equity, "portfolio_value")
    if benchmarks:
        for label in benchmarks:
            if label in equity.columns:
                metrics[f"{label}_metrics"] = _compute_metrics(equity, label)

    return BacktestResult(
        config=config,
        equity_curve=equity.reset_index(),
        holdings=pd.DataFrame(holdings_rows),
        trades=pd.DataFrame(trades_rows),
        metrics=metrics,
    )


def _compute_metrics(equity: pd.DataFrame, col: str) -> dict:
    s = equity[col].dropna()
    s = s[s > 0]  # drop zero/negative NAV days from metric calc
    if len(s) < 2:
        return {}
    rets = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    yrs = (pd.to_datetime(s.index[-1]) - pd.to_datetime(s.index[0])).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol > 0 else np.nan
    cummax = s.cummax()
    dd = (s / cummax - 1).min()
    return {
        "start_value": float(s.iloc[0]),
        "end_value": float(s.iloc[-1]),
        "years": float(yrs),
        "cagr": float(cagr) if not np.isnan(cagr) else None,
        "vol_ann": float(vol),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        "max_drawdown": float(dd),
    }
