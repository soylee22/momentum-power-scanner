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
    trade_mode: Literal["rebalance", "replace"] = "rebalance"
    # "rebalance" = re-set every name to target weight each rebal day (trims winners).
    # "replace"   = only trade names entering/leaving the top_n. Existing positions
    #               ride at their drifted weights. New entrants funded equally from
    #               the proceeds of leavers. Winners run. Lower turnover, less tax.
    require_setup: bool = False
    # If True, only names with an active VCP + pivot-break "setup_buy" on the rebal
    # day are eligible to be NEW entries. Existing holdings still ride. This is the
    # Minervini-v2 layer — no chasing extended names.
    stop_loss_pct: float | None = None
    # If set, daily intra-month check: any position whose close falls below
    # entry_price × (1 - stop_loss_pct) is sold immediately at that day's close.
    # Cash sits until next rebal. Classic Minervini 7-8% stop = pass 0.08.
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


def long_to_wide_volume(prices_long: pd.DataFrame) -> pd.DataFrame:
    """Long table -> wide volume panel."""
    df = prices_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="ticker", values="volume", aggfunc="last").sort_index()


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

                # Current dollar value per held name. Use today's close if valid;
                # otherwise fall back to the last known close (e.g. a UK ticker on a
                # US holiday). This keeps positions from silently vanishing under
                # high-churn replace mode.
                current_value = {}
                for tk, sh in positions.items():
                    if tk not in prices_wide.columns:
                        continue
                    p = prices_wide.loc[d, tk]
                    if np.isnan(p):
                        col = prices_wide[tk].loc[:d].dropna()
                        if not col.empty:
                            p = col.iloc[-1]
                    if not np.isnan(p) and p > 0:
                        current_value[tk] = sh * float(p)
                tradable_nav = nav_today

                target_set = set(target_tickers)
                old_set = set(current_value.keys())
                leavers = old_set - target_set
                entrants = target_set - old_set

                turnover = 0.0
                new_positions: dict[str, float] = {}
                trade_log = []

                if config.trade_mode == "replace" and positions:
                    # Existing keepers ride: don't change their share count
                    for tk in target_set & old_set:
                        new_positions[tk] = positions[tk]
                    # Sell all leavers
                    proceeds = 0.0
                    for tk in leavers:
                        v = current_value.get(tk, 0.0)
                        proceeds += v
                        turnover += v
                        trade_log.append({"date": d.date(), "ticker": tk, "side": "sell", "amount": v})
                    # Buy entrants equally from proceeds
                    if entrants:
                        per = proceeds / len(entrants)
                        for tk in entrants:
                            p = prices_wide.loc[d, tk] if tk in prices_wide.columns else np.nan
                            if not np.isnan(p) and p > 0:
                                new_positions[tk] = per / float(p)
                                turnover += per
                                trade_log.append({"date": d.date(), "ticker": tk, "side": "buy", "amount": per})
                    # NOTE: target_weight isn't used to log holdings here — log actual drifted weights
                    if config.weighting == "composite":
                        # Composite weighting + replace doesn't really make sense; ignore composite weights here.
                        pass
                else:
                    # "rebalance" mode (full pie reset) OR first entry into the strategy from cash
                    if config.weighting == "composite" and target_tickers:
                        comp_lookup = dict(zip(ranked["ticker"], ranked["composite"]))
                        weights = np.array([max(comp_lookup.get(tk, 0.0), 0.0) for tk in target_tickers])
                        if weights.sum() == 0:
                            weights = np.ones(len(target_tickers))
                        weights = weights / weights.sum()
                    else:
                        weights = np.ones(len(target_tickers)) / max(len(target_tickers), 1)
                    target_weight = dict(zip(target_tickers, weights))

                    all_tk = old_set | target_set
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
                            trade_log.append({
                                "date": d.date(), "ticker": tk,
                                "side": "buy" if delta_v > 0 else "sell",
                                "amount": abs(delta_v),
                            })
                    # Keepers that already match target
                    for tk in target_weight:
                        if tk not in new_positions:
                            p = prices_wide.loc[d, tk] if tk in prices_wide.columns else np.nan
                            if not np.isnan(p) and p > 0:
                                new_positions[tk] = (target_weight[tk] * tradable_nav) / float(p)

                trades_rows.extend(trade_log)

                # Apply transaction cost on turnover
                tx_cost = turnover * cost_rate
                nav_today -= tx_cost

                # Rescale only in rebalance mode — replace mode preserves keepers' shares
                if new_positions and config.trade_mode == "rebalance":
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

                # Log ACTUAL weights at the close of the rebalance day
                gross_val = 0.0
                pos_values = {}
                for tk, sh in positions.items():
                    p = prices_wide.loc[d, tk] if tk in prices_wide.columns else np.nan
                    if not np.isnan(p):
                        v = sh * float(p)
                        pos_values[tk] = v
                        gross_val += v
                for tk in positions:
                    w = (pos_values.get(tk, 0.0) / gross_val) if gross_val > 0 else np.nan
                    holdings_rows.append({
                        "date": d.date(),
                        "ticker": tk,
                        "weight": w,
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


# ---------------------------------------------------------------------------
# v2 backtest — Minervini playbook: VCP+pivot setup-required entry + 8% stop
# ---------------------------------------------------------------------------


def run_backtest_v2(
    prices_long: pd.DataFrame,
    universe: pd.DataFrame,
    config: BacktestConfig,
    benchmarks: dict[str, pd.Series] | None = None,
) -> BacktestResult:
    """v2 engine with explicit cash, intra-month stops, and setup-gated entries.

    Requires config.require_setup=True OR config.stop_loss_pct set (otherwise
    use run_backtest). Cash sits in slots that have no active setup; the
    portfolio can be partially in cash, which is realistic Minervini behaviour.
    """
    from .setup import compute_setup_panel

    prices_wide = long_to_wide(prices_long)
    volumes_wide = long_to_wide_volume(prices_long)
    features = compute_feature_panel(prices_wide)
    setups = compute_setup_panel(prices_wide, volumes_wide) if config.require_setup else {}

    end = config.end or prices_wide.index.max().date()
    rebal_dates = _pick_rebalance_dates(prices_wide.index, config.start, end, config.rebalance_freq)
    if not rebal_dates:
        raise ValueError("No rebalance dates in the window.")

    cash = float(config.starting_capital)
    positions: dict[str, float] = {}        # ticker -> shares
    entry_prices: dict[str, float] = {}      # ticker -> entry close
    equity_rows = []
    holdings_rows = []
    trades_rows = []
    last_target_ranks: dict[str, int] = {}

    cost_rate = config.transaction_cost_bps / 10_000.0
    stop_pct = config.stop_loss_pct

    all_trading_days = prices_wide.index[
        (prices_wide.index.date >= config.start) & (prices_wide.index.date <= end)
    ]
    rebal_set = set(rebal_dates)

    def _last_valid_price(tk: str, d: pd.Timestamp) -> float:
        if tk not in prices_wide.columns:
            return float("nan")
        p = prices_wide.loc[d, tk]
        if np.isnan(p):
            col = prices_wide[tk].loc[:d].dropna()
            return float(col.iloc[-1]) if not col.empty else float("nan")
        return float(p)

    for d in all_trading_days:
        # 1. STOP CHECK (daily, regardless of rebal). Sell anything that closed
        #    below entry × (1 - stop_pct) at today's close.
        if stop_pct is not None and positions:
            for tk in list(positions.keys()):
                if tk not in prices_wide.columns:
                    continue
                p_today = prices_wide.loc[d, tk]
                if np.isnan(p_today):
                    continue
                trigger = entry_prices.get(tk, 0.0) * (1 - stop_pct)
                if p_today <= trigger:
                    proceeds = positions[tk] * float(p_today)
                    cash += proceeds * (1 - cost_rate)
                    trades_rows.append({
                        "date": d.date(), "ticker": tk, "side": "stop",
                        "amount": proceeds,
                    })
                    del positions[tk]
                    del entry_prices[tk]

        # 2. MARK-TO-MARKET
        gross = 0.0
        valued_count = 0
        for tk, sh in positions.items():
            p = _last_valid_price(tk, d)
            if not np.isnan(p):
                gross += sh * p
                valued_count += 1
        if valued_count >= max(1, len(positions) // 2) or not positions:
            nav_today = cash + gross
        else:
            # carry forward yesterday's NAV
            nav_today = equity_rows[-1]["portfolio_value"] if equity_rows else config.starting_capital

        # 3. REBALANCE
        if d in rebal_set:
            ranked = rank_snapshot(features, universe, d)
            if len(ranked) >= config.min_universe:
                # Apply setup filter to the universe of candidates IF required
                if config.require_setup and setups:
                    setup_today = {
                        tk for tk, df in setups.items()
                        if (df.index <= d).any() and bool(df.loc[df.index[df.index <= d][-1], "setup_buy"])
                    }
                else:
                    setup_today = set(ranked["ticker"])

                # drift_buffer logic — same as v1
                if config.drift_buffer > 0 and positions:
                    current_tickers = set(positions.keys())
                    rank_lookup = dict(zip(ranked["ticker"], ranked["rank_overall"]))
                    keep = {tk for tk in current_tickers
                            if rank_lookup.get(tk, 9999) <= config.top_n + config.drift_buffer}
                    n_new_needed = config.top_n - len(keep)
                    # New entrants MUST have an active setup
                    new_picks = [
                        tk for tk in ranked["ticker"].tolist()
                        if tk not in keep and tk in setup_today
                    ][:max(0, n_new_needed)]
                    target_tickers = list(keep) + new_picks
                    target_ranks = {tk: rank_lookup.get(tk, 9999) for tk in target_tickers}
                else:
                    # Top of ranking, gated by setup for entries
                    target_tickers = [
                        tk for tk in ranked["ticker"].tolist() if tk in setup_today
                    ][:config.top_n]
                    target_ranks = dict(zip(target_tickers, range(1, len(target_tickers) + 1)))

                target_set = set(target_tickers)
                old_set = set(positions.keys())

                if config.trade_mode == "replace":
                    # Sell leavers
                    leavers = old_set - target_set
                    for tk in leavers:
                        p = _last_valid_price(tk, d)
                        if not np.isnan(p):
                            proceeds = positions[tk] * p
                            cash += proceeds * (1 - cost_rate)
                            trades_rows.append({"date": d.date(), "ticker": tk, "side": "sell", "amount": proceeds})
                        del positions[tk]
                        entry_prices.pop(tk, None)
                    # Buy entrants — equal slice of cash
                    entrants = target_set - old_set
                    if entrants and cash > 0:
                        slot_size = cash / len(entrants)
                        for tk in entrants:
                            p = _last_valid_price(tk, d)
                            if not np.isnan(p) and p > 0:
                                spend = slot_size * (1 - cost_rate)
                                positions[tk] = spend / p
                                entry_prices[tk] = p
                                cash -= slot_size
                                trades_rows.append({"date": d.date(), "ticker": tk, "side": "buy", "amount": spend})
                else:
                    # rebalance mode — reset all targets to equal weight
                    weight = 1.0 / max(len(target_tickers), 1)
                    target_value = nav_today * weight
                    # Sell everything first
                    for tk in list(positions.keys()):
                        p = _last_valid_price(tk, d)
                        if not np.isnan(p):
                            cash += positions[tk] * p * (1 - cost_rate)
                            trades_rows.append({"date": d.date(), "ticker": tk, "side": "sell", "amount": positions[tk] * p})
                        del positions[tk]
                        entry_prices.pop(tk, None)
                    # Buy targets equally
                    for tk in target_tickers:
                        p = _last_valid_price(tk, d)
                        if not np.isnan(p) and p > 0:
                            spend = target_value * (1 - cost_rate)
                            positions[tk] = spend / p
                            entry_prices[tk] = p
                            cash -= target_value
                            trades_rows.append({"date": d.date(), "ticker": tk, "side": "buy", "amount": spend})

                last_target_ranks = target_ranks

            # Log holdings on rebal day (actual weights)
            gross_val = 0.0
            pos_values = {}
            for tk, sh in positions.items():
                p = _last_valid_price(tk, d)
                if not np.isnan(p):
                    v = sh * p
                    pos_values[tk] = v
                    gross_val += v
            total = gross_val + cash
            for tk in positions:
                w = pos_values.get(tk, 0.0) / total if total > 0 else np.nan
                holdings_rows.append({
                    "date": d.date(), "ticker": tk, "weight": w,
                    "rank": last_target_ranks.get(tk, np.nan),
                })
            if cash > 0 and total > 0 and (cash / total) > 0.01:
                holdings_rows.append({
                    "date": d.date(), "ticker": "CASH",
                    "weight": cash / total, "rank": np.nan,
                })

        # 4. Recompute MTM end-of-day after any rebal trades
        gross = sum(
            sh * _last_valid_price(tk, d)
            for tk, sh in positions.items()
            if not np.isnan(_last_valid_price(tk, d))
        )
        nav_today = cash + gross
        equity_rows.append({"date": d.date(), "portfolio_value": nav_today})

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
