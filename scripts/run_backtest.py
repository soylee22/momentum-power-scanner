"""Parameter sweep + HTML report for the momentum backtest.

Outputs:
  - data/backtest/results.parquet  (one row per config × date with NAV)
  - data/backtest/holdings_<config>.parquet
  - data/backtest/trades_<config>.parquet
  - docs/backtest.html             (Palantir-aesthetic report linked from main dashboard)

Run with .venv/bin/python scripts/run_backtest.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from momentum.backtest import BacktestConfig, run_backtest  # noqa: E402
from momentum.universe import load_universe  # noqa: E402
from momentum.palantir import PL  # noqa: E402

PRICES_PATH = ROOT / "data" / "prices_backtest.parquet"
OUT_DIR = ROOT / "data" / "backtest"
DOCS = ROOT / "docs"
BACKTEST_HTML = DOCS / "backtest.html"

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def fetch_benchmark(ticker: str, start: dt.date) -> pd.Series:
    raw = yf.download(ticker, start=start.isoformat(), auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).date
    return s


def load_prices_long() -> pd.DataFrame:
    df = pd.read_parquet(PRICES_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def sweep_configs(start: dt.date) -> list[tuple[str, BacktestConfig]]:
    """Curated sweep — not Cartesian (would be 60 configs). Picks the
    decisions that matter most: top_n, rebalance freq, drift, weighting."""
    configs = []
    for top_n in (5, 10, 20):
        for freq in ("weekly", "monthly", "quarterly"):
            configs.append((
                f"top{top_n}_{freq}_eq",
                BacktestConfig(top_n=top_n, rebalance_freq=freq, weighting="equal",
                               drift_buffer=0, start=start),
            ))
    # Drift-held variants for the headline shape
    configs.append((
        "top10_weekly_drift5",
        BacktestConfig(top_n=10, rebalance_freq="weekly", weighting="equal",
                       drift_buffer=5, start=start),
    ))
    configs.append((
        "top10_monthly_drift5",
        BacktestConfig(top_n=10, rebalance_freq="monthly", weighting="equal",
                       drift_buffer=5, start=start),
    ))
    # Composite-weighted variant
    configs.append((
        "top10_monthly_compW",
        BacktestConfig(top_n=10, rebalance_freq="monthly", weighting="composite",
                       drift_buffer=0, start=start),
    ))
    # Concentrated single-name
    configs.append((
        "top1_monthly",
        BacktestConfig(top_n=1, rebalance_freq="monthly", weighting="equal", start=start),
    ))
    # Top-2 baseline
    configs.append((
        "top2_monthly",
        BacktestConfig(top_n=2, rebalance_freq="monthly", weighting="equal", start=start),
    ))
    return configs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _equity_chart_svg(curves: dict[str, pd.Series], width: int = 1000, height: int = 360) -> str:
    """Multi-line equity curve as inline SVG with shared y-axis."""
    if not curves:
        return ""
    pad_l, pad_r, pad_t, pad_b = 70, 24, 28, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # Common date axis = union of indices
    all_dates = sorted(set().union(*[set(s.index) for s in curves.values()]))
    if not all_dates:
        return ""
    date_to_x = {d: pad_l + i * (plot_w / (len(all_dates) - 1)) for i, d in enumerate(all_dates)}

    # Common y-range
    vmin = min(s.min() for s in curves.values())
    vmax = max(s.max() for s in curves.values())
    span = (vmax - vmin) or 1

    palette = ["#C8923D", "#6E8FC6", "#7CA678", "#A55E5E", "#5E5E5E"]
    paths = []
    legend = []
    for i, (label, s) in enumerate(curves.items()):
        color = palette[i % len(palette)]
        pts = []
        for d, v in s.items():
            if np.isnan(v):
                continue
            x = date_to_x.get(d)
            if x is None:
                continue
            y = pad_t + plot_h - ((v - vmin) / span) * plot_h
            pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            paths.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.6" '
                f'points="{" ".join(pts)}"/>'
            )
            final_val = s.dropna().iloc[-1]
            mult = final_val / s.dropna().iloc[0]
            legend.append((color, label, f"&times;{mult:.2f}"))

    # Y grid
    y_grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = vmin + span * frac
        y = pad_t + plot_h - frac * plot_h
        y_grid.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="currentColor" stroke-opacity="0.08"/>'
        )
        y_grid.append(
            f'<text x="{pad_l - 10}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="currentColor" opacity="0.55">'
            f'£{val/1000:.0f}k</text>'
        )

    # X labels
    n_dates = len(all_dates)
    step = max(1, n_dates // 8)
    x_labels = []
    for i, d in enumerate(all_dates):
        if i % step == 0 or i == n_dates - 1:
            x = date_to_x[d]
            x_labels.append(
                f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" '
                f'font-size="10" fill="currentColor" opacity="0.55">{d}</text>'
            )

    legend_html = " &nbsp; ".join(
        f'<span style="color:{c}">&#9632;</span> {l} <span class="muted">{m}</span>'
        for c, l, m in legend
    )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet" style="max-width:100%;height:auto;">'
        f'{"".join(y_grid)}{"".join(paths)}{"".join(x_labels)}'
        f'</svg>'
        f'<div class="legend">{legend_html}</div>'
    )


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:+.1f}%"


def _fmt_mult(x):
    if x is None:
        return "—"
    return f"{x:.2f}"


def write_report(results: dict[str, "BacktestResult"], benchmarks: dict[str, pd.Series]) -> Path:
    headline_key = "top10_monthly_eq"
    headline = results.get(headline_key) or list(results.values())[0]

    # Equity curves: headline + benchmarks + 2-3 alt configs
    alt_keys = ["top5_monthly_eq", "top20_monthly_eq", "top10_weekly_eq", "top10_monthly_drift5", "top1_monthly"]
    curves: dict[str, pd.Series] = {}
    eq = headline.equity_curve.set_index("date")["portfolio_value"]
    curves[headline_key] = eq
    for k in alt_keys:
        if k in results:
            curves[k] = results[k].equity_curve.set_index("date")["portfolio_value"]
    # Benchmark from headline (already aligned to its date range inside the engine)
    head_df = headline.equity_curve.set_index("date")
    for bench_label in benchmarks:
        if bench_label in head_df.columns:
            curves[bench_label] = head_df[bench_label]

    chart_svg = _equity_chart_svg(curves)

    # Summary table
    rows = []
    for key, r in results.items():
        m = r.metrics
        rows.append({
            "config": key,
            "top_n": r.config.top_n,
            "rebal": r.config.rebalance_freq,
            "weighting": r.config.weighting,
            "drift": r.config.drift_buffer,
            "years": m.get("years"),
            "cagr": m.get("cagr"),
            "vol": m.get("vol_ann"),
            "sharpe": m.get("sharpe"),
            "max_dd": m.get("max_drawdown"),
            "end_value": m.get("end_value"),
            "trades": len(r.trades),
        })
    # Sort by Sharpe desc
    rows.sort(key=lambda x: (x["sharpe"] or -999), reverse=True)
    summary_rows_html = "".join(
        f"""
        <tr>
          <td><code>{r['config']}</code></td>
          <td class="num">{r['top_n']}</td>
          <td>{r['rebal']}</td>
          <td>{r['weighting']}</td>
          <td class="num">{r['drift']}</td>
          <td class="num">{_fmt_pct(r['cagr'])}</td>
          <td class="num">{_fmt_pct(r['vol'])}</td>
          <td class="num"><strong>{_fmt_mult(r['sharpe'])}</strong></td>
          <td class="num neg">{_fmt_pct(r['max_dd'])}</td>
          <td class="num">£{(r['end_value'] or 0)/1000:.1f}k</td>
          <td class="num muted">{r['trades']}</td>
        </tr>
        """
        for r in rows
    )

    # Holdings drill-down: show every rebalance date for the headline config
    head_holdings = headline.holdings.copy()
    if not head_holdings.empty:
        head_holdings["date"] = pd.to_datetime(head_holdings["date"]).dt.date
        universe = load_universe()
        head_holdings = head_holdings.merge(
            universe[["ticker", "name", "sector", "index"]], on="ticker", how="left"
        )
        head_holdings = head_holdings.sort_values(["date", "rank"], ascending=[False, True])
        # Group by date
        holdings_html_parts = []
        for d, grp in head_holdings.groupby("date", sort=False):
            tickers = grp.assign(
                flag=lambda x: np.where(x["index"] == "S&P 500", "\U0001F1FA\U0001F1F8", "\U0001F1EC\U0001F1E7"),
            )
            rows = "".join(
                f"<tr><td class='num'>{int(r['rank'])}</td><td><span class='tk'>{r['ticker']}</span></td>"
                f"<td>{r['flag']} {r.get('name') or ''}</td><td class='muted'>{r.get('sector') or ''}</td>"
                f"<td class='num'>{(r['weight'] or 0)*100:.1f}%</td></tr>"
                for _, r in tickers.iterrows()
            )
            holdings_html_parts.append(f"""
            <details>
              <summary>{d} &mdash; <span class="muted">{len(grp)} positions</span></summary>
              <table class="holdings-table">
                <thead><tr><th>#</th><th>Ticker</th><th>Name</th><th>Sector</th><th class="right">Weight</th></tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </details>
            """)
        holdings_html = "\n".join(holdings_html_parts[:200])  # cap for page weight
        holdings_note = (
            f'Showing all {len(holdings_html_parts)} rebalance dates for <code>{headline_key}</code>. '
            f'Click a date to expand its positions.'
        )
    else:
        holdings_html = "<p class='muted'>No holdings recorded.</p>"
        holdings_note = ""

    # Period table
    config_summary_html = f"""
    <p>
      Universe: current S&amp;P 500 + FTSE 100 (~600 names, point-in-time gates, retroactive RS).<br/>
      Window: <strong>{headline.equity_curve['date'].min()} &mdash; {headline.equity_curve['date'].max()}</strong>,
      starting capital £{headline.config.starting_capital/1000:.0f}k, transaction cost {headline.config.transaction_cost_bps} bps round-trip.<br/>
      <em>Headline survivorship-bias caveat:</em> only today's index constituents are tested.
      Names that were removed / delisted are absent. Realised forward returns will be
      <strong>materially lower</strong> than what's shown here.
    </p>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Momentum Power · Backtest</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --ink: {PL.ink};
      --stage: {PL.stage};
      --bone: {PL.bone};
      --mute: {PL.mute};
      --graphite: {PL.graphite};
      --hairline-lt: {PL.hairline_lt};
      --green: {PL.accent_green};
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--stage); color: var(--graphite); font-family: "Inter", system-ui, sans-serif; font-size: 15px; line-height: 1.5; -webkit-font-smoothing: antialiased; }}
    .page {{ max-width: 1080px; margin: 0 auto; padding: 56px 40px 80px; }}
    .chip {{ display: inline-block; padding: 8px 16px; border: 1px solid var(--graphite); border-radius: 999px; font-size: 11px; font-weight: 500; letter-spacing: 0.08em; }}
    h1 {{ font-size: 48px; font-weight: 400; letter-spacing: -0.018em; line-height: 1.05; margin: 24px 0 16px; }}
    h2 {{ font-size: 22px; font-weight: 500; letter-spacing: -0.005em; margin: 40px 0 16px; }}
    h3 {{ font-size: 14px; font-weight: 600; margin: 0 0 8px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink); }}
    .lede {{ max-width: 720px; color: #4A4A47; font-size: 16px; line-height: 1.6; }}
    .card {{ background: var(--bone); color: var(--ink); border: 1px solid var(--hairline-lt); border-radius: 24px; padding: 28px 28px; margin-top: 20px; }}
    .card.dark {{ background: var(--ink); color: var(--bone); border: 0; }}
    .card.dark .muted {{ color: rgba(245,243,238,0.55); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--hairline-lt); font-variant-numeric: tabular-nums; }}
    thead th {{ text-align: left; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--mute); font-weight: 500; }}
    td.num, th.right {{ text-align: right; font-size: 13px; }}
    td.neg {{ color: #B23D3D; }}
    .muted {{ color: var(--mute); font-size: 12px; }}
    code {{ font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: 12px; background: rgba(0,0,0,0.04); padding: 1px 6px; border-radius: 3px; }}
    .tk {{ font-weight: 600; font-size: 13px; }}
    .legend {{ margin-top: 12px; font-size: 12px; color: var(--graphite); letter-spacing: 0.02em; line-height: 1.8; }}
    .legend .muted {{ color: var(--mute); font-size: 11px; }}
    details {{ margin: 6px 0; border-bottom: 1px solid var(--hairline-lt); padding-bottom: 6px; }}
    details summary {{ cursor: pointer; font-size: 13px; padding: 8px 0; }}
    details[open] summary {{ font-weight: 600; }}
    .holdings-table {{ margin: 8px 0 16px; font-size: 12px; }}
    .holdings-table th, .holdings-table td {{ padding: 6px 8px; }}
    .foot {{ margin-top: 56px; font-size: 11px; color: var(--mute); letter-spacing: 0.04em; }}
    .foot a {{ color: var(--graphite); }}
  </style>
</head>
<body>
  <div class="page">
    <span class="chip">MOMENTUM POWER&nbsp;&nbsp;|&nbsp;&nbsp;BACKTEST</span>
    <h1>Did this strategy actually work?</h1>
    <p class="lede">
      Eleven years of point-in-time scans over the current S&amp;P 500 + FTSE 100 universe.
      Same Stage 2 gates, same composite formula, same dedupe rules as the live scanner.
      Multiple configurations compared head-to-head against SPY and a buy-and-hold of the
      MSCI ACWI tracker (VWRP).
    </p>

    {config_summary_html}

    <h2>Equity curves</h2>
    <div class="card dark">
      {chart_svg}
    </div>

    <h2>Parameter sweep</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>CONFIG</th>
            <th class="right">N</th>
            <th>REBAL</th>
            <th>WEIGHT</th>
            <th class="right">DRIFT</th>
            <th class="right">CAGR</th>
            <th class="right">VOL</th>
            <th class="right">SHARPE</th>
            <th class="right">MAX DD</th>
            <th class="right">END £100k</th>
            <th class="right">TRADES</th>
          </tr>
        </thead>
        <tbody>{summary_rows_html}</tbody>
      </table>
      <p class="muted" style="margin-top:16px;">
        Sorted by Sharpe descending. CAGR is annualised, vol is annualised daily-return stdev,
        max DD is peak-to-trough on portfolio value. Trades column is round-trip count.
      </p>
    </div>

    <h2>Holdings drill-down · {headline_key}</h2>
    <div class="card">
      <p class="muted">{holdings_note}</p>
      {holdings_html}
    </div>

    <div class="foot">
      <p>
        <strong>Read these caveats before deploying real capital.</strong><br/>
        &middot; Survivorship bias: universe is the current index membership, not point-in-time. Removed names absent.<br/>
        &middot; Past results overstated. The strategy will not perform as well live.<br/>
        &middot; Transaction costs modelled at {headline.config.transaction_cost_bps} bps round-trip. UK CGT not modelled.<br/>
        &middot; No slippage / liquidity modelling. Backtests assume execution at the closing print.<br/>
      </p>
      <p>
        <a href="index.html">&larr; Back to live dashboard</a>
        &nbsp;|&nbsp;
        <a href="https://github.com/soylee22/momentum-power-scanner">Source repo</a>
      </p>
    </div>
  </div>
</body>
</html>
"""
    BACKTEST_HTML.write_text(html)
    return BACKTEST_HTML


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading prices...")
    prices = load_prices_long()
    universe = load_universe()

    start = dt.date(2017, 1, 1)
    print(f"Fetching benchmarks (SPY, VWRP.L) from {start}...")
    spy = fetch_benchmark("SPY", start)
    vwrp = fetch_benchmark("VWRP.L", start)
    benchmarks = {}
    if not spy.empty:
        benchmarks["SPY"] = spy
    if not vwrp.empty:
        benchmarks["VWRP"] = vwrp
    print(f"  benchmarks loaded: {list(benchmarks.keys())}")

    configs = sweep_configs(start)
    print(f"Running {len(configs)} configs...")

    results: dict[str, "BacktestResult"] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs:
        print(f"  > {name}", flush=True)
        try:
            r = run_backtest(prices, universe, cfg, benchmarks=benchmarks)
        except Exception as exc:
            print(f"    FAILED: {exc!r}")
            continue
        results[name] = r
        r.equity_curve.assign(config=name).to_parquet(OUT_DIR / f"equity_{name}.parquet", index=False)
        r.holdings.assign(config=name).to_parquet(OUT_DIR / f"holdings_{name}.parquet", index=False)
        r.trades.assign(config=name).to_parquet(OUT_DIR / f"trades_{name}.parquet", index=False)
        m = r.metrics
        print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  "
              f"DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k")

    print("Writing report...")
    out = write_report(results, benchmarks)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
