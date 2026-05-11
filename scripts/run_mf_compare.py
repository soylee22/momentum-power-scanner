"""Compare v1 momentum strategy, v1 + cash overlay, MarketFighter, and
MarketFighter + cash overlay over a common window. Writes docs/marketfighter.html.
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from momentum.backtest import BacktestConfig, run_backtest  # noqa: E402
from momentum.marketfighter import (  # noqa: E402
    MFConfig, run_marketfighter, FACTOR_BASKET, SECTOR_BASKET, SECTOR_BASKET_EXPANDED,
)
from momentum.universe import load_universe  # noqa: E402
from momentum.palantir import PL  # noqa: E402

PRICES_V1 = ROOT / "data" / "prices_backtest.parquet"
PRICES_MF = ROOT / "data" / "prices_mf.parquet"
OUT_DIR = ROOT / "data" / "mf_compare"
HTML = ROOT / "docs" / "marketfighter.html"

# Common start so all configs are apples-to-apples
START = dt.date(2019, 4, 1)  # gated by WSML.L inception (Mar 2018) + 12mo lookback

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def _load_long(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _spy_series(prices_long: pd.DataFrame) -> pd.Series:
    s = prices_long[prices_long["ticker"] == "SPY"].set_index("date")["close"]
    s.index = pd.to_datetime(s.index)
    return s


def fetch_benchmark(ticker: str) -> pd.Series:
    raw = yf.download(ticker, start=START.isoformat(), auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).date
    return s


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:+.1f}%"


def _fmt_mult(x):
    if x is None:
        return "—"
    return f"{x:.2f}"


def _equity_chart_svg(curves, width=1000, height=360):
    if not curves:
        return ""
    pad_l, pad_r, pad_t, pad_b = 70, 24, 28, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_dates = sorted(set().union(*[set(s.index) for s in curves.values()]))
    if not all_dates:
        return ""
    date_to_x = {d: pad_l + i * (plot_w / (len(all_dates) - 1)) for i, d in enumerate(all_dates)}
    vmin = min(s.min() for s in curves.values())
    vmax = max(s.max() for s in curves.values())
    span = (vmax - vmin) or 1
    palette = ["#C8923D", "#6E8FC6", "#7CA678", "#A55E5E", "#5E5E5E", "#B07AB0"]
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
            paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{" ".join(pts)}"/>')
            final = s.dropna().iloc[-1]
            mult = final / s.dropna().iloc[0]
            legend.append((color, label, f"&times;{mult:.2f}"))
    y_grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = vmin + span * frac
        y = pad_t + plot_h - frac * plot_h
        y_grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="currentColor" stroke-opacity="0.08"/>')
        y_grid.append(f'<text x="{pad_l - 10}" y="{y + 3:.1f}" text-anchor="end" font-size="10" fill="currentColor" opacity="0.55">£{val/1000:.0f}k</text>')
    n = len(all_dates)
    step = max(1, n // 8)
    x_labels = []
    for i, d in enumerate(all_dates):
        if i % step == 0 or i == n - 1:
            x_labels.append(f'<text x="{date_to_x[d]:.1f}" y="{height - 18}" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.55">{d}</text>')
    legend_html = " &nbsp; ".join(f'<span style="color:{c}">&#9632;</span> {l} <span class="muted">{m}</span>' for c, l, m in legend)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" style="max-width:100%;height:auto;">'
        f'{"".join(y_grid)}{"".join(paths)}{"".join(x_labels)}'
        f'</svg><div class="legend">{legend_html}</div>'
    )


def write_report(results: dict, benchmarks: dict) -> Path:
    plot_keys = list(results.keys())
    curves = {}
    for k in plot_keys:
        s = results[k].equity_curve.set_index("date")["portfolio_value"]
        curves[k] = s
    # Bench from any result with attached benchmarks
    any_eq = next(iter(results.values())).equity_curve.set_index("date")
    for bench in benchmarks:
        if bench in any_eq.columns:
            curves[bench] = any_eq[bench]

    chart = _equity_chart_svg(curves)

    rows = []
    for key, r in results.items():
        m = r.metrics
        overlay = r.config.overlay_sma_months
        rows.append({
            "config": key,
            "overlay": f"{overlay}mo SMA" if overlay else "—",
            "cagr": m.get("cagr"),
            "sharpe": m.get("sharpe"),
            "max_dd": m.get("max_drawdown"),
            "end": m.get("end_value"),
            "trades": len(r.trades),
        })
    rows.sort(key=lambda x: (x["sharpe"] or -999), reverse=True)
    rows_html = "".join(
        f"""<tr>
          <td><code>{r['config']}</code></td>
          <td>{r['overlay']}</td>
          <td class="num">{_fmt_pct(r['cagr'])}</td>
          <td class="num"><strong>{_fmt_mult(r['sharpe'])}</strong></td>
          <td class="num neg">{_fmt_pct(r['max_dd'])}</td>
          <td class="num">£{(r['end'] or 0)/1000:.1f}k</td>
          <td class="num muted">{r['trades']}</td>
        </tr>""" for r in rows
    )

    # Latest MarketFighter pick (for reference)
    mf_key = next((k for k in results if k.startswith("mf_")), None)
    mf_block = ""
    if mf_key:
        mfh = results[mf_key].holdings
        if not mfh.empty:
            last = mfh[mfh["date"] == mfh["date"].max()]
            picks = " · ".join(f"{r['leg']}: <strong>{r['ticker']}</strong>" for _, r in last.iterrows())
            mf_block = f'<p class="lede">Latest MarketFighter pick (rebalance {mfh["date"].max()}): {picks}</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MarketFighter clone vs v1</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --ink: {PL.ink}; --stage: {PL.stage}; --bone: {PL.bone}; --mute: {PL.mute};
      --graphite: {PL.graphite}; --hairline-lt: {PL.hairline_lt}; --green: {PL.accent_green};
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--stage); color: var(--graphite); font-family: "Inter", system-ui, sans-serif; font-size: 15px; line-height: 1.5; }}
    .page {{ max-width: 1080px; margin: 0 auto; padding: 56px 40px 80px; }}
    .chip {{ display: inline-block; padding: 8px 16px; border: 1px solid var(--graphite); border-radius: 999px; font-size: 11px; font-weight: 500; letter-spacing: 0.08em; }}
    h1 {{ font-size: 48px; font-weight: 400; letter-spacing: -0.018em; margin: 24px 0 16px; line-height: 1.05; }}
    h2 {{ font-size: 22px; font-weight: 500; margin: 40px 0 16px; }}
    h3 {{ font-size: 14px; font-weight: 600; margin: 0 0 8px; letter-spacing: 0.04em; text-transform: uppercase; }}
    .lede {{ max-width: 720px; color: #4A4A47; font-size: 16px; line-height: 1.6; }}
    .card {{ background: var(--bone); border: 1px solid var(--hairline-lt); border-radius: 24px; padding: 28px; margin-top: 20px; color: var(--ink); }}
    .card.dark {{ background: var(--ink); color: var(--bone); border: 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--hairline-lt); font-variant-numeric: tabular-nums; }}
    thead th {{ text-align: left; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--mute); font-weight: 500; }}
    td.num {{ text-align: right; font-size: 13px; }}
    td.neg {{ color: #B23D3D; }}
    .muted {{ color: var(--mute); font-size: 12px; }}
    code {{ font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: 12px; background: rgba(0,0,0,0.04); padding: 1px 6px; border-radius: 3px; }}
    .legend {{ margin-top: 12px; font-size: 12px; line-height: 1.8; }}
    .pillars {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 20px 0; }}
    .pillar {{ padding: 18px; background: var(--bone); border-radius: 14px; border: 1px solid var(--hairline-lt); }}
    .pillar p {{ margin: 0; font-size: 13px; line-height: 1.55; color: var(--graphite); }}
    .foot {{ margin-top: 56px; font-size: 11px; color: var(--mute); }}
    .foot a {{ color: var(--graphite); }}
  </style>
</head>
<body>
  <div class="page">
    <span class="chip">MOMENTUM POWER&nbsp;&nbsp;|&nbsp;&nbsp;MARKETFIGHTER COMPARISON</span>
    <h1>Two strategies, four configs, one window.</h1>
    <p class="lede">
      Side-by-side test of the existing top-10 single-name momentum strategy
      vs a faithful UK-investable clone of the MarketFighter substack strategy
      (8 factor ETFs + 7 sector ETFs, monthly rotation, 50/50 split). Each
      tested with and without a 10-month SMA cash overlay on SPY. Common
      window {START} - present.
    </p>

    {mf_block}

    <div class="pillars">
      <div class="pillar">
        <h3>v1 top10 drift5 RP</h3>
        <p>Stage 2 trend gates + IBD composite ranker. Top 10 names from S&amp;P 500 + FTSE 100 + 3 metals. Monthly rebal, drift buffer 5, replace mode. The strategy you've been planning to run.</p>
      </div>
      <div class="pillar">
        <h3>MarketFighter clone</h3>
        <p>Factor basket: IUMO/IUQA/IUVL/CUS1/IEMO/IEQU/IEVL/WSML. Sector basket: IUIT/IHCU/IUES/IUCD/IUCS/IUIS/IUMS. Each month, pick the top 1 in each basket by 12m return. 50/50 split. All LSE-listed UCITS, T212-accessible.</p>
      </div>
    </div>

    <h2>Equity curves</h2>
    <div class="card dark">{chart}</div>

    <h2>Head-to-head</h2>
    <div class="card">
      <table>
        <thead><tr>
          <th>CONFIG</th>
          <th>OVERLAY</th>
          <th class="right">CAGR</th>
          <th class="right">SHARPE</th>
          <th class="right">MAX DD</th>
          <th class="right">END £100k</th>
          <th class="right">TRADES</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
      <p class="muted" style="margin-top:16px;">
        Sorted by Sharpe. Cash overlay = 10-month SMA on SPY; at month-end, if SPY closed below SMA, that leg goes to cash next month. Same survivorship-bias caveat applies to v1 (universe is current S&amp;P 500 + FTSE 100 constituents). MarketFighter is index-based so survivorship bias does not apply to it.
      </p>
    </div>

    <div class="foot">
      <p>
        <strong>Caveats.</strong><br/>
        Window starts {START} (gated by WSML.L inception). Includes COVID 2020 crash, 2022 bear, AI boom 2023-25. Slightly different from author's 2000-2025 window — his published CAGR is partly inflated by the DotCom-to-recovery window. This test is on a fair like-for-like recent period.<br/>
        WSML.L is a substitute for "MSCI Europe Small Cap" (UCITS for that index doesn't appear on yfinance). World small cap covers similar factor.<br/>
        Trade costs modelled at 5 bps round-trip. UK CGT not modelled.
      </p>
      <p>
        <a href="index.html">← Live dashboard</a>
        &nbsp;|&nbsp; <a href="backtest.html">v1 backtest sweep</a>
        &nbsp;|&nbsp; <a href="minervini.html">v2 experimental</a>
        &nbsp;|&nbsp; <a href="https://github.com/soylee22/momentum-power-scanner">Source</a>
      </p>
    </div>
  </div>
</body>
</html>
"""
    HTML.write_text(html)
    return HTML


def main() -> None:
    print("Loading data...")
    v1_prices = _load_long(PRICES_V1)
    mf_prices = _load_long(PRICES_MF)
    universe = load_universe()
    spy_series = _spy_series(mf_prices)

    print("Benchmarks...")
    spy_bench = fetch_benchmark("SPY")
    vwrp_bench = fetch_benchmark("VWRP.L")
    benchmarks = {}
    if not spy_bench.empty:
        benchmarks["SPY"] = spy_bench
    if not vwrp_bench.empty:
        benchmarks["VWRP"] = vwrp_bench

    results = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # v1: top10 drift5 RP, no overlay
    print("  > v1_top10_drift5_RP_baseline")
    cfg = BacktestConfig(top_n=10, rebalance_freq="monthly", weighting="equal",
                         drift_buffer=5, trade_mode="replace", start=START)
    r = run_backtest(v1_prices, universe, cfg, benchmarks=benchmarks)
    results["v1_top10_drift5_RP_baseline"] = r
    m = r.metrics
    print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k")

    # v1 + cash overlay
    print("  > v1_top10_drift5_RP_overlay10mo")
    cfg = BacktestConfig(top_n=10, rebalance_freq="monthly", weighting="equal",
                         drift_buffer=5, trade_mode="replace", start=START,
                         overlay_sma_months=10, overlay_ticker="SPY")
    r = run_backtest(v1_prices, universe, cfg, benchmarks=benchmarks, overlay_prices=spy_series)
    results["v1_top10_drift5_RP_overlay10mo"] = r
    m = r.metrics
    print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k")

    # MarketFighter, no overlay
    print("  > mf_clone_no_overlay")
    mfc = MFConfig(start=START, overlay_sma_months=None)
    r = run_marketfighter(mf_prices, mfc, benchmarks=benchmarks)
    results["mf_clone_no_overlay"] = r
    m = r.metrics
    print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k")

    # MarketFighter, 10mo overlay
    print("  > mf_clone_overlay10mo")
    mfc = MFConfig(start=START, overlay_sma_months=10)
    r = run_marketfighter(mf_prices, mfc, benchmarks=benchmarks)
    results["mf_clone_overlay10mo"] = r
    m = r.metrics
    print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k")

    # MarketFighter EXPANDED (11 GICS sectors + semi thematic), no overlay
    print("  > mf_expanded_no_overlay")
    mfc = MFConfig(start=START, overlay_sma_months=None)
    r = run_marketfighter(mf_prices, mfc, benchmarks=benchmarks,
                         sector_basket=SECTOR_BASKET_EXPANDED)
    results["mf_expanded_no_overlay"] = r
    m = r.metrics
    print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k")

    # MarketFighter EXPANDED + overlay
    print("  > mf_expanded_overlay10mo")
    mfc = MFConfig(start=START, overlay_sma_months=10)
    r = run_marketfighter(mf_prices, mfc, benchmarks=benchmarks,
                         sector_basket=SECTOR_BASKET_EXPANDED)
    results["mf_expanded_overlay10mo"] = r
    m = r.metrics
    print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k")

    # Apply OUR Stage 2 + composite algorithm to the ETF basket (top 2 from
    # pooled 15+ ETFs). Tests whether the trend-template filter adds value
    # over plain relative momentum.
    print("  > stage2_on_etf_top2_drift2_RP")
    etf_tickers = FACTOR_BASKET + SECTOR_BASKET_EXPANDED
    etf_universe = pd.DataFrame({
        "ticker": etf_tickers,
        "name": etf_tickers,
        "sector": "ETF",
        "industry": "ETF",
        "exchange": "LSE",
        "country": "UK",
        "currency": "GBP",
        "index": "MF Basket",
    })
    etf_cfg = BacktestConfig(
        top_n=2, rebalance_freq="monthly", weighting="equal",
        drift_buffer=2, trade_mode="replace", start=START, min_universe=3,
    )
    r = run_backtest(mf_prices, etf_universe, etf_cfg, benchmarks=benchmarks)
    results["stage2_on_etf_top2"] = r
    m = r.metrics
    print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k")

    # Same but with cash overlay
    print("  > stage2_on_etf_top2_overlay10mo")
    etf_cfg = BacktestConfig(
        top_n=2, rebalance_freq="monthly", weighting="equal",
        drift_buffer=2, trade_mode="replace", start=START, min_universe=3,
        overlay_sma_months=10, overlay_ticker="SPY",
    )
    r = run_backtest(mf_prices, etf_universe, etf_cfg, benchmarks=benchmarks, overlay_prices=spy_series)
    results["stage2_on_etf_top2_overlay"] = r
    m = r.metrics
    print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k")

    for name, r in results.items():
        r.equity_curve.assign(config=name).to_parquet(OUT_DIR / f"equity_{name}.parquet", index=False)
        r.holdings.assign(config=name).to_parquet(OUT_DIR / f"holdings_{name}.parquet", index=False)
        r.trades.assign(config=name).to_parquet(OUT_DIR / f"trades_{name}.parquet", index=False)

    out = write_report(results, benchmarks)
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
