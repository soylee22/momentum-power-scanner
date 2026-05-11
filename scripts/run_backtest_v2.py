"""Minervini v2 backtest — VCP + pivot-break setup-gated entries, 8% stops.

Writes:
  - data/backtest_v2/*.parquet (one per config)
  - docs/minervini.html        (separate wing of the dashboard)

Comparison vs v1 (Stage 2 + composite, no setup, no stops) is the headline.
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

from momentum.backtest import BacktestConfig, run_backtest, run_backtest_v2  # noqa: E402
from momentum.universe import load_universe  # noqa: E402
from momentum.palantir import PL  # noqa: E402

PRICES_PATH = ROOT / "data" / "prices_backtest.parquet"
OUT_DIR = ROOT / "data" / "backtest_v2"
DOCS = ROOT / "docs"
HTML = DOCS / "minervini.html"

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


def load_prices() -> pd.DataFrame:
    df = pd.read_parquet(PRICES_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def sweep_configs(start: dt.date) -> list[tuple[str, BacktestConfig, str]]:
    """Each item: (name, config, family) where family is one of:
        - 'v1'  (pure rank, no setup, no stops)            for comparison
        - 'v2'  (setup-gated entry + stops)                the new thing
        - 'v2_stops_only' (stops on, no setup gate)        isolates the stop effect
        - 'v2_setup_only' (setup gate, no stop)            isolates the setup effect
    """
    items = []

    # v1 baselines — same engine as before
    for top_n in (10,):
        for buf in (0, 5):
            items.append((
                f"v1_top{top_n}_drift{buf}_RP",
                BacktestConfig(top_n=top_n, rebalance_freq="monthly",
                               weighting="equal", drift_buffer=buf,
                               trade_mode="replace", start=start),
                "v1",
            ))

    # v2 — stops only (no setup filter)
    for stop in (0.07, 0.08, 0.10):
        items.append((
            f"v2_top10_drift5_RP_stop{int(stop*100)}",
            BacktestConfig(top_n=10, rebalance_freq="monthly",
                           weighting="equal", drift_buffer=5,
                           trade_mode="replace", start=start,
                           stop_loss_pct=stop),
            "v2_stops_only",
        ))

    # v2 — setup only (no stops)
    items.append((
        "v2_top10_drift5_RP_setup",
        BacktestConfig(top_n=10, rebalance_freq="monthly",
                       weighting="equal", drift_buffer=5,
                       trade_mode="replace", start=start,
                       require_setup=True),
        "v2_setup_only",
    ))

    # v2 — full Minervini (setup + 8% stop) at top 10 / top 5
    for top_n in (10, 5):
        items.append((
            f"v2_top{top_n}_drift5_RP_setup_stop8",
            BacktestConfig(top_n=top_n, rebalance_freq="monthly",
                           weighting="equal", drift_buffer=5,
                           trade_mode="replace", start=start,
                           require_setup=True, stop_loss_pct=0.08),
            "v2",
        ))
        items.append((
            f"v2_top{top_n}_drift0_RP_setup_stop8",
            BacktestConfig(top_n=top_n, rebalance_freq="monthly",
                           weighting="equal", drift_buffer=0,
                           trade_mode="replace", start=start,
                           require_setup=True, stop_loss_pct=0.08),
            "v2",
        ))

    return items


def _equity_chart_svg(curves: dict[str, pd.Series], width: int = 1000, height: int = 360) -> str:
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


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:+.1f}%"


def _fmt_mult(x):
    if x is None:
        return "—"
    return f"{x:.2f}"


def write_report(results: dict[str, dict], benchmarks: dict) -> Path:
    """results: {key: {'result': BacktestResult, 'family': str}}"""
    # Curves: v1 baseline + v2 full + benchmarks
    keys_to_plot = [
        "v1_top10_drift5_RP",
        "v2_top10_drift5_RP_stop8",
        "v2_top10_drift5_RP_setup",
        "v2_top10_drift5_RP_setup_stop8",
    ]
    curves = {}
    for k in keys_to_plot:
        if k in results:
            s = results[k]["result"].equity_curve.set_index("date")["portfolio_value"]
            curves[k] = s
    # Benchmark from any result
    any_eq = next(iter(results.values()))["result"].equity_curve.set_index("date")
    for bench in benchmarks:
        if bench in any_eq.columns:
            curves[bench] = any_eq[bench]

    chart = _equity_chart_svg(curves)

    # Sweep table
    rows = []
    for key, info in results.items():
        r = info["result"]
        m = r.metrics
        rows.append({
            "config": key, "family": info["family"],
            "top_n": r.config.top_n, "rebal": r.config.rebalance_freq,
            "drift": r.config.drift_buffer, "mode": r.config.trade_mode,
            "setup": "✓" if r.config.require_setup else "—",
            "stop": f"{r.config.stop_loss_pct*100:.0f}%" if r.config.stop_loss_pct else "—",
            "cagr": m.get("cagr"), "sharpe": m.get("sharpe"),
            "max_dd": m.get("max_drawdown"),
            "end": m.get("end_value"),
            "trades": len(r.trades),
        })
    rows.sort(key=lambda x: (x["sharpe"] or -999), reverse=True)
    rows_html = "".join(
        f"""<tr class="fam-{r['family']}">
          <td><code>{r['config']}</code></td>
          <td><span class="badge bg-{r['family']}">{r['family']}</span></td>
          <td class="num">{r['top_n']}</td>
          <td class="num">{r['drift']}</td>
          <td>{r['setup']}</td>
          <td>{r['stop']}</td>
          <td class="num">{_fmt_pct(r['cagr'])}</td>
          <td class="num"><strong>{_fmt_mult(r['sharpe'])}</strong></td>
          <td class="num neg">{_fmt_pct(r['max_dd'])}</td>
          <td class="num">£{(r['end'] or 0)/1000:.1f}k</td>
          <td class="num muted">{r['trades']}</td>
        </tr>""" for r in rows
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Momentum Power · Minervini v2</title>
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
    .lede {{ max-width: 720px; color: #4A4A47; font-size: 16px; line-height: 1.6; }}
    .card {{ background: var(--bone); border: 1px solid var(--hairline-lt); border-radius: 24px; padding: 28px; margin-top: 20px; color: var(--ink); }}
    .card.dark {{ background: var(--ink); color: var(--bone); border: 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--hairline-lt); font-variant-numeric: tabular-nums; }}
    thead th {{ text-align: left; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--mute); font-weight: 500; }}
    td.num {{ text-align: right; font-size: 13px; }}
    td.neg {{ color: #B23D3D; }}
    .muted {{ color: var(--mute); font-size: 12px; }}
    code {{ font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: 11px; background: rgba(0,0,0,0.04); padding: 1px 6px; border-radius: 3px; }}
    .legend {{ margin-top: 12px; font-size: 12px; line-height: 1.8; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; }}
    .bg-v1 {{ background: rgba(110,143,198,0.18); color: #4566A0; }}
    .bg-v2 {{ background: rgba(124,166,120,0.22); color: #4F7E4B; }}
    .bg-v2_setup_only {{ background: rgba(200,146,61,0.18); color: #966817; }}
    .bg-v2_stops_only  {{ background: rgba(165,94,94,0.18); color: #8E3E3E; }}
    .foot {{ margin-top: 56px; font-size: 11px; color: var(--mute); }}
    .pillars {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 20px 0; }}
    .pillar {{ padding: 18px; background: var(--bone); border-radius: 14px; border: 1px solid var(--hairline-lt); }}
    .pillar h3 {{ margin: 0 0 8px; font-size: 14px; letter-spacing: 0.04em; text-transform: uppercase; }}
    .pillar p {{ margin: 0; font-size: 13px; line-height: 1.55; color: var(--graphite); }}
  </style>
</head>
<body>
  <div class="page">
    <span class="chip">MOMENTUM POWER&nbsp;&nbsp;|&nbsp;&nbsp;MINERVINI v2 (EXPERIMENTAL WING)</span>
    <h1>Adding the pressure layer.</h1>
    <p class="lede">
      v1 was Minervini-inspired but materially incomplete: Stage 2 gates + IBD-style
      composite ranking, no entry pattern, no stops. This wing adds the two pieces
      that make Minervini's actual edge: a VCP + pivot-break setup gate on entries,
      and an 8% hard stop on every position. Then re-runs the backtest.
    </p>

    <h2>What's new in v2</h2>
    <div class="pillars">
      <div class="pillar">
        <h3>VCP + pivot setup gate</h3>
        <p>
          A name only becomes a buy candidate if its volatility (ATR-20 / ATR-60) is
          contracting, its volume is drying up, it's within 5% of the 52-week high,
          AND today it broke above the prior 10-day high on >1.3x average volume.
          Filters out Stage-2-but-extended names entirely.
        </p>
      </div>
      <div class="pillar">
        <h3>8% hard stop</h3>
        <p>
          Every position carries an entry-price stop set at purchase. If close ≤ entry
          × (1 - stop_pct) on any day between rebalances, position is sold immediately
          at that day's close. Cash sits idle until next rebalance. Position-level
          loss capped; concentration risk capped.
        </p>
      </div>
    </div>

    <h2>Equity curves</h2>
    <div class="card dark">{chart}</div>

    <h2>v1 vs v2 sweep</h2>
    <div class="card">
      <table>
        <thead><tr>
          <th>CONFIG</th><th>FAMILY</th>
          <th class="right">N</th><th class="right">DRIFT</th>
          <th>SETUP</th><th>STOP</th>
          <th class="right">CAGR</th><th class="right">SHARPE</th>
          <th class="right">MAX DD</th><th class="right">END £100k</th>
          <th class="right">TRADES</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
      <p class="muted" style="margin-top:16px;">
        Sorted by Sharpe. Families: <code>v1</code> = current scanner; <code>v2_setup_only</code> = setup gate without stops; <code>v2_stops_only</code> = stops without setup gate; <code>v2</code> = full Minervini playbook.
      </p>
    </div>

    <div class="foot">
      <p>
        <strong>What this experiment can and can't tell you.</strong><br/>
        VCP detection here is numerical (ATR + volume contraction + near-high + breakout day), not true pattern recognition. The real Minervini-grade VCP is harder to encode robustly across hundreds of names. So if the v2 setup filter LOOKS too loose, that's possibly why.<br/>
        Stops are clean and faithful — 8% hard at close, no slippage modelling.<br/>
        Same survivorship-bias caveat as v1: universe is today's S&amp;P 500 + FTSE 100 constituents.
      </p>
      <p>
        <a href="index.html">← Live dashboard</a>
        &nbsp;|&nbsp; <a href="backtest.html">v1 backtest</a>
        &nbsp;|&nbsp; <a href="methodology.html">Methodology</a>
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
    print("Loading prices...")
    prices = load_prices()
    universe = load_universe()
    start = dt.date(2017, 1, 1)
    print(f"Fetching benchmarks from {start}...")
    benchmarks = {}
    for tk in ("SPY", "VWRP.L"):
        s = fetch_benchmark(tk, start)
        if not s.empty:
            benchmarks[tk.replace(".L", "")] = s

    configs = sweep_configs(start)
    print(f"Running {len(configs)} configs...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, cfg, family in configs:
        print(f"  > {name} ({family})", flush=True)
        try:
            if family == "v1":
                r = run_backtest(prices, universe, cfg, benchmarks=benchmarks)
            else:
                r = run_backtest_v2(prices, universe, cfg, benchmarks=benchmarks)
        except Exception as exc:
            print(f"    FAILED: {exc!r}")
            continue
        results[name] = {"result": r, "family": family}
        r.equity_curve.assign(config=name).to_parquet(OUT_DIR / f"equity_{name}.parquet", index=False)
        r.holdings.assign(config=name).to_parquet(OUT_DIR / f"holdings_{name}.parquet", index=False)
        r.trades.assign(config=name).to_parquet(OUT_DIR / f"trades_{name}.parquet", index=False)
        m = r.metrics
        print(f"    CAGR={(m.get('cagr') or 0)*100:5.1f}%  Sharpe={(m.get('sharpe') or 0):4.2f}  "
              f"DD={(m.get('max_drawdown') or 0)*100:5.1f}%  End=£{(m.get('end_value') or 0)/1000:.1f}k  "
              f"Trades={len(r.trades)}")

    print("Writing report...")
    out = write_report(results, benchmarks)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
