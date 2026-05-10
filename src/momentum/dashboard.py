"""GitHub Pages dashboard renderer.

Reads all snapshots from data/history/, renders a single static HTML
page in the Palantir aesthetic with four tabs:
  - Top 10:       latest headline names, click-through to per-ticker chart
  - Watchlist:    all Stage 2 survivors beyond the top 10 (ranks 11+),
                  sortable + filterable
  - Persistence:  every ticker that has ever made the top 10, with
                  weeks-in-top-10, best rank, current rank, and a
                  composite-trajectory sparkline
  - Breadth:      survivor-count line chart over time (market breadth tell)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .palantir import PL

ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_DIR = ROOT / "data" / "history"
OUTPUTS_CHARTS = ROOT / "outputs" / "charts"
DOCS = ROOT / "docs"
DOCS_CHARTS = DOCS / "charts"


def _load_snapshots() -> list[dict]:
    snaps = sorted(HISTORY_DIR.glob("*.json"))
    return [json.loads(p.read_text()) for p in snaps]


# ---------------------------------------------------------------------------
# Aggregations across snapshots
# ---------------------------------------------------------------------------


def _persistence(snaps: list[dict]) -> list[dict]:
    """For every ticker that has appeared in top 10 across all snapshots,
    return a summary row with weeks-in-top-10, best rank, current rank
    (or None if dropped), last seen date, and composite trajectory."""
    if not snaps:
        return []
    by_ticker: dict[str, dict] = {}
    for snap in snaps:
        asof = snap["asof"]
        items = snap.get("survivors") or snap.get("top10") or []
        for r in items:
            tk = r["ticker"]
            row = by_ticker.setdefault(tk, {
                "ticker": tk,
                "name": r.get("name", ""),
                "sector": r.get("sector", ""),
                "index": r.get("index", ""),
                "appearances": [],
            })
            row["appearances"].append({
                "asof": asof,
                "rank": r.get("rank_overall"),
                "composite": r.get("composite"),
            })

    latest_asof = snaps[-1]["asof"]
    rows: list[dict] = []
    for info in by_ticker.values():
        apps = info["appearances"]
        weeks_in_top10 = sum(1 for a in apps if a["rank"] and a["rank"] <= 10)
        if weeks_in_top10 == 0:
            continue
        ranks = [a["rank"] for a in apps if a["rank"]]
        best_rank = min(ranks) if ranks else None
        last_app = apps[-1]
        currently_present = last_app["asof"] == latest_asof
        current_rank = last_app["rank"] if currently_present else None
        trajectory = [a["composite"] for a in apps if a["composite"] is not None]
        rows.append({
            "ticker": info["ticker"],
            "name": info["name"],
            "sector": info["sector"],
            "index": info["index"],
            "weeks_in_top10": weeks_in_top10,
            "total_appearances": len(apps),
            "best_rank": best_rank,
            "current_rank": current_rank,
            "last_seen": last_app["asof"],
            "trajectory": trajectory,
        })
    rows.sort(key=lambda r: (-r["weeks_in_top10"], r["best_rank"] or 9999))
    return rows


def _breadth(snaps: list[dict]) -> list[dict]:
    return [
        {
            "asof": s["asof"],
            "universe_size": s["universe_size"],
            "survivor_count": s["survivor_count"],
            "survivor_rate": (s["survivor_count"] / s["universe_size"] * 100) if s["universe_size"] else 0.0,
        }
        for s in snaps
    ]


# ---------------------------------------------------------------------------
# Inline SVG helpers
# ---------------------------------------------------------------------------


def _sparkline_svg(values: list[float], width: int = 80, height: int = 24) -> str:
    if not values:
        return f'<svg width="{width}" height="{height}"></svg>'
    if len(values) == 1:
        return (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<circle cx="{width/2}" cy="{height/2}" r="2.5" fill="currentColor"/>'
            f'</svg>'
        )
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = i * (width / (n - 1))
        y = height - ((v - vmin) / span) * (height - 4) - 2
        pts.append((x, y))
    last_x, last_y = pts[-1]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<polyline fill="none" stroke="currentColor" stroke-width="1.4" points="{polyline}"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2" fill="currentColor"/>'
        f'</svg>'
    )


def _breadth_chart_svg(breadth_data: list[dict], width: int = 960, height: int = 320) -> str:
    if not breadth_data:
        return '<svg></svg>'
    counts = [d["survivor_count"] for d in breadth_data]
    asofs = [d["asof"] for d in breadth_data]
    n = len(counts)
    pad_l, pad_r, pad_t, pad_b = 64, 24, 32, 48
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    if n == 1:
        cx = pad_l + plot_w / 2
        cy = pad_t + plot_h / 2
        return (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<text x="{cx}" y="{cy - 36}" text-anchor="middle" font-size="10" '
            f'fill="currentColor" opacity="0.5" letter-spacing="0.14em">'
            f'SINGLE SNAPSHOT · TIME SERIES BUILDS WEEKLY</text>'
            f'<circle cx="{cx}" cy="{cy}" r="6" fill="currentColor"/>'
            f'<text x="{cx}" y="{cy + 32}" text-anchor="middle" font-size="13" '
            f'fill="currentColor">{counts[0]} survivors · {asofs[0]}</text>'
            f'</svg>'
        )

    vmin, vmax = min(counts), max(counts)
    span = (vmax - vmin) or 1
    pts = []
    for i, v in enumerate(counts):
        x = pad_l + i * (plot_w / (n - 1))
        y = pad_t + plot_h - ((v - vmin) / span) * plot_h
        pts.append((x, y, v, asofs[i]))
    polyline_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="currentColor"/>'
        for x, y, _, _ in pts
    )
    step = max(1, n // 8)
    x_labels = "".join(
        f'<text x="{x:.1f}" y="{height - 16}" text-anchor="middle" '
        f'font-size="10" fill="currentColor" opacity="0.55">{asof[5:]}</text>'
        for i, (x, _, _, asof) in enumerate(pts) if i % step == 0 or i == n - 1
    )
    y_ticks = []
    for frac in (0.0, 0.5, 1.0):
        val = vmin + span * frac
        y = pad_t + plot_h - frac * plot_h
        y_ticks.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="currentColor" stroke-opacity="0.08"/>'
        )
        y_ticks.append(
            f'<text x="{pad_l - 10}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="currentColor" opacity="0.55">{int(val)}</text>'
        )
    y_ticks_html = "".join(y_ticks)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet" style="max-width:100%;height:auto;">'
        f'{y_ticks_html}'
        f'<polyline fill="none" stroke="currentColor" stroke-width="1.6" points="{polyline_pts}"/>'
        f'{dots}{x_labels}'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _top10_card(latest: dict) -> str:
    rows_html = []
    for r in latest.get("top10", []):
        flag = "\U0001F1FA\U0001F1F8" if r["index"] == "S&P 500" else "\U0001F1EC\U0001F1E7"
        ticker_id = r["ticker"].replace(".", "_")
        rows_html.append(f"""
          <tr>
            <td class="rank">{int(r['rank_overall']):02d}</td>
            <td class="ticker">
              <a href="charts/{ticker_id}.png">
                <span class="tk">{r['ticker']}</span>
                <span class="tk-meta">{flag} {r.get('name', '')} · {r.get('sector', '')}</span>
              </a>
            </td>
            <td class="num">{r['rs_rating']:.0f}</td>
            <td class="num">{r['dist_from_high']*100:.1f}%</td>
            <td class="num accent">+{r['return_12m']*100:.0f}%</td>
            <td class="num">{r['composite']:.3f}</td>
          </tr>
        """)
    rows = "\n".join(rows_html)
    return f"""
    <div class="card">
      <h2>The list</h2>
      <table>
        <thead>
          <tr>
            <th>RANK</th>
            <th>NAME</th>
            <th class="right">RS</th>
            <th class="right">52wH dist</th>
            <th class="right">1Y</th>
            <th class="right">COMPOSITE</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


def _watchlist_card(watchlist_json: str, survivors: int) -> str:
    return f"""
    <div class="lightcard" id="watchlist">
      <h2>Watchlist · ranks 11 to {survivors}</h2>
      <div class="sub">All Stage 2 survivors beyond the top 10. Sortable, filterable. Click a column header to sort.</div>
      <div class="controls">
        <input type="search" id="wl-search" placeholder="Filter by ticker, name, sector...">
        <select id="wl-index">
          <option value="">All indices</option>
          <option value="S&amp;P 500">S&amp;P 500</option>
          <option value="FTSE 100">FTSE 100</option>
        </select>
        <select id="wl-sector"><option value="">All sectors</option></select>
      </div>
      <table id="wl-table" class="interactive">
        <thead>
          <tr>
            <th data-key="rank_overall" class="sorted asc">#</th>
            <th data-key="ticker">TICKER</th>
            <th data-key="name">NAME</th>
            <th data-key="sector">SECTOR</th>
            <th data-key="index">IDX</th>
            <th class="right" data-key="rs_rating">RS</th>
            <th class="right" data-key="dist_from_high">52wH DIST</th>
            <th class="right" data-key="return_12m">1Y</th>
            <th class="right" data-key="composite">COMPOSITE</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
      <div class="empty" id="wl-empty" style="display:none;">No matches.</div>
    </div>
    """


def _persistence_card(rows: list[dict]) -> str:
    if not rows:
        return """
        <div class="lightcard">
          <h2>Persistence</h2>
          <div class="empty">No persistence data yet — first weekly scan still loading.</div>
        </div>
        """
    row_html = []
    for r in rows:
        flag = "\U0001F1FA\U0001F1F8" if r["index"] == "S&P 500" else "\U0001F1EC\U0001F1E7"
        spark = _sparkline_svg(r["trajectory"])
        current = (
            f'<span class="num">{r["current_rank"]}</span>'
            if r["current_rank"] is not None
            else '<span class="muted">dropped</span>'
        )
        row_html.append(f"""
          <tr>
            <td class="ticker"><span class="tk">{r['ticker']}</span></td>
            <td>{flag} {r['name']} <span class="muted">· {r['sector']}</span></td>
            <td class="num accent">{r['weeks_in_top10']}</td>
            <td class="num">{r['best_rank'] or '—'}</td>
            <td class="num">{current}</td>
            <td class="spark">{spark}</td>
            <td class="num muted">{r['last_seen']}</td>
          </tr>
        """)
    body = "\n".join(row_html)
    return f"""
    <div class="lightcard">
      <h2>Persistence · weeks in top 10</h2>
      <div class="sub">
        Every ticker that has reached the top 10 at any weekly scan. Sparkline shows composite score across all appearances. "Now" = current top-10 rank, or "dropped" if not in this week's headline ten.
      </div>
      <table>
        <thead>
          <tr>
            <th>TICKER</th>
            <th>NAME</th>
            <th class="right">WEEKS IN T10</th>
            <th class="right">BEST</th>
            <th class="right">NOW</th>
            <th>TRAJECTORY</th>
            <th class="right">LAST SEEN</th>
          </tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """


def _breadth_card(breadth_data: list[dict]) -> str:
    if not breadth_data:
        return """
        <div class="lightcard">
          <h2>Breadth</h2>
          <div class="empty">No breadth data yet.</div>
        </div>
        """
    chart = _breadth_chart_svg(breadth_data)
    latest = breadth_data[-1]
    earliest = breadth_data[0]
    counts = [d["survivor_count"] for d in breadth_data]
    delta = counts[-1] - counts[0] if len(counts) > 1 else 0
    delta_str = f"{delta:+d}" if delta != 0 else "0"
    return f"""
    <div class="lightcard">
      <h2>Breadth · Stage 2 survivors over time</h2>
      <div class="sub">
        How many of ~{latest['universe_size']} S&amp;P 500 + FTSE 100 names pass all 8 Minervini gates each week. Rising = healthy trend market with broad participation. Falling = narrowing leadership.
      </div>
      <div class="chart-wrap">{chart}</div>
      <div class="breadth-meta">
        <span><span class="meta-num">{latest['survivor_count']}</span><span class="meta-lbl">THIS WEEK</span></span>
        <span><span class="meta-num">{delta_str}</span><span class="meta-lbl">VS FIRST SNAPSHOT ({earliest['asof']})</span></span>
        <span><span class="meta-num">{latest['survivor_rate']:.1f}%</span><span class="meta-lbl">SURVIVOR RATE</span></span>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Main dashboard render
# ---------------------------------------------------------------------------


def render_dashboard() -> Path:
    snaps = _load_snapshots()
    if not snaps:
        raise FileNotFoundError("No snapshots in data/history/. Run the scanner first.")
    latest = snaps[-1]

    DOCS.mkdir(parents=True, exist_ok=True)
    DOCS_CHARTS.mkdir(parents=True, exist_ok=True)

    if OUTPUTS_CHARTS.exists():
        for src in OUTPUTS_CHARTS.glob("*.png"):
            shutil.copy(src, DOCS_CHARTS / src.name)

    history_path = DOCS / "history.json"
    history_path.write_text(json.dumps(snaps, indent=2, default=float))

    universe = latest["universe_size"]
    survivors = latest["survivor_count"]
    rate = (survivors / universe * 100) if universe else 0.0

    # Sections
    top10_section = _top10_card(latest)

    survivors_full = latest.get("survivors") or []
    watchlist = [r for r in survivors_full if int(r.get("rank_overall", 0)) > 10]
    watchlist_json = json.dumps(watchlist, default=float)
    watchlist_section = (
        _watchlist_card(watchlist_json, survivors)
        if watchlist
        else """
        <div class="lightcard">
          <h2>Watchlist</h2>
          <div class="empty">First populated by the next weekly scan.</div>
        </div>
        """
    )

    persistence_rows = _persistence(snaps)
    persistence_section = _persistence_card(persistence_rows)

    breadth_data = _breadth(snaps)
    breadth_section = _breadth_card(breadth_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Momentum Power · {latest['asof']}</title>
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
      --hairline-dk: {PL.hairline_dk};
      --hairline-lt: {PL.hairline_lt};
      --green: {PL.accent_green};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--stage);
      color: var(--graphite);
      font-family: "Inter", system-ui, -apple-system, sans-serif;
      font-size: 15px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{ max-width: 1080px; margin: 0 auto; padding: 64px 40px 80px; }}
    .chip {{
      display: inline-block;
      padding: 8px 16px;
      border: 1px solid var(--graphite);
      border-radius: 999px;
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.08em;
      color: var(--graphite);
    }}
    h1 {{
      font-size: 56px;
      font-weight: 400;
      letter-spacing: -0.018em;
      line-height: 1.05;
      margin: 28px 0 16px;
    }}
    .lede {{ max-width: 640px; color: #4A4A47; font-size: 16px; line-height: 1.6; }}
    hr {{ border: 0; border-top: 1px solid var(--hairline-lt); margin: 56px 0 36px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 32px;
      margin: 24px 0 40px;
    }}
    .stat .num {{ font-size: 56px; font-weight: 600; letter-spacing: -0.025em; line-height: 1; }}
    .stat.accent .num {{ color: var(--green); }}
    .stat .lbl {{ font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--mute); margin-top: 8px; }}

    /* Tabs */
    .tabs {{
      display: flex;
      gap: 4px;
      margin: 12px 0 28px;
      border-bottom: 1px solid var(--hairline-lt);
      flex-wrap: wrap;
    }}
    .tab-btn {{
      font-family: inherit;
      font-size: 11px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-weight: 500;
      background: transparent;
      border: 0;
      padding: 14px 18px;
      cursor: pointer;
      color: var(--mute);
      border-bottom: 2px solid transparent;
      margin-bottom: -1px;
      transition: color 120ms ease;
    }}
    .tab-btn:hover {{ color: var(--graphite); }}
    .tab-btn.active {{ color: var(--ink); border-bottom-color: var(--ink); }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}

    /* Cards */
    .card {{
      background: var(--ink);
      color: var(--bone);
      border-radius: 28px;
      padding: 36px 32px;
    }}
    .card h2 {{ font-size: 24px; font-weight: 500; letter-spacing: -0.005em; margin: 0 0 24px; }}
    .lightcard {{
      background: var(--bone);
      color: var(--ink);
      border: 1px solid var(--hairline-lt);
      border-radius: 28px;
      padding: 36px 32px;
    }}
    .lightcard h2 {{ font-size: 24px; font-weight: 500; letter-spacing: -0.005em; margin: 0 0 8px; }}
    .lightcard .sub {{ font-size: 12px; color: var(--mute); letter-spacing: 0.02em; margin-bottom: 24px; line-height: 1.5; max-width: 720px; }}

    /* Tables */
    table {{ width: 100%; border-collapse: collapse; }}
    thead th {{
      text-align: left;
      font-size: 9px;
      font-weight: 500;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--mute);
      padding: 8px 10px;
      border-bottom: 1px solid var(--hairline-dk);
    }}
    thead th.right {{ text-align: right; }}
    tbody td {{
      padding: 14px 10px;
      border-bottom: 1px solid var(--hairline-dk);
      font-variant-numeric: tabular-nums;
    }}
    .card td.rank {{ color: var(--mute); font-size: 12px; width: 36px; }}
    .card td.ticker a {{ color: var(--bone); text-decoration: none; display: block; }}
    .card td.ticker a:hover .tk {{ color: var(--green); }}
    .card td.ticker .tk {{ font-weight: 600; font-size: 14px; }}
    .card td.ticker .tk-meta {{ display: block; color: var(--mute); font-size: 11px; margin-top: 2px; }}
    td.num {{ text-align: right; font-size: 13px; }}
    td.num.accent {{ color: var(--green); font-weight: 600; }}
    .muted {{ color: var(--mute); }}

    /* Light-card overrides */
    .lightcard thead th {{ color: var(--mute); border-bottom: 1px solid var(--hairline-lt); }}
    .lightcard table.interactive thead th {{ cursor: pointer; user-select: none; }}
    .lightcard table.interactive thead th.sorted::after {{ content: " \\25BE"; color: var(--graphite); }}
    .lightcard table.interactive thead th.sorted.asc::after {{ content: " \\25B4"; }}
    .lightcard tbody td {{ border-bottom: 1px solid var(--hairline-lt); color: var(--ink); }}
    .lightcard td.ticker .tk {{ color: var(--ink); font-weight: 600; font-size: 14px; }}
    .lightcard td.num.accent {{ color: var(--green); }}

    /* Watchlist controls */
    .controls {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
    .controls input, .controls select {{
      font-family: inherit;
      font-size: 13px;
      padding: 8px 12px;
      border: 1px solid var(--hairline-lt);
      border-radius: 999px;
      background: var(--stage);
      color: var(--ink);
    }}
    .controls input {{ min-width: 220px; flex: 1; }}

    /* Persistence sparklines */
    td.spark {{ color: var(--graphite); width: 100px; }}
    td.spark svg {{ display: block; }}

    /* Breadth chart */
    .chart-wrap {{
      background: var(--stage);
      border-radius: 16px;
      padding: 16px;
      color: var(--ink);
      overflow-x: auto;
    }}
    .breadth-meta {{
      display: flex;
      gap: 48px;
      margin-top: 24px;
      flex-wrap: wrap;
    }}
    .breadth-meta .meta-num {{
      font-size: 32px;
      font-weight: 600;
      letter-spacing: -0.02em;
      display: block;
      line-height: 1;
    }}
    .breadth-meta .meta-lbl {{
      font-size: 10px;
      letter-spacing: 0.12em;
      color: var(--mute);
      text-transform: uppercase;
      display: block;
      margin-top: 6px;
    }}

    .empty {{ padding: 24px; text-align: center; color: var(--mute); font-size: 13px; }}

    .foot {{
      margin-top: 56px;
      font-size: 11px;
      color: var(--mute);
      letter-spacing: 0.04em;
      line-height: 1.6;
    }}
    .foot a {{ color: var(--graphite); }}
    @media (max-width: 720px) {{
      h1 {{ font-size: 36px; }}
      .stats {{ grid-template-columns: 1fr; }}
      .card, .lightcard {{ padding: 24px 20px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <span class="chip">MOMENTUM POWER&nbsp;&nbsp;|&nbsp;&nbsp;{latest['asof']}</span>
    <h1>Top 10 momentum names this week.</h1>
    <p class="lede">
      Stage 2 trend-template survivors, ranked by composite IBD-style relative
      strength, 52-week-high proximity, and one-year return. Local-currency
      returns. S&amp;P 500 + FTSE 100 universe.
    </p>

    <hr/>

    <div class="stats">
      <div class="stat"><div class="num">{survivors}</div><div class="lbl">STAGE 2 SURVIVORS</div></div>
      <div class="stat"><div class="num">{universe}</div><div class="lbl">UNIVERSE SCANNED</div></div>
      <div class="stat accent"><div class="num">{rate:.1f}%</div><div class="lbl">SURVIVOR RATE</div></div>
    </div>

    <div class="tabs">
      <button class="tab-btn active" data-tab="list">Top 10</button>
      <button class="tab-btn" data-tab="watchlist">Watchlist</button>
      <button class="tab-btn" data-tab="persistence">Persistence</button>
      <button class="tab-btn" data-tab="breadth">Breadth</button>
    </div>

    <div class="tab-panel active" data-panel="list">
      {top10_section}
    </div>
    <div class="tab-panel" data-panel="watchlist">
      {watchlist_section}
    </div>
    <div class="tab-panel" data-panel="persistence">
      {persistence_section}
    </div>
    <div class="tab-panel" data-panel="breadth">
      {breadth_section}
    </div>

    <div class="foot">
      Filter: Mark Minervini Stage 2 trend-template (8 binary gates).
      Ranker: composite of IBD-style RS rating + 52-week-high proximity + 1-year
      return. Independent computation, yfinance data, no third-party feeds.
      <br/><br/>
      <a href="methodology.html">Read the full methodology &rarr;</a>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <a href="history.json">Snapshot history (JSON)</a>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <a href="https://github.com/soylee22/momentum-power-scanner">Source repo</a>
    </div>
  </div>

  <script>
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const t = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.querySelector('.tab-panel[data-panel="' + t + '"]').classList.add('active');
        if (history.replaceState) history.replaceState(null, '', '#' + t);
      }});
    }});
    // Honour hash on load
    const hash = window.location.hash.replace('#','');
    if (hash) {{
      const btn = document.querySelector('.tab-btn[data-tab="' + hash + '"]');
      if (btn) btn.click();
    }}

    // Watchlist sort + filter
    (function() {{
      const WL_DATA = {watchlist_json};
      if (!WL_DATA.length) return;
      const tbody = document.querySelector('#wl-table tbody');
      if (!tbody) return;
      const searchEl = document.getElementById('wl-search');
      const indexEl = document.getElementById('wl-index');
      const sectorEl = document.getElementById('wl-sector');
      const emptyEl = document.getElementById('wl-empty');
      const headers = document.querySelectorAll('#wl-table thead th[data-key]');
      let sortKey = 'rank_overall';
      let sortDir = 'asc';

      const sectors = [...new Set(WL_DATA.map(r => r.sector).filter(Boolean))].sort();
      for (const s of sectors) {{
        const opt = document.createElement('option');
        opt.value = s; opt.textContent = s;
        sectorEl.appendChild(opt);
      }}

      function fmtPct(x) {{ return (x*100).toFixed(1) + '%'; }}
      function fmtNum3(x) {{ return Number(x).toFixed(3); }}
      function fmtInt(x) {{ return Math.round(x).toString(); }}

      function render() {{
        const q = searchEl.value.trim().toLowerCase();
        const idx = indexEl.value;
        const sec = sectorEl.value;
        let rows = WL_DATA.filter(r => {{
          if (idx && r.index !== idx) return false;
          if (sec && r.sector !== sec) return false;
          if (!q) return true;
          const hay = (r.ticker + ' ' + (r.name||'') + ' ' + (r.sector||'')).toLowerCase();
          return hay.includes(q);
        }});
        rows.sort((a,b) => {{
          let va = a[sortKey], vb = b[sortKey];
          if (typeof va === 'string') {{ va = va.toLowerCase(); vb = (vb||'').toLowerCase(); }}
          if (va < vb) return sortDir === 'asc' ? -1 : 1;
          if (va > vb) return sortDir === 'asc' ? 1 : -1;
          return 0;
        }});
        emptyEl.style.display = rows.length ? 'none' : 'block';
        tbody.innerHTML = rows.map(r => {{
          const flag = r.index === 'S&P 500' ? '\U0001F1FA\U0001F1F8' : '\U0001F1EC\U0001F1E7';
          const ret = (r.return_12m*100);
          const retCls = ret >= 0 ? 'num accent' : 'num';
          const retStr = (ret>=0?'+':'') + ret.toFixed(0) + '%';
          return '<tr>' +
            '<td class="rank">' + r.rank_overall + '</td>' +
            '<td class="ticker"><span class="tk">' + r.ticker + '</span></td>' +
            '<td>' + (r.name||'') + '</td>' +
            '<td>' + (r.sector||'') + '</td>' +
            '<td>' + flag + '</td>' +
            '<td class="num">' + fmtInt(r.rs_rating) + '</td>' +
            '<td class="num">' + fmtPct(r.dist_from_high) + '</td>' +
            '<td class="' + retCls + '">' + retStr + '</td>' +
            '<td class="num">' + fmtNum3(r.composite) + '</td>' +
          '</tr>';
        }}).join('');
      }}

      headers.forEach(h => h.addEventListener('click', () => {{
        const key = h.dataset.key;
        if (sortKey === key) {{
          sortDir = sortDir === 'asc' ? 'desc' : 'asc';
        }} else {{
          sortKey = key;
          sortDir = ['rank_overall','ticker','name','sector','index'].includes(key) ? 'asc' : 'desc';
        }}
        headers.forEach(x => x.classList.remove('sorted','asc'));
        h.classList.add('sorted');
        if (sortDir === 'asc') h.classList.add('asc');
        render();
      }}));

      [searchEl, indexEl, sectorEl].forEach(el => el.addEventListener('input', render));
      render();
    }})();
  </script>
</body>
</html>
"""
    out = DOCS / "index.html"
    out.write_text(html)
    return out


def main() -> None:
    out = render_dashboard()
    print(f"dashboard -> {out}")


if __name__ == "__main__":
    main()
