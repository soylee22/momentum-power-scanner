"""GitHub Pages dashboard renderer.

Reads all snapshots from data/history/, renders a single static HTML
page in the Palantir aesthetic with:
  - Latest top 10 (sortable, click-through to per-ticker chart)
  - Survivor count + universe size sparkline-style strip
  - Full snapshot history JSON loaded inline for client-side filtering
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


def _watchlist_html(watchlist_json: str, survivors: int) -> str:
    return f"""
    <div class="watchlist" id="watchlist">
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
      <table id="wl-table">
        <thead>
          <tr>
            <th data-key="rank_overall" data-dir="asc" class="sorted asc">#</th>
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

    <script>
      const WL_DATA = {watchlist_json};
      const tbody = document.querySelector('#wl-table tbody');
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
          const flag = r.index === 'S&P 500' ? '🇺🇸' : '🇬🇧';
          const ret = (r.return_12m*100);
          const retCls = ret >= 0 ? 'num accent' : 'num';
          const retStr = (ret>=0?'+':'') + ret.toFixed(0) + '%';
          return `
            <tr>
              <td class="rank">${{r.rank_overall}}</td>
              <td class="ticker"><span class="tk">${{r.ticker}}</span></td>
              <td>${{r.name||''}}</td>
              <td>${{r.sector||''}}</td>
              <td>${{flag}}</td>
              <td class="num">${{fmtInt(r.rs_rating)}}</td>
              <td class="num">${{fmtPct(r.dist_from_high)}}</td>
              <td class="${{retCls}}">${{retStr}}</td>
              <td class="num">${{fmtNum3(r.composite)}}</td>
            </tr>`;
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
    </script>
"""


def render_dashboard() -> Path:
    snaps = _load_snapshots()
    if not snaps:
        raise FileNotFoundError("No snapshots in data/history/. Run the scanner first.")
    latest = snaps[-1]

    DOCS.mkdir(parents=True, exist_ok=True)
    DOCS_CHARTS.mkdir(parents=True, exist_ok=True)

    # Copy charts (relative URLs work better on Pages than absolute)
    if OUTPUTS_CHARTS.exists():
        for src in OUTPUTS_CHARTS.glob("*.png"):
            shutil.copy(src, DOCS_CHARTS / src.name)

    history_path = DOCS / "history.json"
    history_path.write_text(json.dumps(snaps, indent=2, default=float))

    rows_html = []
    for r in latest["top10"]:
        flag = "🇺🇸" if r["index"] == "S&P 500" else "🇬🇧"
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

    universe = latest["universe_size"]
    survivors = latest["survivor_count"]
    rate = (survivors / universe * 100) if universe else 0.0

    # Deeper survivor list (ranks 11+). Older snapshots without 'survivors'
    # gracefully degrade — the card simply doesn't render.
    survivors_full = latest.get("survivors") or []
    watchlist = [r for r in survivors_full if int(r.get("rank_overall", 0)) > 10]
    watchlist_json = json.dumps(watchlist, default=float)
    watchlist_section = ""
    if watchlist:
        watchlist_section = _watchlist_html(watchlist_json, survivors)

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
    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 64px 40px 80px;
    }}
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
    .lede {{
      max-width: 640px;
      color: #4A4A47;
      font-size: 16px;
      line-height: 1.6;
    }}
    hr {{ border: 0; border-top: 1px solid var(--hairline-lt); margin: 56px 0 36px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 32px;
      margin: 24px 0 40px;
    }}
    .stat .num {{
      font-size: 56px;
      font-weight: 600;
      letter-spacing: -0.025em;
      line-height: 1;
    }}
    .stat.accent .num {{ color: var(--green); }}
    .stat .lbl {{
      font-size: 10px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--mute);
      margin-top: 8px;
    }}
    .card {{
      background: var(--ink);
      color: var(--bone);
      border-radius: 28px;
      padding: 36px 32px;
    }}
    .card h2 {{
      font-size: 24px;
      font-weight: 500;
      letter-spacing: -0.005em;
      margin: 0 0 24px;
    }}
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
    td.rank {{ color: var(--mute); font-size: 12px; width: 36px; }}
    td.ticker a {{ color: var(--bone); text-decoration: none; display: block; }}
    td.ticker a:hover .tk {{ color: var(--green); }}
    td.ticker .tk {{ font-weight: 600; font-size: 14px; }}
    td.ticker .tk-meta {{ display: block; color: var(--mute); font-size: 11px; margin-top: 2px; }}
    td.num {{ text-align: right; font-size: 13px; }}
    td.num.accent {{ color: var(--green); font-weight: 600; }}
    .watchlist {{
      background: var(--bone);
      color: var(--ink);
      border: 1px solid var(--hairline-lt);
      border-radius: 28px;
      padding: 36px 32px;
      margin-top: 32px;
    }}
    .watchlist h2 {{
      font-size: 24px;
      font-weight: 500;
      letter-spacing: -0.005em;
      margin: 0 0 8px;
    }}
    .watchlist .sub {{
      font-size: 12px;
      color: var(--mute);
      letter-spacing: 0.04em;
      margin-bottom: 24px;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
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
    .watchlist thead th {{
      color: var(--mute);
      border-bottom: 1px solid var(--hairline-lt);
      cursor: pointer;
      user-select: none;
    }}
    .watchlist thead th.sorted::after {{ content: " \\25BE"; color: var(--graphite); }}
    .watchlist thead th.sorted.asc::after {{ content: " \\25B4"; }}
    .watchlist tbody td {{
      border-bottom: 1px solid var(--hairline-lt);
      color: var(--ink);
    }}
    .watchlist td.rank {{ color: var(--mute); }}
    .watchlist td.ticker .tk {{ color: var(--ink); font-weight: 600; }}
    .watchlist td.ticker .tk-meta {{ color: var(--mute); }}
    .watchlist td.num.accent {{ color: var(--green); }}
    .empty {{
      padding: 24px;
      text-align: center;
      color: var(--mute);
      font-size: 13px;
    }}
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
      .card {{ padding: 24px 20px; }}
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

    {watchlist_section}

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
