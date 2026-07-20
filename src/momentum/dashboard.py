"""GitHub Pages dashboard renderer.

Reads all snapshots from data/history/, renders a single static HTML
page in the Palantir aesthetic with four tabs:
  - Top 100:      latest headline names, click-through to per-ticker chart
  - Watchlist:    Stage 2 survivors beyond the headline (ranks 101+),
                  sortable + filterable
  - Persistence:  every ticker that has ever made the headline list, with
                  weeks-in-list, best rank, current rank, and a
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

# Keep in sync with scanner.DEFAULT_TOP_N
DEFAULT_TOP_N = 100


def _headline_n(snap: dict) -> int:
    """Headline list size for a snapshot (default 100; older snaps may differ)."""
    n = snap.get("top_n")
    if n is not None:
        try:
            return max(1, int(n))
        except (TypeError, ValueError):
            pass
    return DEFAULT_TOP_N


def _headline_rows(snap: dict) -> list[dict]:
    """Top-N composite rows for the main list.

    Prefer the full survivor pool (always ranked) so older snapshots that only
    stored 10 headline rows still expand to top 100 on rebuild.
    """
    n = _headline_n(snap)
    survivors = snap.get("survivors") or []
    if survivors:
        ordered = sorted(
            survivors,
            key=lambda r: int(r.get("rank_overall") or 10**9),
        )
        return ordered[:n]
    return list(snap.get("top10") or [])[:n]


def _load_snapshots() -> list[dict]:
    snaps = sorted(HISTORY_DIR.glob("*.json"))
    return [json.loads(p.read_text()) for p in snaps]


def _index_glyph(idx: str) -> str:
    """Map index/asset class -> emoji glyph for the dashboard tables."""
    if idx in ("S&P 500", "US $1bn+") or (idx or "").startswith("US"):
        return "\U0001F1FA\U0001F1F8"  # 🇺🇸
    if idx in ("FTSE 100", "FTSE 250", "UK $1bn+") or (idx or "").startswith(("FTSE", "UK")):
        return "\U0001F1EC\U0001F1E7"  # 🇬🇧
    if idx == "Commodity ETF":
        return "\U0001F947"            # 🥇 gold medal
    if idx == "Thematic ETF":
        return "\U0001F4A1"            # 💡 idea / theme
    if idx == "Sector ETF":
        return "\U0001F4CA"            # 📊 sector
    if idx == "Factor ETF":
        return "\u2696"                # ⚖ factor
    if idx == "Crypto ETF":
        return "₿"                     # bitcoin sign
    return "•"


# ---------------------------------------------------------------------------
# Aggregations across snapshots
# ---------------------------------------------------------------------------


def _persistence(snaps: list[dict]) -> list[dict]:
    """For every ticker that has appeared in the headline list across snapshots,
    return weeks-in-list, best rank, current rank, last seen, trajectory.

    Threshold is per-snapshot headline size (100 now; 10 on older weeks that
    only stored a short top10 without a full survivors pool).
    """
    if not snaps:
        return []
    by_ticker: dict[str, dict] = {}
    for snap in snaps:
        asof = snap["asof"]
        n = _headline_n(snap)
        # Older snaps may only have top10 (10 rows) and no survivors — use that.
        items = _headline_rows(snap) if snap.get("survivors") else (snap.get("top10") or [])
        # For persistence counting, treat rank <= n as "in headline" that week.
        for r in items:
            rank = r.get("rank_overall")
            if rank is None or int(rank) > n:
                continue
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
                "rank": rank,
                "composite": r.get("composite"),
            })

    latest_asof = snaps[-1]["asof"]
    latest_n = _headline_n(snaps[-1])
    rows: list[dict] = []
    for info in by_ticker.values():
        apps = info["appearances"]
        weeks_in_list = len(apps)
        if weeks_in_list == 0:
            continue
        ranks = [a["rank"] for a in apps if a["rank"]]
        best_rank = min(ranks) if ranks else None
        last_app = apps[-1]
        currently_present = last_app["asof"] == latest_asof and (
            last_app["rank"] is not None and int(last_app["rank"]) <= latest_n
        )
        # If last appearance isn't this week, check latest survivors for current rank.
        current_rank = None
        if last_app["asof"] == latest_asof:
            current_rank = last_app["rank"]
        else:
            for r in _headline_rows(snaps[-1]):
                if r.get("ticker") == info["ticker"]:
                    current_rank = r.get("rank_overall")
                    currently_present = True
                    break
        trajectory = [a["composite"] for a in apps if a["composite"] is not None]
        rows.append({
            "ticker": info["ticker"],
            "name": info["name"],
            "sector": info["sector"],
            "index": info["index"],
            "weeks_in_top10": weeks_in_list,  # field name kept for template BC
            "total_appearances": len(apps),
            "best_rank": best_rank,
            "current_rank": current_rank if currently_present else None,
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


def _pct(x) -> str:
    """Render a 0-1 percentile rank as a 0-100 integer string. None -> '—'."""
    if x is None:
        return "—"
    try:
        return f"{x*100:.0f}"
    except Exception:
        return "—"


def _k_ratio_cell(x) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
        if v != v:  # NaN
            return "—"
        return f"{v:.1f}"
    except Exception:
        return "—"


def _top10_card(latest: dict) -> str:
    headline = _headline_rows(latest)
    n = len(headline)
    rows_html = []
    for r in headline:
        flag = _index_glyph(r.get("index") or "")
        ticker_id = r["ticker"].replace(".", "_")
        rank = int(r["rank_overall"])
        rows_html.append(f"""
          <tr>
            <td class="rank">{rank:02d}</td>
            <td class="ticker">
              <a href="charts/{ticker_id}.png">
                <span class="tk">{r['ticker']}</span>
                <span class="tk-meta">{flag} {r.get('name', '')} · {r.get('sector', '')}</span>
              </a>
            </td>
            <td class="num">{r['rs_rating']:.0f}</td>
            <td class="num">{r['dist_from_high']*100:.1f}%</td>
            <td class="num accent">+{r['return_12m']*100:.0f}%</td>
            <td class="num">{_k_ratio_cell(r.get('k_ratio'))}</td>
            <td class="num comp-rs">{_pct(r.get('rank_rs'))}</td>
            <td class="num comp-prox">{_pct(r.get('rank_prox'))}</td>
            <td class="num comp-1y">{_pct(r.get('rank_1yr'))}</td>
            <td class="num comp-k">{_pct(r.get('rank_k'))}</td>
            <td class="num composite">{r['composite']:.3f}</td>
          </tr>
        """)
    rows = "\n".join(rows_html)
    return f"""
    <div class="card">
      <h2>The list · top {n}</h2>
      <div class="card-sub">
        RS, 52wH dist, 1Y and K-ratio are the raw inputs. The four columns under
        <span class="tag">COMPONENTS</span> are within-survivors percentile ranks (0-100)
        feeding the composite. K-ratio (Kestner) is path consistency — a one-off spike
        can still post a huge 1Y return but will lag on K. See the <em>Method</em> tab.
      </div>
      <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th rowspan="2">RANK</th>
            <th rowspan="2">NAME</th>
            <th class="right" rowspan="2">RS</th>
            <th class="right" rowspan="2">52wH dist</th>
            <th class="right" rowspan="2">1Y</th>
            <th class="right" rowspan="2">K</th>
            <th class="right group" colspan="4">COMPONENTS (percentile)</th>
            <th class="right" rowspan="2">COMPOSITE</th>
          </tr>
          <tr>
            <th class="right sub">RS rk</th>
            <th class="right sub">Prox rk</th>
            <th class="right sub">1Y rk</th>
            <th class="right sub">K rk</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      </div>
    </div>
    """


def _watchlist_card(watchlist_json: str, survivors: int, after_rank: int) -> str:
    start = after_rank + 1
    return f"""
    <div class="lightcard" id="watchlist">
      <h2>Watchlist · ranks {start} to {survivors}</h2>
      <div class="sub">All Stage 2 survivors beyond the top {after_rank}. Sortable, filterable. Click a column header to sort.</div>
      <div class="controls">
        <input type="search" id="wl-search" placeholder="Filter by ticker, name, sector...">
        <select id="wl-index">
          <option value="">All indices</option>
          <option value="US $1bn+">US $1bn+</option>
          <option value="UK $1bn+">UK $1bn+</option>
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
            <th class="right" data-key="k_ratio">K</th>
            <th class="right" data-key="rank_rs">RS RK</th>
            <th class="right" data-key="rank_prox">PROX RK</th>
            <th class="right" data-key="rank_1yr">1Y RK</th>
            <th class="right" data-key="rank_k">K RK</th>
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
        flag = _index_glyph(r["index"])
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
      <h2>Persistence · weeks in top {DEFAULT_TOP_N}</h2>
      <div class="sub">
        Every ticker that has reached the headline list at any weekly scan. Sparkline shows composite score across all appearances. "Now" = current headline rank, or "dropped" if not in this week's top {DEFAULT_TOP_N}.
      </div>
      <table>
        <thead>
          <tr>
            <th>TICKER</th>
            <th>NAME</th>
            <th class="right">WEEKS IN LIST</th>
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


def _marketfighter_card() -> str:
    """Compute today's MarketFighter pick from the live price cache and render
    a tab section. Two picks (factor + sector), each by 12m trailing return."""
    try:
        import pandas as pd
        from .marketfighter import (
            compute_live_picks, FACTOR_BASKET, SECTOR_BASKET_EXPANDED,
        )
        # Prefer the MF-specific cache (always has the full basket); fall back
        # to the live scanner cache otherwise.
        mf_path = ROOT / "data" / "prices_mf.parquet"
        live_path = ROOT / "data" / "prices.parquet"
        mf_tickers = set(FACTOR_BASKET + SECTOR_BASKET_EXPANDED)
        df = None
        for path in (mf_path, live_path):
            if not path.exists():
                continue
            cand = pd.read_parquet(path)
            cand["date"] = pd.to_datetime(cand["date"]).dt.date
            cand = cand[cand["ticker"].isin(mf_tickers)]
            if not cand.empty and cand["ticker"].nunique() >= 10:
                df = cand
                break
        if df is None or df.empty:
            return ('<div class="lightcard"><h2>MarketFighter live</h2>'
                    '<div class="empty">MF tickers not yet in price cache.</div></div>')
        picks = compute_live_picks(df, overlay_sma_months=None)  # no overlay by default for live view
    except Exception as exc:
        return f'<div class="lightcard"><h2>MarketFighter live</h2><div class="empty">computation failed: {exc!r}</div></div>'

    def _pick_html(label: str, pick: dict | None, rankings: list) -> str:
        if pick is None:
            body = '<div class="mf-cash">CASH — no setup or overlay-out</div>'
        else:
            body = (
                f'<div class="mf-ticker">{pick["ticker"]}</div>'
                f'<div class="mf-meta">12m total return: '
                f'<strong>{pick["return_12m"]*100:+.1f}%</strong></div>'
            )
        top3 = "".join(
            f'<tr><td>{i+1}</td><td><code>{tk}</code></td>'
            f'<td class="num {"accent" if r>=0 else ""}">{r*100:+.1f}%</td></tr>'
            for i, (tk, r) in enumerate(rankings[:5])
        )
        return f"""
        <div class="mf-leg">
          <div class="mf-leg-label">{label}</div>
          {body}
          <table class="mf-runners">
            <thead><tr><th>#</th><th>Ticker</th><th class="right">12m</th></tr></thead>
            <tbody>{top3}</tbody>
          </table>
        </div>
        """

    factor_html = _pick_html("FACTOR LEG", picks["factor_pick"], picks["factor_rankings"])
    sector_html = _pick_html("SECTOR LEG", picks["sector_pick"], picks["sector_rankings"])
    asof = picks["asof"]
    return f"""
    <div class="lightcard">
      <h2>MarketFighter · today's pick</h2>
      <div class="sub">
        Parallel ETF rotation strategy. Each month, pick the single best-performing ETF in each basket by 12-month total return. Hold both 50/50. No Stage 2 filter, no composite ranker — pure relative momentum, as designed by the MarketFighter Substack author. Backtest 2019-04 to 2026-05 with expanded 12-ETF sector basket: <strong>23.6% CAGR, Sharpe 1.02, max DD -31%</strong>. See <a href="marketfighter.html">full comparison</a>.
      </div>
      <div class="mf-grid">
        {factor_html}
        {sector_html}
      </div>
      <p class="muted" style="margin-top:24px;">
        Asof {asof}. Trade rule: execute on the last trading day of each month if the pick differs from last month. Often it won't.
      </p>
    </div>
    """


def _method_card() -> str:
    """Annotated breakdown of how the composite score is computed."""
    return """
    <div class="lightcard method-card">
      <h2>How the composite score is calculated</h2>
      <div class="sub">
        Every Stage&nbsp;2 survivor each week is scored by a single number — the composite.
        It's the equal-weight mean of <strong>four</strong> within-pool percentile ranks.
      </div>

      <div class="pipeline">
        <span class="step">Universe<br><span class="muted">Yahoo US + UK equities ≥ $1bn</span></span>
        <span class="arrow">&rarr;</span>
        <span class="step">Stage 2 gate<br><span class="muted">8 binary Minervini tests, all must pass</span></span>
        <span class="arrow">&rarr;</span>
        <span class="step">Survivor pool<br><span class="muted">survivors this week</span></span>
        <span class="arrow">&rarr;</span>
        <span class="step gold">Composite rank<br><span class="muted">the equation below</span></span>
      </div>

      <div class="formula-stage">
        <div class="formula" style="font-size:22px;">
          <span class="lhs">composite<sub>i</sub></span>
          <span class="eq">=</span>
          <div class="frac">
            <div class="numer">
              <span class="term term-rs">rank<sub>RS</sub></span>
              <span class="plus">+</span>
              <span class="term term-prox">rank<sub>prox</sub></span>
              <span class="plus">+</span>
              <span class="term term-1y">rank<sub>1y</sub></span>
              <span class="plus">+</span>
              <span class="term term-k">rank<sub>K</sub></span>
            </div>
            <div class="denom">4</div>
          </div>
        </div>

        <div class="term-cards" style="grid-template-columns:repeat(2,1fr);">
          <div class="term-card term-rs">
            <span class="badge">RS<sub>rk</sub></span>
            <h3>IBD-style Relative Strength</h3>
            <p>Weighted blend of 3m/6m/9m/12m returns (front-loaded), percentile-ranked across the full universe to 1–99, then re-ranked within survivors.</p>
          </div>
          <div class="term-card term-prox">
            <span class="badge">prox<sub>rk</sub></span>
            <h3>52-week-high proximity</h3>
            <p>Distance from 52-week high. Closer to the high ranks higher. Penalises faded former leaders.</p>
          </div>
          <div class="term-card term-1y">
            <span class="badge">1y<sub>rk</sub></span>
            <h3>One-year return</h3>
            <p>Total return over 252 trading days, local currency. Anchors the score in capital gain magnitude.</p>
          </div>
          <div class="term-card term-k">
            <span class="badge">K<sub>rk</sub></span>
            <h3>K-ratio (path consistency)</h3>
            <p>Kestner K-ratio: fit <code>log(price) = a + b·t</code> over 252 days, take <code>b / se(b)</code> (trend-slope t-stat). A smooth grind scores high; a one-off spike then chop scores low even if total 1Y return is huge.</p>
          </div>
        </div>

        <div class="why-block">
          <h3>Why the K-ratio is there</h3>
          <p>
            RS and 1Y both care about <em>how much</em> a stock moved. They do not care about the path.
            A biotech that was flat for 11 months then doubled in a week can top a pure-return sort.
            The K-ratio asks whether the climb was consistent. Equal-weighting it with the other three
            is what keeps one-off P moves from owning the board.
          </p>
        </div>
      </div>
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
        How many of ~{latest['universe_size']} US + UK names (≥ $1bn market cap) pass all 8 Minervini gates each week. Rising = healthy trend market with broad participation. Falling = narrowing leadership.
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

    # Sync charts: drop stale names so docs/ only has this week's headline set.
    if OUTPUTS_CHARTS.exists():
        wanted = {p.name for p in OUTPUTS_CHARTS.glob("*.png")}
        for stale in DOCS_CHARTS.glob("*.png"):
            if stale.name not in wanted:
                stale.unlink()
        for src in OUTPUTS_CHARTS.glob("*.png"):
            shutil.copy(src, DOCS_CHARTS / src.name)

    history_path = DOCS / "history.json"
    history_path.write_text(json.dumps(snaps, indent=2, default=float))

    universe = latest["universe_size"]
    survivors = latest["survivor_count"]
    rate = (survivors / universe * 100) if universe else 0.0
    headline_n = _headline_n(latest)
    headline = _headline_rows(latest)

    # Sections
    top10_section = _top10_card(latest)

    survivors_full = latest.get("survivors") or []
    watchlist = [
        r for r in survivors_full if int(r.get("rank_overall", 0)) > headline_n
    ]
    watchlist_json = json.dumps(watchlist, default=float)
    watchlist_section = (
        _watchlist_card(watchlist_json, survivors, after_rank=headline_n)
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

    method_section = _method_card()
    marketfighter_section = _marketfighter_card()

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

    /* Component-colour palette (RS / prox / 1y) — shared across tables + Method tab */
    .comp-rs   {{ color: #C8923D; }}
    .comp-prox {{ color: #6E8FC6; }}
    .comp-1y   {{ color: #7CA678; }}
    .comp-k    {{ color: #B07CA6; }}
    .card .comp-rs   {{ color: #E4B36C; }}
    .card .comp-prox {{ color: #9BB6E0; }}
    .card .comp-1y   {{ color: #A8CFA3; }}
    .card .comp-k    {{ color: #D0A0C6; }}
    .term-k {{ color: #B07CA6; }}
    .term-card.term-k {{ border-top: 3px solid #B07CA6; }}
    .term-card.term-k .badge {{ background: #B07CA6; color: var(--bone); }}
    td.composite {{ font-weight: 600; }}

    .card-sub {{
      font-size: 12px;
      color: var(--mute);
      letter-spacing: 0.02em;
      margin: -12px 0 24px;
      max-width: 720px;
      line-height: 1.55;
    }}
    .card-sub .tag {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      background: rgba(255,255,255,0.06);
      color: var(--bone);
      font-size: 9px;
      letter-spacing: 0.14em;
      margin: 0 2px;
    }}
    thead th.group {{ border-bottom: 1px solid var(--hairline-dk); text-align: right; padding-bottom: 4px; color: var(--mute); font-weight: 500; }}
    thead th.sub {{ font-size: 9px; font-weight: 400; padding-top: 4px; }}
    .table-wrap {{ overflow-x: auto; }}

    /* Method tab */
    .method-card {{ padding: 40px 32px 48px; }}
    .pipeline {{
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
      padding: 20px 0 36px;
      border-bottom: 1px solid var(--hairline-lt);
      margin-bottom: 36px;
      color: var(--ink);
      font-size: 13px;
      line-height: 1.4;
    }}
    .pipeline .step {{
      padding: 12px 16px;
      border: 1px solid var(--hairline-lt);
      border-radius: 12px;
      background: var(--stage);
    }}
    .pipeline .step.gold {{ border-color: #C8923D; background: rgba(200,146,61,0.08); }}
    .pipeline .arrow {{ color: var(--mute); font-size: 18px; }}
    .pipeline .muted {{ font-size: 11px; }}

    .formula-stage {{ padding-top: 8px; }}
    .formula {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 18px;
      font-family: "Inter", serif;
      font-size: 34px;
      letter-spacing: -0.01em;
      padding: 32px 8px 12px;
      color: var(--ink);
    }}
    .formula .lhs {{ font-weight: 500; }}
    .formula .eq {{ color: var(--mute); font-weight: 300; }}
    .formula .frac {{
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      vertical-align: middle;
    }}
    .formula .numer {{
      display: flex;
      gap: 14px;
      align-items: center;
      padding: 0 24px 10px;
      border-bottom: 2px solid var(--ink);
    }}
    .formula .denom {{ padding-top: 10px; font-weight: 500; }}
    .formula .plus {{ color: var(--mute); font-weight: 300; }}
    .formula .term {{
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 8px;
      transition: background 120ms ease;
    }}
    .formula .term:hover, .formula .term.active {{ background: rgba(0,0,0,0.06); }}
    .formula .term-rs   {{ color: #C8923D; }}
    .formula .term-prox {{ color: #6E8FC6; }}
    .formula .term-1y   {{ color: #7CA678; }}
    .formula sub, .formula sup {{ font-size: 0.55em; font-weight: 400; }}

    .braces {{
      display: block;
      width: 100%;
      max-width: 720px;
      margin: 0 auto 16px;
      height: 56px;
    }}
    .brace {{
      fill: none;
      stroke-width: 1.4;
      stroke-linecap: round;
    }}
    .brace-rs   {{ stroke: #C8923D; }}
    .brace-prox {{ stroke: #6E8FC6; }}
    .brace-1y   {{ stroke: #7CA678; }}

    .term-cards {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
      margin-bottom: 32px;
    }}
    .term-card {{
      border: 1px solid var(--hairline-lt);
      border-radius: 16px;
      padding: 22px 20px;
      background: var(--stage);
      transition: transform 120ms ease, border-color 120ms ease;
    }}
    .term-card.dim {{ opacity: 0.35; }}
    .term-card.term-rs   {{ border-top: 3px solid #C8923D; }}
    .term-card.term-prox {{ border-top: 3px solid #6E8FC6; }}
    .term-card.term-1y   {{ border-top: 3px solid #7CA678; }}
    .term-card .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 11px;
      letter-spacing: 0.08em;
      font-weight: 600;
      margin-bottom: 12px;
    }}
    .term-card.term-rs   .badge {{ background: #C8923D; color: var(--bone); }}
    .term-card.term-prox .badge {{ background: #6E8FC6; color: var(--bone); }}
    .term-card.term-1y   .badge {{ background: #7CA678; color: var(--bone); }}
    .term-card h3 {{
      font-size: 14px;
      font-weight: 600;
      margin: 0 0 12px;
      color: var(--ink);
      letter-spacing: -0.005em;
    }}
    .term-card ol {{ padding-left: 18px; margin: 0 0 10px; font-size: 13px; color: var(--graphite); line-height: 1.55; }}
    .term-card li {{ margin-bottom: 6px; }}
    .term-card code {{
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      font-size: 12px;
      background: rgba(0,0,0,0.04);
      padding: 1px 5px;
      border-radius: 3px;
    }}
    .term-card .why {{
      font-size: 12px;
      color: var(--mute);
      font-style: italic;
      margin: 8px 0 0;
      line-height: 1.5;
    }}

    .why-block {{
      background: var(--ink);
      color: var(--bone);
      border-radius: 16px;
      padding: 24px 28px;
    }}
    .why-block h3 {{
      font-size: 14px;
      font-weight: 600;
      margin: 0 0 10px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--bone);
    }}
    .why-block p {{ font-size: 13px; line-height: 1.65; color: rgba(245,243,238,0.85); margin: 0 0 12px; }}
    .why-block code {{
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      background: rgba(255,255,255,0.08);
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 12px;
      color: var(--bone);
    }}

    @media (max-width: 720px) {{
      .formula {{ font-size: 22px; gap: 10px; }}
      .term-cards {{ grid-template-columns: 1fr; }}
      .braces {{ height: 38px; }}
    }}

    /* MarketFighter live tab */
    .mf-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .mf-leg {{ border: 1px solid var(--hairline-lt); border-radius: 14px; padding: 22px 20px; background: var(--stage); }}
    .mf-leg-label {{ font-size: 10px; letter-spacing: 0.14em; color: var(--mute); text-transform: uppercase; margin-bottom: 12px; }}
    .mf-ticker {{ font-size: 36px; font-weight: 600; letter-spacing: -0.02em; line-height: 1; color: var(--ink); }}
    .mf-meta {{ font-size: 12px; color: var(--graphite); margin-top: 8px; }}
    .mf-cash {{ font-size: 24px; font-weight: 500; color: var(--mute); letter-spacing: 0.08em; padding: 6px 0; }}
    .mf-runners {{ width: 100%; margin-top: 16px; font-size: 12px; }}
    .mf-runners th {{ font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--mute); }}
    .mf-runners td {{ padding: 6px 4px; }}
    .mf-runners code {{ font-size: 11px; }}
    @media (max-width: 720px) {{
      .mf-grid {{ grid-template-columns: 1fr; }}
    }}

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
    <h1>Top {headline_n} momentum names this week.</h1>
    <p class="lede">
      Stage 2 trend-template survivors, ranked by composite IBD-style relative
      strength, 52-week-high proximity, and one-year return. Local-currency
      returns. US + UK equities with market cap ≥ $1bn. Headline list is the
      top {headline_n} by composite; full survivor pool is on the Watchlist tab.
    </p>

    <hr/>

    <div class="stats">
      <div class="stat"><div class="num">{survivors}</div><div class="lbl">STAGE 2 SURVIVORS</div></div>
      <div class="stat"><div class="num">{universe}</div><div class="lbl">UNIVERSE SCANNED</div></div>
      <div class="stat accent"><div class="num">{rate:.1f}%</div><div class="lbl">SURVIVOR RATE</div></div>
    </div>

    <div class="tabs">
      <button class="tab-btn active" data-tab="list">Top {headline_n}</button>
      <button class="tab-btn" data-tab="watchlist">Watchlist</button>
      <button class="tab-btn" data-tab="persistence">Persistence</button>
      <button class="tab-btn" data-tab="breadth">Breadth</button>
      <button class="tab-btn" data-tab="marketfighter">MarketFighter</button>
      <button class="tab-btn" data-tab="method">Method</button>
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
    <div class="tab-panel" data-panel="marketfighter">
      {marketfighter_section}
    </div>
    <div class="tab-panel" data-panel="method">
      {method_section}
    </div>

    <div class="foot">
      Filter: Mark Minervini Stage 2 trend-template (8 binary gates).
      Ranker: composite of IBD-style RS rating + 52-week-high proximity + 1-year
      return. Independent computation, yfinance data, no third-party feeds.
      <br/><br/>
      <a href="backtest.html">Backtest results &rarr;</a>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <a href="methodology.html">Full methodology</a>
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

    // Method tab: click a term to dim the others
    document.querySelectorAll('.formula .term').forEach(t => {{
      t.addEventListener('click', () => {{
        const target = t.dataset.term;
        const isActive = t.classList.contains('active');
        document.querySelectorAll('.formula .term').forEach(x => x.classList.remove('active'));
        document.querySelectorAll('.term-card').forEach(c => c.classList.remove('dim'));
        document.querySelectorAll('.brace').forEach(b => b.style.opacity = '1');
        if (!isActive) {{
          t.classList.add('active');
          document.querySelectorAll('.term-card').forEach(c => {{
            if (!c.classList.contains('term-' + target)) c.classList.add('dim');
          }});
          document.querySelectorAll('.brace').forEach(b => {{
            if (!b.classList.contains('brace-' + target)) b.style.opacity = '0.18';
          }});
        }}
      }});
    }});

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
          const flag = (
            (r.index === 'S&P 500' || r.index === 'US $1bn+' || (r.index || '').startsWith('US'))
              ? '\U0001F1FA\U0001F1F8' :
            (r.index === 'UK $1bn+' || r.index === 'FTSE 100' || r.index === 'FTSE 250'
              || (r.index || '').startsWith('FTSE') || (r.index || '').startsWith('UK'))
              ? '\U0001F1EC\U0001F1E7' :
            r.index === 'Commodity ETF'? '\U0001F947' :
            r.index === 'Crypto ETF'   ? '₿' :
            '•'
          );
          const ret = (r.return_12m*100);
          const retCls = ret >= 0 ? 'num accent' : 'num';
          const retStr = (ret>=0?'+':'') + ret.toFixed(0) + '%';
          const pctRk = v => (v == null) ? '—' : Math.round(v*100);
          return '<tr>' +
            '<td class="rank">' + r.rank_overall + '</td>' +
            '<td class="ticker"><span class="tk">' + r.ticker + '</span></td>' +
            '<td>' + (r.name||'') + '</td>' +
            '<td>' + (r.sector||'') + '</td>' +
            '<td>' + flag + '</td>' +
            '<td class="num">' + fmtInt(r.rs_rating) + '</td>' +
            '<td class="num">' + fmtPct(r.dist_from_high) + '</td>' +
            '<td class="' + retCls + '">' + retStr + '</td>' +
            '<td class="num">' + (r.k_ratio == null || isNaN(r.k_ratio) ? '—' : Number(r.k_ratio).toFixed(1)) + '</td>' +
            '<td class="num comp-rs">' + pctRk(r.rank_rs) + '</td>' +
            '<td class="num comp-prox">' + pctRk(r.rank_prox) + '</td>' +
            '<td class="num comp-1y">' + pctRk(r.rank_1yr) + '</td>' +
            '<td class="num comp-k">' + pctRk(r.rank_k) + '</td>' +
            '<td class="num composite">' + fmtNum3(r.composite) + '</td>' +
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
