# Momentum Power Scanner

Weekly scanner that screens **S&P 500 + FTSE 100** for **Mark Minervini Stage 2 Trend Template** survivors, ranks them by an **IBD-style relative strength** composite, and publishes the **top 10 momentum names** as a Gmail digest, GitHub Pages dashboard, and per-ticker Palantir-aesthetic charts.

Built to mirror the methodology behind retail systematic-momentum portfolios such as Benson212's *US Momentum Power* T212 pie. Independent: this is its own scanner, not derived from any third-party feed.

## Methodology

### 1. Universe
- **S&P 500** (Wikipedia constituents, cached weekly)
- **FTSE 100** (Wikipedia constituents, `.L` suffix for yfinance)
- ~600 names. Returns computed in **local currency** (USD/GBP) so FX noise does not pollute the momentum signal.

### 2. Filter — Minervini Stage 2 Trend Template (8 binary gates)
All must pass.

1. Price > 150-day SMA AND Price > 200-day SMA
2. 150-day SMA > 200-day SMA
3. 200-day SMA trending up for ≥ 1 month (today's 200d > 200d 21 trading days ago)
4. 50-day SMA > 150-day SMA AND 50-day SMA > 200-day SMA
5. Price > 50-day SMA
6. Price ≥ 30 % above 52-week low
7. Price within 25 % of 52-week high
8. RS rating ≥ 70 (computed across the full universe)

### 3. Rank — composite among survivors

For each survivor, take the **percentile rank within the survivor pool** of:
- IBD-style RS rating (`0.4·r3m + 0.2·r6m + 0.2·r9m + 0.2·r12m`, then percentile-ranked across the full universe)
- 52-week-high proximity (closer = higher)
- 1-year total return

Composite = mean of those three ranks. **Top 10 by composite.**

### 4. Outputs each run

- `outputs/latest.csv` — top 10 with all gate flags + scores
- `outputs/latest.md` — markdown table
- `outputs/charts/{TICKER}.png` — Palantir-dark chart with 50/150/200d MAs + 52w high/low bands
- `outputs/digest_latest.html` — editorial Gmail digest (Palantir aesthetic)
- `docs/index.html` — GitHub Pages dashboard with sortable table + history
- `data/history/YYYY-MM-DD.json` — append-only weekly snapshots

## Running locally

```bash
just install          # uv venv + editable install
just scan-dry         # full pipeline, no email
just scan             # full pipeline + Gmail send (needs GMAIL_APP_PASSWORD)
just dashboard        # rebuild docs/index.html from latest
just test             # pytest
```

## Schedule

GitHub Actions cron `0 21 * * 0` — Sunday 21:00 UTC (after Friday US close, before Monday Asia open). Local LaunchAgent fallback is optional.

## Repo layout

```
src/momentum/
  palantir.py        Vendored Palantir style tokens (matplotlib + CSS palette)
  universe.py        S&P 500 + FTSE 100 ticker fetch (Wikipedia, weekly cache)
  fetch.py           yfinance batch download with parquet cache
  minervini.py       8 trend-template gates, returns boolean + per-gate dict
  rs_rating.py       IBD-style RS rating (1-99 percentile)
  composite.py       Composite ranker over survivors
  scanner.py         Pipeline orchestrator (entry point: mscan)
  charts.py          Per-ticker matplotlib charts
  digest.py          Gmail HTML editorial digest
  dashboard.py       GitHub Pages HTML
data/history/        Weekly snapshot JSONs (committed)
data/cache/          yfinance parquet cache (gitignored)
docs/                GitHub Pages source
outputs/             Latest run artefacts
```
