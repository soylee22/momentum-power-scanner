"""Main pipeline orchestrator.

Universe -> prices -> features -> RS rating -> Minervini gates ->
survivor pool -> composite rank -> top 10 -> persist + render
charts + dashboard + email digest.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from .composite import rank_composite
from .fetch import load_prices
from .minervini import evaluate_gates, features_from_prices
from .rs_rating import rs_rating_table
from .universe import load_universe

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS = ROOT / "outputs"
HISTORY_DIR = ROOT / "data" / "history"

console = Console()


def _dedupe_share_classes(survivors: pd.DataFrame) -> pd.DataFrame:
    """Where a single company has multiple share classes in the survivor pool
    (e.g. Alphabet GOOGL/GOOG, Berkshire BRK.A/BRK.B, Fox FOX/FOXA), keep
    only one — the lower-priced (more retail-accessible) class.

    Detection: strip class-suffix patterns from the company name and group.
    Selection: keep the lowest-price ticker in each group.
    """
    if survivors.empty:
        return survivors
    df = survivors.copy()
    base = (
        df["name"].astype(str)
        .str.replace(r"\s*\(\s*Class\s+[A-Z]\s*\)\s*", "", regex=True, case=False)
        .str.replace(r"\s+Class\s+[A-Z]\s*$", "", regex=True, case=False)
        .str.strip()
        .str.rstrip(".")
    )
    df["_base_name"] = base
    df = df.sort_values("price", ascending=True).drop_duplicates("_base_name", keep="first")
    return df.drop(columns=["_base_name"]).sort_index()


def run(asof: dt.date | None = None, top_n: int = 10) -> dict:
    asof = asof or dt.date.today()
    console.rule(f"[bold]Momentum Power Scan · asof {asof.isoformat()}")

    # 1. Universe
    universe = load_universe()
    us_n = int((universe["country"] == "US").sum()) if "country" in universe.columns else 0
    uk_n = int(
        ((universe["country"] == "UK") & ~universe["index"].astype(str).str.contains("ETF", na=False)).sum()
    ) if "country" in universe.columns else 0
    console.print(
        f"Universe: {len(universe)} tickers "
        f"(US {us_n}, UK {uk_n}; Yahoo ≥ $1bn)"
    )

    # 2. Prices
    console.print("Fetching prices (this can take a few minutes the first time)...")
    prices = load_prices(universe["ticker"].tolist(), asof=asof)
    console.print(f"  prices loaded for {len(prices)} tickers")

    # 3. Features
    feats = []
    for tk, df in prices.items():
        f = features_from_prices(tk, df)
        if f is not None:
            feats.append(f)
    console.print(f"  features computed for {len(feats)} tickers")

    # 4. RS rating
    rs = rs_rating_table(feats)
    rs_lookup = dict(zip(rs["ticker"], rs["rs_rating"]))

    # 5. Apply gates
    rows = []
    for f in feats:
        rs_score = float(rs_lookup.get(f.ticker, float("nan")))
        gates = evaluate_gates(f, rs_score if rs_score == rs_score else 0.0)
        rows.append({
            "ticker": f.ticker,
            "price": f.price,
            "sma50": f.sma50,
            "sma150": f.sma150,
            "sma200": f.sma200,
            "high_52w": f.high_52w,
            "low_52w": f.low_52w,
            "return_3m": f.return_3m,
            "return_6m": f.return_6m,
            "return_9m": f.return_9m,
            "return_12m": f.return_12m,
            "rs_rating": rs_score,
            "stage2": all(gates.values()),
            "gates_passed": int(sum(gates.values())),
            **{k: v for k, v in gates.items()},
        })
    df = pd.DataFrame(rows)
    df = df.merge(
        universe[["ticker", "name", "sector", "country", "currency", "index"]],
        on="ticker", how="left",
    )

    survivors = df[df["stage2"]].copy()
    console.print(
        f"  Stage 2 survivors: [bold green]{len(survivors)}[/] / {len(df)}  "
        f"(US {(survivors['country'] == 'US').sum() if 'country' in survivors.columns else '?'}, "
        f"UK {(survivors['country'] == 'UK').sum() if 'country' in survivors.columns else '?'})"
    )

    survivors_before_dedupe = len(survivors)
    survivors = _dedupe_share_classes(survivors)
    if len(survivors) < survivors_before_dedupe:
        console.print(
            f"  Dedupe share classes: {survivors_before_dedupe} -> {len(survivors)} "
            f"(kept the cheaper / more retail-accessible class)"
        )

    if survivors.empty:
        console.print("[red]No Stage 2 survivors this week. Saving an empty top 10.[/]")
        ranked = pd.DataFrame()
    else:
        ranked = rank_composite(survivors)

    top = ranked.head(top_n).copy() if not ranked.empty else pd.DataFrame()

    # 6. Persist
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # full universe scored snapshot (compact)
    df.sort_values(
        ["stage2", "rs_rating"], ascending=[False, False]
    ).to_parquet(OUTPUTS / "scored_universe.parquet", index=False)

    if not top.empty:
        top.to_csv(OUTPUTS / "latest.csv", index=False)
        _write_markdown(top, OUTPUTS / "latest.md", asof)

    snapshot_cols = [
        "rank_overall", "ticker", "name", "sector", "country", "index",
        "price", "rs_rating", "dist_from_high",
        "return_3m", "return_6m", "return_12m",
        "rank_rs", "rank_prox", "rank_1yr", "composite",
    ]
    # Persist the FULL ranked survivor pool, not just the top 10. The top 10
    # is the headline; the deeper list (ranks 11 to ~survivors_count) is the
    # near-miss watchlist and gets surfaced on the dashboard for filtering.
    snapshot = {
        "asof": asof.isoformat(),
        "universe_size": int(len(df)),
        "survivor_count": int(len(survivors)),
        "top10": top[
            [c for c in snapshot_cols if c not in ("return_3m", "return_6m")]
        ].to_dict(orient="records") if not top.empty else [],
        "survivors": ranked[
            [c for c in snapshot_cols if c in ranked.columns]
        ].to_dict(orient="records") if not ranked.empty else [],
    }
    snap_path = HISTORY_DIR / f"{asof.isoformat()}.json"
    snap_path.write_text(json.dumps(snapshot, indent=2, default=float))
    console.print(f"  snapshot -> {snap_path.relative_to(ROOT)}")

    # 7. Render top 10 in console
    _print_top(top)

    return {"asof": asof, "top": top, "survivors": survivors, "all": df}


def _index_short(index, country=None) -> str:
    """Compact index label for console table."""
    idx = str(index or "")
    if country == "US" or idx.startswith("US") or idx == "S&P 500":
        return "US"
    if country == "UK" or idx.startswith(("UK", "FTSE")) or idx.endswith("ETF"):
        return "UK"
    return (idx[:6] if idx else "?")


def _write_markdown(top: pd.DataFrame, path: Path, asof: dt.date) -> None:
    lines = [
        f"# Momentum Power · Top 10 · {asof.isoformat()}",
        "",
        "| # | Ticker | Name | Index | Sector | Price | RS | 52wH dist | 1y return | Composite |",
        "|---|--------|------|-------|--------|------:|---:|---------:|---------:|---------:|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {int(r['rank_overall'])} | `{r['ticker']}` | {r.get('name', '')} | "
            f"{r.get('index', '')} | {r.get('sector', '')} | "
            f"{r['price']:.2f} | {r['rs_rating']:.0f} | "
            f"{r['dist_from_high'] * 100:.1f}% | {r['return_12m'] * 100:+.1f}% | "
            f"{r['composite']:.3f} |"
        )
    path.write_text("\n".join(lines))


def _print_top(top: pd.DataFrame) -> None:
    if top.empty:
        return
    table = Table(title="Top 10 by composite momentum", show_lines=False, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Ticker")
    table.add_column("Name", max_width=28)
    table.add_column("Idx", justify="center")
    table.add_column("Sector", max_width=22)
    table.add_column("RS", justify="right")
    table.add_column("52wH dist", justify="right")
    table.add_column("1y", justify="right")
    table.add_column("Composite", justify="right")
    for _, r in top.iterrows():
        table.add_row(
            str(int(r["rank_overall"])),
            r["ticker"],
            str(r.get("name", "")),
            _index_short(r.get("index"), r.get("country")),
            str(r.get("sector", "")),
            f"{r['rs_rating']:.0f}",
            f"{r['dist_from_high'] * 100:.1f}%",
            f"{r['return_12m'] * 100:+.1f}%",
            f"{r['composite']:.3f}",
        )
    console.print(table)


def main() -> None:
    result = run()

    # Optional downstream renders (charts + digest + dashboard).
    # These are no-ops if the corresponding env disables them.
    if os.environ.get("MOMENTUM_SKIP_CHARTS") != "1":
        try:
            from .charts import render_top_charts
            render_top_charts(result["top"], result["all"])
        except Exception as exc:
            console.print(f"[yellow]charts skipped: {exc!r}[/]")

    if os.environ.get("MOMENTUM_SKIP_DASHBOARD") != "1":
        try:
            from .dashboard import render_dashboard
            render_dashboard()
        except Exception as exc:
            console.print(f"[yellow]dashboard skipped: {exc!r}[/]")

    if os.environ.get("MOMENTUM_SKIP_DIGEST") != "1":
        try:
            from .digest import send_digest
            send_digest(result["asof"])
        except Exception as exc:
            console.print(f"[yellow]digest skipped: {exc!r}[/]")


if __name__ == "__main__":
    main()
