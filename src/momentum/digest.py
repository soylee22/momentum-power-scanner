"""Gmail HTML editorial digest in the Palantir aesthetic.

Reads the latest snapshot from data/history/, renders an HTML email
(dark stage, bone text, emerald single-accent) and sends via Gmail SMTP
using GMAIL_APP_PASSWORD.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .palantir import PL

ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_DIR = ROOT / "data" / "history"
OUTPUTS = ROOT / "outputs"

DEFAULT_TO = "leeslater1992@gmail.com"
DEFAULT_FROM = "leeslater1992@gmail.com"
DEFAULT_DASH = "https://soylee22.github.io/momentum-power-scanner/"


def _latest_snapshot() -> dict:
    snaps = sorted(HISTORY_DIR.glob("*.json"))
    if not snaps:
        raise FileNotFoundError("No snapshots in data/history/. Run the scanner first.")
    return json.loads(snaps[-1].read_text())


def _previous_snapshot() -> dict | None:
    snaps = sorted(HISTORY_DIR.glob("*.json"))
    if len(snaps) < 2:
        return None
    return json.loads(snaps[-2].read_text())


def _classify_change(curr_top: list[dict], prev_top: list[dict]) -> dict:
    curr_set = {r["ticker"] for r in curr_top}
    prev_set = {r["ticker"] for r in prev_top} if prev_top else set()
    prev_rank = {r["ticker"]: int(r["rank_overall"]) for r in (prev_top or [])}

    entries = sorted(curr_set - prev_set)
    exits = sorted(prev_set - curr_set)
    held = curr_set & prev_set

    rank_deltas = {}
    for r in curr_top:
        tk = r["ticker"]
        if tk in held:
            rank_deltas[tk] = prev_rank[tk] - int(r["rank_overall"])  # +ve = climbed
    return {
        "entries": entries,
        "exits": exits,
        "rank_deltas": rank_deltas,
        "prev_top": prev_top or [],
    }


def _row_html(r: dict, rank_delta: int | None, is_new: bool) -> str:
    flag = "🇺🇸" if r["index"] == "S&P 500" else "🇬🇧"
    if is_new:
        arrow = '<span style="color:%s; font-weight:700; font-size:10px; letter-spacing:0.1em;">NEW</span>' % PL.accent_green
    elif rank_delta is None or rank_delta == 0:
        arrow = '<span style="color:%s;">→</span>' % PL.mute
    elif rank_delta > 0:
        arrow = '<span style="color:%s; font-weight:600;">↑%d</span>' % (PL.accent_green, rank_delta)
    else:
        arrow = '<span style="color:%s; font-weight:600;">↓%d</span>' % (PL.mute, abs(rank_delta))

    return f"""
    <tr>
      <td style="padding:11px 10px; border-bottom:1px solid {PL.hairline_dk};
                 font-family:'Inter',Arial,sans-serif; color:{PL.bone};
                 font-variant-numeric:tabular-nums; font-size:13px;">
        <div style="display:flex; align-items:baseline; gap:10px;">
          <span style="color:{PL.mute}; font-size:11px; font-weight:500; min-width:18px;">{int(r['rank_overall']):02d}</span>
          <span style="color:{PL.bone}; font-weight:600;">{r['ticker']}</span>
          {arrow}
        </div>
        <div style="color:{PL.mute}; font-size:11px; margin-top:2px; padding-left:28px;">
          {flag} {r.get('name', '')} · {r.get('sector', '')}
        </div>
      </td>
      <td style="padding:11px 10px; border-bottom:1px solid {PL.hairline_dk};
                 text-align:right; font-family:'Inter',Arial,monospace;
                 font-variant-numeric:tabular-nums; font-size:13px; color:{PL.bone};">
        {r['rs_rating']:.0f}
      </td>
      <td style="padding:11px 10px; border-bottom:1px solid {PL.hairline_dk};
                 text-align:right; font-family:'Inter',Arial,monospace;
                 font-variant-numeric:tabular-nums; font-size:13px; color:{PL.bone};">
        {r['dist_from_high']*100:.1f}%
      </td>
      <td style="padding:11px 10px; border-bottom:1px solid {PL.hairline_dk};
                 text-align:right; font-family:'Inter',Arial,monospace;
                 font-variant-numeric:tabular-nums; font-size:13px; color:{PL.accent_green}; font-weight:600;">
        +{r['return_12m']*100:.0f}%
      </td>
      <td style="padding:11px 10px; border-bottom:1px solid {PL.hairline_dk};
                 text-align:right; font-family:'Inter',Arial,monospace;
                 font-variant-numeric:tabular-nums; font-size:13px; color:{PL.bone};">
        {r['composite']:.3f}
      </td>
    </tr>
    """


def render_digest_html(snap: dict, prev: dict | None, dashboard_url: str) -> str:
    asof = snap["asof"]
    universe = snap["universe_size"]
    survivors = snap["survivor_count"]
    top = snap["top10"]

    prev_top = prev["top10"] if prev else []
    change = _classify_change(top, prev_top)

    rows = []
    for r in top:
        delta = change["rank_deltas"].get(r["ticker"])
        is_new = r["ticker"] in change["entries"]
        rows.append(_row_html(r, delta, is_new))
    rows_html = "\n".join(rows)

    movement_lines = []
    if change["entries"]:
        movement_lines.append(
            f'<span style="color:{PL.accent_green};">→ NEW</span> '
            + ", ".join(change["entries"])
        )
    if change["exits"]:
        movement_lines.append(
            f'<span style="color:{PL.mute};">↳ EXITED</span> '
            + ", ".join(change["exits"])
        )
    movement_html = (
        '<div style="margin-top:18px; font-size:12px; color:%s; '
        'font-family:Inter,Arial,sans-serif; line-height:1.6;">%s</div>'
        % (PL.bone, "<br/>".join(movement_lines))
    ) if movement_lines else ""

    rate = (survivors / universe) if universe else 0.0

    return f"""<!DOCTYPE html>
<html><body style="margin:0; padding:0; background:{PL.stage};">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{PL.stage};">
  <tr>
    <td align="center" style="padding:32px 16px;">

      <table role="presentation" width="640" cellpadding="0" cellspacing="0"
             style="background:{PL.ink}; border-radius:28px; overflow:hidden;">

        <!-- Chip + title block -->
        <tr><td style="padding:40px 40px 8px;">
          <span style="display:inline-block; padding:8px 16px; border:1px solid {PL.bone};
                       border-radius:999px; font-size:11px; font-weight:500; letter-spacing:0.08em;
                       color:{PL.bone}; font-family:Inter,Arial,sans-serif;">
            MOMENTUM POWER&nbsp;&nbsp;|&nbsp;&nbsp;{asof}
          </span>
        </td></tr>

        <tr><td style="padding:8px 40px 28px;">
          <h1 style="font-family:Inter,Arial,sans-serif; color:{PL.bone};
                     font-size:42px; font-weight:400; letter-spacing:-0.015em;
                     line-height:1.08; margin:18px 0 0;">
            Top 10 momentum names this week.
          </h1>
          <p style="font-family:Inter,Arial,sans-serif; color:{PL.mute};
                    font-size:15px; line-height:1.5; margin:18px 0 0;">
            Stage 2 trend-template survivors, ranked by composite IBD-style
            relative strength, 52-week-high proximity, and one-year return.
            Local-currency returns. S&amp;P 500 + FTSE 100 universe.
          </p>
        </td></tr>

        <!-- Stats strip -->
        <tr><td style="padding:0 40px 28px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
            <tr>
              <td style="border-top:1px solid {PL.hairline_dk}; padding:20px 0 0;">
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                  <tr>
                    <td>
                      <div style="font-family:Inter,Arial,sans-serif; color:{PL.bone};
                                  font-size:32px; font-weight:600; letter-spacing:-0.02em;">
                        {survivors}
                      </div>
                      <div style="font-family:Inter,Arial,sans-serif; color:{PL.mute};
                                  font-size:10px; letter-spacing:0.1em; text-transform:uppercase;
                                  margin-top:4px;">
                        STAGE 2 SURVIVORS
                      </div>
                    </td>
                    <td>
                      <div style="font-family:Inter,Arial,sans-serif; color:{PL.bone};
                                  font-size:32px; font-weight:600; letter-spacing:-0.02em;">
                        {universe}
                      </div>
                      <div style="font-family:Inter,Arial,sans-serif; color:{PL.mute};
                                  font-size:10px; letter-spacing:0.1em; text-transform:uppercase;
                                  margin-top:4px;">
                        UNIVERSE SCANNED
                      </div>
                    </td>
                    <td>
                      <div style="font-family:Inter,Arial,sans-serif; color:{PL.accent_green};
                                  font-size:32px; font-weight:600; letter-spacing:-0.02em;">
                        {rate*100:.1f}%
                      </div>
                      <div style="font-family:Inter,Arial,sans-serif; color:{PL.mute};
                                  font-size:10px; letter-spacing:0.1em; text-transform:uppercase;
                                  margin-top:4px;">
                        SURVIVOR RATE
                      </div>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
          </table>
        </td></tr>

        <!-- Top 10 table -->
        <tr><td style="padding:0 28px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
            <thead>
              <tr>
                <th style="padding:8px 10px; text-align:left;
                           font-family:Inter,Arial,sans-serif; font-size:9px;
                           font-weight:500; letter-spacing:0.12em; color:{PL.mute};
                           text-transform:uppercase; border-bottom:1px solid {PL.hairline_dk};">
                  RANK&nbsp;·&nbsp;NAME
                </th>
                <th style="padding:8px 10px; text-align:right;
                           font-family:Inter,Arial,sans-serif; font-size:9px;
                           font-weight:500; letter-spacing:0.12em; color:{PL.mute};
                           text-transform:uppercase; border-bottom:1px solid {PL.hairline_dk};">
                  RS
                </th>
                <th style="padding:8px 10px; text-align:right;
                           font-family:Inter,Arial,sans-serif; font-size:9px;
                           font-weight:500; letter-spacing:0.12em; color:{PL.mute};
                           text-transform:uppercase; border-bottom:1px solid {PL.hairline_dk};">
                  52wH
                </th>
                <th style="padding:8px 10px; text-align:right;
                           font-family:Inter,Arial,sans-serif; font-size:9px;
                           font-weight:500; letter-spacing:0.12em; color:{PL.mute};
                           text-transform:uppercase; border-bottom:1px solid {PL.hairline_dk};">
                  1Y
                </th>
                <th style="padding:8px 10px; text-align:right;
                           font-family:Inter,Arial,sans-serif; font-size:9px;
                           font-weight:500; letter-spacing:0.12em; color:{PL.mute};
                           text-transform:uppercase; border-bottom:1px solid {PL.hairline_dk};">
                  COMPOSITE
                </th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
          {movement_html}
        </td></tr>

        <!-- Footer -->
        <tr><td style="padding:36px 40px 36px;">
          <a href="{dashboard_url}"
             style="display:inline-block; padding:14px 24px; border:1px solid {PL.bone};
                    border-radius:999px; color:{PL.bone}; text-decoration:none;
                    font-family:Inter,Arial,sans-serif; font-size:13px; letter-spacing:0.04em;">
            Open dashboard&nbsp;&nbsp;→
          </a>
          <div style="margin-top:32px; font-family:Inter,Arial,sans-serif;
                      color:{PL.mute}; font-size:10px; letter-spacing:0.06em;
                      line-height:1.5;">
            Methodology: Mark Minervini Stage 2 trend-template (8 binary gates)
            as the filter; composite of IBD-style RS rating + 52-week-high
            proximity + 1-year return as the ranker. Independent computation,
            no third-party feeds. Local-currency returns to neutralise FX noise.
          </div>
        </td></tr>

      </table>
    </td>
  </tr>
</table>
</body></html>"""


def send_digest(asof: dt.date | None = None) -> None:
    load_dotenv()
    snap = _latest_snapshot()
    prev = _previous_snapshot()

    dashboard_url = os.environ.get("DASHBOARD_URL", DEFAULT_DASH)
    html = render_digest_html(snap, prev, dashboard_url)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "digest_latest.html").write_text(html)

    if os.environ.get("DIGEST_DRY_RUN") == "1":
        print(f"[DRY] digest written -> {(OUTPUTS / 'digest_latest.html').relative_to(ROOT)}")
        return

    pw = os.environ.get("GMAIL_APP_PASSWORD")
    if not pw:
        print("[skip] GMAIL_APP_PASSWORD not set, no email sent")
        return

    sender = os.environ.get("DIGEST_FROM", DEFAULT_FROM)
    recipient = os.environ.get("DIGEST_TO", DEFAULT_TO)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Momentum Power · {snap['asof']} · Top 10"
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))

    pw_clean = pw.replace(" ", "")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(sender, pw_clean)
        s.sendmail(sender, [recipient], msg.as_string())
    print(f"digest sent -> {recipient}")


def main() -> None:
    send_digest()


if __name__ == "__main__":
    main()
