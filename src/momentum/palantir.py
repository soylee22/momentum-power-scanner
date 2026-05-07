"""Palantir style tokens for Python output (matplotlib, plotly, python-pptx, openpyxl).

Source: ~/SecondBrain/wiki/references/palantir-style-guide.md
Generated: 2026-05-05

Usage:

    from palantir_tokens import PL, mpl_dark_rcparams, plotly_dark_template

    plt.rcParams.update(mpl_dark_rcparams())
    fig.set_facecolor(PL.ink)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _Palette:
    # Core duotone
    ink: str = "#0D1014"
    stage: str = "#E2E2E0"
    paper: str = "#FFFFFF"
    bone: str = "#D4D4CC"
    mute: str = "#A8A8A4"
    graphite: str = "#1A1A1A"
    hairline_dk: str = "#3A3D42"
    hairline_lt: str = "#9A9A95"
    card_mid: str = "#C8C8C5"

    # Accents (use one per page)
    accent_green: str = "#10D070"   # slide-deck accent
    accent_lilac: str = "#B8A2D6"   # white-document accent

    # Geometry (px / pt)
    radius_card: int = 28
    radius_pill: int = 999
    hairline: float = 1.0
    safe_margin: int = 80

    # Type stack
    font_primary: str = "Söhne"
    font_fallback: str = "Inter"


PL = _Palette()


# ---------------------------------------------------------------------------
# RGB helpers (for python-pptx, openpyxl, etc.)
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def hex_to_pptx_rgb(hex_str: str):
    """Return a pptx.dml.color.RGBColor from a hex string."""
    from pptx.dml.color import RGBColor
    return RGBColor(*hex_to_rgb(hex_str))


# ---------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------

def mpl_dark_rcparams(font: str = "Inter") -> dict:
    """rcParams dict for the dark Palantir chart card."""
    return {
        "figure.facecolor": PL.ink,
        "axes.facecolor": PL.ink,
        "savefig.facecolor": PL.ink,
        "savefig.edgecolor": PL.ink,
        "axes.edgecolor": PL.ink,
        "axes.labelcolor": PL.mute,
        "xtick.color": PL.mute,
        "ytick.color": PL.mute,
        "text.color": PL.bone,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.grid": False,
        "font.family": font,
        "font.size": 11,
        "axes.titlesize": 22,
        "axes.titlelocation": "left",
        "axes.titleweight": "regular",
        "axes.titlepad": 24,
        "axes.titlecolor": PL.bone,
        "legend.frameon": False,
        "legend.labelcolor": PL.mute,
    }


def mpl_light_rcparams(font: str = "Inter") -> dict:
    """rcParams dict for the light-stage Palantir chart."""
    return {
        "figure.facecolor": PL.stage,
        "axes.facecolor": PL.stage,
        "savefig.facecolor": PL.stage,
        "savefig.edgecolor": PL.stage,
        "axes.edgecolor": PL.stage,
        "axes.labelcolor": PL.graphite,
        "xtick.color": PL.graphite,
        "ytick.color": PL.graphite,
        "text.color": PL.graphite,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.grid": False,
        "font.family": font,
        "font.size": 11,
        "axes.titlesize": 22,
        "axes.titlelocation": "left",
        "axes.titleweight": "regular",
        "axes.titlepad": 24,
        "axes.titlecolor": PL.graphite,
        "legend.frameon": False,
        "legend.labelcolor": PL.graphite,
    }


# Color sequences (use weight/fill, not hue, for categories — but if you must)
PL_SEQUENCE_DARK = [PL.bone, PL.mute, PL.accent_green]
PL_SEQUENCE_LIGHT = [PL.graphite, PL.hairline_lt, PL.accent_green]


# ---------------------------------------------------------------------------
# plotly
# ---------------------------------------------------------------------------

def plotly_dark_template() -> dict:
    """Plotly layout dict for the dark Palantir chart card."""
    return dict(
        paper_bgcolor=PL.ink,
        plot_bgcolor=PL.ink,
        font=dict(family="Inter", color=PL.bone, size=12),
        title=dict(font=dict(size=22, color=PL.bone), x=0.04, xanchor="left"),
        xaxis=dict(showgrid=False, zeroline=False, color=PL.mute,
                   linecolor=PL.hairline_dk, ticks="outside", tickcolor=PL.hairline_dk),
        yaxis=dict(showgrid=False, zeroline=False, color=PL.mute, linecolor=PL.hairline_dk),
        colorway=PL_SEQUENCE_DARK,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PL.mute, size=11)),
    )


def plotly_light_template() -> dict:
    return dict(
        paper_bgcolor=PL.stage,
        plot_bgcolor=PL.stage,
        font=dict(family="Inter", color=PL.graphite, size=12),
        title=dict(font=dict(size=22, color=PL.graphite), x=0.04, xanchor="left"),
        xaxis=dict(showgrid=False, zeroline=False, color=PL.graphite,
                   linecolor=PL.hairline_lt, ticks="outside", tickcolor=PL.hairline_lt),
        yaxis=dict(showgrid=False, zeroline=False, color=PL.graphite, linecolor=PL.hairline_lt),
        colorway=PL_SEQUENCE_LIGHT,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PL.graphite, size=11)),
    )


# ---------------------------------------------------------------------------
# Mermaid theme block (paste into mermaid markdown)
# ---------------------------------------------------------------------------

MERMAID_INIT_DARK = (
    "%%{init: {"
    "'theme':'base',"
    "'themeVariables':{"
    f"'background':'{PL.ink}',"
    f"'primaryColor':'{PL.ink}',"
    f"'primaryTextColor':'{PL.bone}',"
    f"'primaryBorderColor':'{PL.hairline_dk}',"
    f"'lineColor':'{PL.mute}',"
    f"'secondaryColor':'{PL.graphite}',"
    f"'tertiaryColor':'{PL.accent_green}',"
    "'fontFamily':'Inter'"
    "}}}%%"
)


if __name__ == "__main__":
    # Self-check
    print(f"Palantir tokens loaded: {PL}")
    print(f"\nMermaid init block:\n{MERMAID_INIT_DARK}")
