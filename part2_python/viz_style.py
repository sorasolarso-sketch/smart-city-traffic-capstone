"""Shared plotting style for every figure in the capstone.

Centralising the palette here means all figures across Parts 2 and 3 read
as one system rather than as a pile of separately-styled charts.

Palette notes
-------------
The categorical hues are assigned in a fixed order and never cycled. The
blue/orange pair used for the two-series figures was checked for
colour-vision-deficiency separation before use: worst adjacent pair
Delta E 24.7 under protanopia and 33.6 under normal vision, comfortably
above the 8 and 15 floors respectively.

Sequential encodings (heatmaps) use a single blue hue running light to
dark, never a rainbow, so that magnitude reads monotonically.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Surfaces and ink ------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#7a7973"
GRID = "#e5e4e0"

# --- Categorical hues, in fixed assignment order ---------------------------
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# --- Single-hue sequential ramp, light to dark -----------------------------
SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("capstone_blue", SEQUENTIAL_STEPS)

# --- Status colours, reserved and never reused as a series -----------------
STATUS = {
    "good": "#1a7f4b",
    "warning": "#b07400",
    "serious": "#c1461f",
    "critical": "#a51f1f",
}

FIGSIZE_WIDE = (11, 5.5)
FIGSIZE_SQUARE = (8, 6.5)
DPI = 150


def apply_style() -> None:
    """Install the project's matplotlib defaults."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": DPI,

        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": TEXT_PRIMARY,

        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.labelsize": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlelocation": "left",
        # Generous pad so that a subtitle can sit between the title and the
        # axes without either colliding.
        "axes.titlepad": 30,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),

        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",

        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": TEXT_SECONDARY,

        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
    })


def add_subtitle(ax, text: str) -> None:
    """Place an explanatory subtitle directly under the axes title."""
    ax.text(
        0.0, 1.012, text, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=9.5, color=TEXT_SECONDARY,
    )


def add_source_note(fig, text: str) -> None:
    """Small provenance note in the lower-left of the figure."""
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=8, color=TEXT_MUTED)
