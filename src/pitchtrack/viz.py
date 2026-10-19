"""Matplotlib styling and pitch drawing shared by every figure."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, Rectangle

from pitchtrack.io import PITCH_LENGTH, PITCH_WIDTH

# Validated categorical order (light surface) and chart ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Ordinal blue ramp (steps 250 -> 650) for ordered categories such as speed zones.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
# Diverging: away (red) <- neutral gray -> home (blue).
DIVERGING = LinearSegmentedColormap.from_list(
    "away_home", ["#e34948", "#f0efec", "#2a78d6"])
HOME, AWAY = SERIES[0], SERIES[7]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.titleweight": "semibold",
        "axes.titlesize": 12, "axes.titlelocation": "left", "axes.titlepad": 12,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def draw_pitch(ax: plt.Axes, line_color: str = BASELINE, lw: float = 1.0) -> plt.Axes:
    """Draw a 105 x 68 m pitch centred on (0, 0)."""
    L, W = PITCH_LENGTH / 2, PITCH_WIDTH / 2
    kw = dict(fill=False, edgecolor=line_color, linewidth=lw, zorder=1)
    ax.add_patch(Rectangle((-L, -W), 2 * L, 2 * W, **kw))
    ax.plot([0, 0], [-W, W], color=line_color, lw=lw, zorder=1)
    ax.add_patch(Circle((0, 0), 9.15, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * L - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((s * L - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, **kw))
        ax.add_patch(Arc((s * (L - 11), 0), 18.3, 18.3,
                         theta1=127 if s > 0 else -53, theta2=233 if s > 0 else 53,
                         edgecolor=line_color, linewidth=lw, zorder=1))
    ax.set_xlim(-L - 2, L + 2)
    ax.set_ylim(-W - 2, W + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
