"""Shared style and helpers for the visual_*.py chart scripts.

Every chart in this project uses the same recessive look: no top/right spines, a
light horizontal grid, muted axis text and one blue accent colour. Keeping the
palette, the axis styling and the small statistics helpers here means the three
chart scripts only describe what they draw.

Matplotlib is pinned to the Agg backend so the scripts also run on a machine
without a display.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib import ticker  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1f9e58"
PURPLE = "#8b5cf6"
RED = "#e34948"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
SPINE = "#c3c2b7"

PALETTE = (BLUE, ORANGE, GREEN, PURPLE, RED)

FIGURE_SIZE = (12, 5.6)
FIGURE_DPI = 120


def read_jsonl(path):
    """Returns the non-empty JSON objects of a JSONL file, in file order."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def style_axis(axis):
    """Applies the shared chart chrome: light y-grid, muted labels, open frame."""
    axis.grid(True, axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(SPINE)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.xaxis.label.set_color(INK_2)
    axis.yaxis.label.set_color(INK_2)
    axis.title.set_color(INK)


def percent_axis(axis):
    """Formats an axis in whole percentages (0.29 -> 29%)."""
    axis.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value:.0%}")
    )


def draw_title(fig, title, note=""):
    """Draws the figure title and an optional one-line caption underneath."""
    fig.suptitle(title, fontsize=13, color=INK)
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=8, color=MUTED)


def save(fig, out_path):
    """Writes the PNG (creating parent directories) and releases the figure."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def binomial_ci(correct, total, z=1.96):
    """Half-width of the normal-approximation 95% CI of a proportion."""
    if not total:
        return 0.0
    rate = correct / total
    return z * math.sqrt(max(rate * (1 - rate), 0.0) / total)


def format_value(value):
    """Compact axis label: 1e-4 for a small learning rate, 8 for a LoRA rank."""
    if isinstance(value, str):
        return value
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:g}"
