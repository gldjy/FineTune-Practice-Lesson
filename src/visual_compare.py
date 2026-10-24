"""Compare evaluated models on the same test split.

Reads the per-sample JSONL results written by ``test_qwen3.py`` and draws overall
accuracy with 95% confidence intervals plus the single/multiple-choice breakdown,
one group per ``--run LABEL=PATH``. Two models evaluated on different numbers of
samples are allowed; the caption reports each ``n``.

Example:
    python src/visual_compare.py \\
        --run "base=logs/test_results/test-20261024-052351.jsonl" \\
        --run "lr=3e-4=logs/test_results/test-20261024-052000.jsonl" \\
        --out figures/test_compare.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from visual_style import (
    BLUE,
    FIGURE_DPI,
    FIGURE_SIZE,
    INK_2,
    ORANGE,
    PALETTE,
    binomial_ci,
    draw_title,
    percent_axis,
    read_jsonl,
    save,
    style_axis,
)

SUBSETS = ("overall", "single", "multiple")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Draw accuracy per evaluated model and question type.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        required=True,
        metavar="LABEL=PATH",
        help="One evaluated model; repeat for each model in the figure",
    )
    parser.add_argument("--out", default="figures/compare.png", help="Output PNG")
    parser.add_argument(
        "--title", default="Accuracy on the same test split", help="Figure title"
    )
    parser.add_argument(
        "--note", default="", help="Small caption printed under the panels"
    )
    return parser.parse_args()


def accuracy_stats(samples):
    """Overall / single-choice / multi-choice accuracy of one result file."""
    subsets = {
        "overall": samples,
        "single": [row for row in samples if len(row["expected"]) <= 1],
        "multiple": [row for row in samples if len(row["expected"]) > 1],
    }
    stats = {}
    for name, rows in subsets.items():
        correct = sum(bool(row["correct"]) for row in rows)
        stats[name] = {
            "correct": correct,
            "n": len(rows),
            "acc": correct / len(rows) if rows else 0.0,
            "ci": binomial_ci(correct, len(rows)),
        }
    return stats


def collect(pairs):
    """Returns one entry per LABEL=PATH pair, skipping incomplete result files."""
    entries = []
    for pair in pairs:
        # rsplit: the label itself may contain '=', e.g. "lr=3e-4=PATH"
        label, _, path = pair.rpartition("=")
        if not label or not path:
            sys.exit(f"--run expects LABEL=PATH, got {pair!r}")
        if not Path(path).exists():
            sys.exit(f"No such result file: {path}")
        rows = read_jsonl(path)
        if len(rows) < 2:
            sys.exit(f"No samples in {path}")
        meta = rows[0].get("meta", {})
        samples = rows[1:]
        total = meta.get("samples")
        if total is not None and len(samples) != total:
            print(f"  skip incomplete {label} ({len(samples)}/{total})")
            continue

        entry = {"label": label, **accuracy_stats(samples)}
        entries.append(entry)
    if not entries:
        sys.exit("No complete result file to plot")
    return entries


def plot(entries, out_path, title, note):
    """Draws the overall bars and the question-type breakdown."""
    labels = [entry["label"] for entry in entries]
    positions = list(range(len(entries)))

    fig, (ax_all, ax_type) = plt.subplots(1, 2, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.18, wspace=0.22)

    overall = [entry["overall"]["acc"] for entry in entries]
    colors = [PALETTE[index % len(PALETTE)] for index in positions]
    bars = ax_all.bar(
        positions,
        overall,
        width=0.55,
        color=colors,
        yerr=[entry["overall"]["ci"] for entry in entries],
        capsize=5,
        error_kw=dict(ecolor=INK_2, linewidth=1.1),
    )
    for bar, entry in zip(bars, entries):
        ax_all.annotate(
            f"{entry['overall']['acc']:.1%}\nn={entry['overall']['n']}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_2,
        )
    ax_all.set_xticks(positions)
    ax_all.set_xticklabels(labels)
    ax_all.set_ylim(0, min(1.0, max(overall) * 1.5 + 0.05))
    percent_axis(ax_all)
    ax_all.set_ylabel("Accuracy")
    ax_all.set_title("Overall accuracy (95% CI)")
    style_axis(ax_all)

    width = 0.36
    single = [entry["single"]["acc"] for entry in entries]
    multiple = [entry["multiple"]["acc"] for entry in entries]
    bars_single = ax_type.bar(
        [position - width / 2 - 0.01 for position in positions],
        single,
        width,
        color=BLUE,
        label="Single-choice",
    )
    bars_multiple = ax_type.bar(
        [position + width / 2 + 0.01 for position in positions],
        multiple,
        width,
        color=ORANGE,
        label="Multiple-choice",
    )
    for bar in list(bars_single) + list(bars_multiple):
        ax_type.annotate(
            f"{bar.get_height():.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_2,
        )
    ax_type.set_xticks(positions)
    ax_type.set_xticklabels(labels)
    ax_type.set_ylim(0, min(1.0, max(single + multiple) * 1.4 + 0.05))
    percent_axis(ax_type)
    ax_type.set_ylabel("Accuracy")
    ax_type.set_title("By question type")
    ax_type.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    style_axis(ax_type)

    draw_title(fig, title, note)
    save(fig, out_path)


def main():
    args = parse_args()
    entries = collect(args.run)
    note = args.note or "; ".join(
        f"{entry['label']}: n={entry['overall']['n']}" for entry in entries
    )
    plot(entries, args.out, args.title, note)

    for entry in entries:
        print(
            f"  {entry['label']:<24s} overall {entry['overall']['acc']:.2%} "
            f"(n={entry['overall']['n']}) | single {entry['single']['acc']:.2%} "
            f"(n={entry['single']['n']}) | multiple {entry['multiple']['acc']:.2%} "
            f"(n={entry['multiple']['n']})"
        )


if __name__ == "__main__":
    main()
