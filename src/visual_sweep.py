"""Hyperparameter sweep: validation loss and answer accuracy, one point per trial.

Reads the per-run summary written by ``tune_qwen3.py`` (one JSON object per line,
fields such as ``lr``, ``epoch``, ``eval_loss``, ``exact_match``) and draws the
metric against one swept axis, ordered by that axis.

Use ``--where`` to keep a comparable slice, for example only the runs at a fixed
number of epochs:

Example:
    python src/visual_sweep.py --axis lr --where epoch=1 \\
        --out figures/lr_sweep.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from visual_style import (
    BLUE,
    FIGURE_DPI,
    FIGURE_SIZE,
    GREEN,
    INK_2,
    binomial_ci,
    draw_title,
    format_value,
    percent_axis,
    read_jsonl,
    save,
    style_axis,
)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Draw one hyperparameter axis against validation metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        default="output/opt/runs.jsonl",
        help="Per-run summary written by tune_qwen3.py",
    )
    parser.add_argument("--axis", default="lr", help="Hyperparameter to sweep")
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep only runs whose field equals VALUE; repeat to add conditions",
    )
    parser.add_argument("--out", default="figures/sweep.png", help="Output PNG")
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument(
        "--note", default="", help="Small caption printed under the panels"
    )
    return parser.parse_args()


def matches(run, condition):
    """True when one field of a run equals the ``key=value`` condition."""
    key, _, wanted = condition.partition("=")
    if key not in run:
        return False
    actual = run[key]
    try:
        return math.isclose(float(actual), float(wanted), rel_tol=1e-9)
    except (TypeError, ValueError):
        return str(actual) == wanted


def validation_samples(runs_path, run_name):
    """Reads the validation-set size of a run from its metrics.json, if present."""
    metrics = Path(runs_path).parent / "runs" / run_name / "metrics.json"
    if not metrics.exists():
        return None
    with open(metrics, encoding="utf-8") as handle:
        return json.load(handle).get("eval", {}).get("n")


def collect(runs_path, axis, conditions):
    """Returns the selected runs, sorted by the swept axis."""
    if not Path(runs_path).exists():
        sys.exit(f"No run summary at {runs_path}; run tune_qwen3.py or pass --runs")
    runs = [run for run in read_jsonl(runs_path) if axis in run]
    for condition in conditions:
        runs = [run for run in runs if matches(run, condition)]
    if not runs:
        sys.exit(f"No runs in {runs_path} match {conditions or 'no filter'}")

    for run in runs:
        run["_n"] = validation_samples(runs_path, run["run"])
    return sorted(runs, key=lambda run: float(run[axis]))


def plot(runs, axis, out_path, title, note):
    """Draws the loss curve and the accuracy bars."""
    labels = [format_value(run[axis]) for run in runs]
    positions = list(range(len(runs)))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.16, wspace=0.24)

    losses = [run["eval_loss"] for run in runs]
    ax_loss.plot(positions, losses, color=BLUE, marker="o", markersize=6, linewidth=1.8)
    for position, loss in zip(positions, losses):
        ax_loss.annotate(
            f"{loss:.3f}",
            xy=(position, loss),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax_loss.margins(y=0.25)
    ax_loss.set_xticks(positions)
    ax_loss.set_xticklabels(labels)
    ax_loss.set_xlabel(axis)
    ax_loss.set_ylabel("Validation loss")
    ax_loss.set_title("Validation loss (lower is better)")
    style_axis(ax_loss)

    accuracies = [run["exact_match"] for run in runs]
    errors = [
        binomial_ci(run["exact_match"], run["_n"]) if run["_n"] else 0.0 for run in runs
    ]
    bars = ax_acc.bar(
        positions,
        accuracies,
        width=0.5,
        color=GREEN,
        yerr=errors if any(errors) else None,
        capsize=5,
        error_kw=dict(ecolor=INK_2, linewidth=1.1),
    )
    for bar, accuracy in zip(bars, accuracies):
        ax_acc.annotate(
            f"{accuracy:.0%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK_2,
        )
    ax_acc.set_xticks(positions)
    ax_acc.set_xticklabels(labels)
    ax_acc.set_ylim(0, min(1.0, max(accuracies) * 1.4 + 0.05))
    percent_axis(ax_acc)
    ax_acc.set_xlabel(axis)
    ax_acc.set_ylabel("Exact-match accuracy")
    ax_acc.set_title("Answer accuracy (teacher-forced proxy)")
    style_axis(ax_acc)

    draw_title(fig, title, note)
    save(fig, out_path)


def main():
    args = parse_args()
    runs = collect(args.runs, args.axis, args.where)
    if len({float(run[args.axis]) for run in runs}) < 2:
        sys.exit(f"Need at least two distinct {args.axis} values to plot a sweep")

    title = args.title or f"{args.axis} sweep (Qwen3-0.6B LoRA)"
    values = [float(run[args.axis]) for run in runs]
    if len(set(values)) < len(values):
        print(
            "  warn: repeated axis values; add --where to isolate one configuration"
        )
    plot(runs, args.axis, args.out, title, args.note)

    for run in runs:
        size = f"n={run['_n']}" if run["_n"] else "n=?"
        print(
            f"  {run['run']:<28s} {args.axis}={format_value(run[args.axis]):<7s} "
            f"loss {run['eval_loss']:.4f} | exact {run['exact_match']:.1%} ({size})"
        )


if __name__ == "__main__":
    main()
