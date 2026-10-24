"""Training curves: loss (train and validation) plus the learning-rate schedule.

Two log shapes are understood:

* ``--state``: a Trainer ``trainer_state.json`` such as the ones inside the
  checkpoints written by ``train_qwen3.py``. It carries the train loss, the
  evaluation loss and the learning-rate schedule.
* ``--log``: the JSONL run log written by ``tune_qwen3.py``. Repeat the flag to
  overlay several runs, which is how a sweep is compared.

Examples:
    python src/visual_training.py \\
        --state output/law-qwen3/checkpoint-1119/trainer_state.json
    python src/visual_training.py \\
        --log output/opt/runs/mini-lr2e-4-e3/train_log.jsonl \\
        --log output/opt/runs/medium-lr3e-4-e1/train_log.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from visual_style import (
    FIGURE_DPI,
    FIGURE_SIZE,
    INK_2,
    PALETTE,
    draw_title,
    save,
    style_axis,
)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Draw training loss and learning-rate curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--state",
        help="trainer_state.json of a single training run",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        metavar="PATH",
        help="train_log.jsonl from tune_qwen3.py; repeat to overlay runs",
    )
    parser.add_argument("--out", default="figures/training.png", help="Output PNG")
    parser.add_argument(
        "--title", default="Qwen3-0.6B LoRA training curves", help="Figure title"
    )
    parser.add_argument(
        "--note", default="", help="Small caption printed under the panels"
    )
    return parser.parse_args()


def load_state(path):
    """Loads one trainer_state.json into a run record."""
    if not Path(path).exists():
        sys.exit(f"No such file: {path}")
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)

    train, evals = [], []
    for record in state.get("log_history", []):
        epoch = record.get("epoch")
        if epoch is None:
            continue
        if "loss" in record:
            train.append((epoch, record["loss"], record.get("learning_rate")))
        if "eval_loss" in record:
            evals.append((epoch, record["eval_loss"]))
    if not train:
        sys.exit(f"No training-loss records in {path}")

    label = f"{Path(path).parent.parent.name}/{Path(path).parent.name}"
    return {"label": label, "train": train, "evals": evals}


def load_log(path):
    """Loads one tune_qwen3.py JSONL log into a run record."""
    if not Path(path).exists():
        sys.exit(f"No such file: {path}")
    train = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "loss" in record and "epoch" in record:
                train.append(
                    (record["epoch"], record["loss"], record.get("learning_rate"))
                )
    if not train:
        sys.exit(f"No training-loss records in {path}")
    return {"label": Path(path).parent.name, "train": train, "evals": []}


def plot_runs(runs, out_path, title, note):
    """Draws the loss panel and the learning-rate panel."""
    fig, (ax_loss, ax_lr) = plt.subplots(1, 2, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.16, wspace=0.22)

    has_lr = False
    for index, run in enumerate(runs):
        color = PALETTE[index % len(PALETTE)]
        epochs = [point[0] for point in run["train"]]
        losses = [point[1] for point in run["train"]]
        ax_loss.plot(
            epochs,
            losses,
            color=color,
            linewidth=1.4,
            marker="x",
            markersize=4,
            label=run["label"],
        )
        if run["evals"]:
            eval_epochs = [point[0] for point in run["evals"]]
            eval_losses = [point[1] for point in run["evals"]]
            ax_loss.plot(
                eval_epochs,
                eval_losses,
                color=color,
                linewidth=1.4,
                linestyle="--",
                marker="*",
                markersize=8,
                label=f"{run['label']} (val)",
            )

        lr_points = [point for point in run["train"] if point[2] is not None]
        if lr_points:
            has_lr = True
            ax_lr.plot(
                [point[0] for point in lr_points],
                [point[2] for point in lr_points],
                color=color,
                linewidth=1.8,
                label=run["label"],
            )

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    style_axis(ax_loss)
    if len(runs) > 1 or runs[0]["evals"]:
        ax_loss.legend(
            loc="upper right", frameon=True, framealpha=0.9, edgecolor="none"
        )

    if has_lr:
        ax_lr.set_title("Learning rate")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.yaxis.set_major_formatter(
            lambda value, _: f"{value:.1e}".replace("e-0", "e-")
        )
    else:
        ax_lr.set_visible(False)
    style_axis(ax_lr)

    if len(runs) == 1:
        last = runs[0]["evals"][-1] if runs[0]["evals"] else runs[0]["train"][-1]
        ax_loss.annotate(
            f"{last[1]:.4f}",
            xy=(last[0], last[1]),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=INK_2,
        )

    draw_title(fig, title, note)
    save(fig, out_path)


def main():
    args = parse_args()
    if bool(args.state) == bool(args.log):
        sys.exit("Pass either --state or one or more --log, not both")

    if args.state:
        runs = [load_state(args.state)]
    else:
        runs = [load_log(path) for path in args.log]

    plot_runs(runs, args.out, args.title, args.note)
    for run in runs:
        final = run["evals"][-1][1] if run["evals"] else run["train"][-1][1]
        print(
            f"  {run['label']:<32s} {len(run['train']):3d} train points, "
            f"{len(run['evals']):2d} eval points, last loss {final:.4f}"
        )


if __name__ == "__main__":
    main()
