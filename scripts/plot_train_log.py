#!/usr/bin/env python3
"""Parse FastWAM train.log and plot loss curves vs training step.

Requires: pip install matplotlib

Example:
  python /home/xuewenyao/code/FastWAM/scripts/plot_train_log.py \\
    --log /home/xuewenyao/code/FastWAM/train.log \\
    --output /home/xuewenyao/code/FastWAM/train_loss.png
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

_LOG = logging.getLogger(__name__)

# Matches: [train] epoch=0 step=10/22250 loss=1.6360 loss_action=1.1275 loss_video=0.5085 ...
_TRAIN_LINE = re.compile(
    r"\[train\]\s+epoch=\d+\s+step=(\d+)/\d+\s+"
    r"loss=([\d.]+)\s+loss_action=([\d.]+)\s+loss_video=([\d.]+)"
)


def _split_runs_by_step_reset(
    points: list[tuple[int, float, float, float]],
) -> list[list[tuple[int, float, float, float]]]:
    """Split into runs when step decreases (new training job in same log)."""
    if not points:
        return []
    runs: list[list[tuple[int, float, float, float]]] = []
    current: list[tuple[int, float, float, float]] = []
    prev_step = -1
    for step, loss, la, lv in points:
        if current and step < prev_step:
            runs.append(current)
            current = []
        current.append((step, loss, la, lv))
        prev_step = step
    if current:
        runs.append(current)
    return runs


def parse_train_log(log_path: Path) -> list[list[tuple[int, float, float, float]]]:
    """Read log file and return one list of runs; each run is (step, loss, ...)."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    raw: list[tuple[int, float, float, float]] = []
    for line in text.splitlines():
        m = _TRAIN_LINE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        loss = float(m.group(2))
        loss_action = float(m.group(3))
        loss_video = float(m.group(4))
        raw.append((step, loss, loss_action, loss_video))
    return _split_runs_by_step_reset(raw)


def _annotate_final_loss_values(
    ax,
    x_last: float,
    loss: float,
    loss_action: float,
    loss_video: float,
    series_colors: tuple,
) -> None:
    """Draw numeric labels near the last point of each series (vertical offset reduces overlap)."""
    c0, c1, c2 = series_colors
    entries = (
        (loss, "loss", c0, 14.0),
        (loss_action, "loss_action", c1, 0.0),
        (loss_video, "loss_video", c2, -14.0),
    )
    for value, name, color, dy_pts in entries:
        ax.annotate(
            f"{name}={value:.4f}",
            xy=(x_last, value),
            xytext=(8, dy_pts),
            textcoords="offset points",
            fontsize=8,
            color=color,
            ha="left",
            va="center",
        )


def plot_runs(
    runs: list[list[tuple[int, float, float, float]]],
    output_path: Path,
    merge_runs: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    if merge_runs or len(runs) <= 1:
        step_offset = 0
        all_steps: list[int] = []
        all_loss: list[float] = []
        all_la: list[float] = []
        all_lv: list[float] = []
        for run in runs:
            if not run:
                continue
            base = step_offset
            for step, loss, la, lv in run:
                all_steps.append(base + step)
                all_loss.append(loss)
                all_la.append(la)
                all_lv.append(lv)
            step_offset += run[-1][0]
        ax.plot(all_steps, all_loss, label="loss", color=colors[0], linewidth=1.2)
        ax.plot(all_steps, all_la, label="loss_action", color=colors[1], linewidth=1.2)
        ax.plot(all_steps, all_lv, label="loss_video", color=colors[2], linewidth=1.2)
        if all_steps:
            _annotate_final_loss_values(
                ax,
                float(all_steps[-1]),
                all_loss[-1],
                all_la[-1],
                all_lv[-1],
                (colors[0], colors[1], colors[2]),
            )
    else:
        last_run: list[tuple[int, float, float, float]] | None = None
        last_run_color = colors[0]
        for i, run in enumerate(runs):
            if not run:
                continue
            steps = [p[0] for p in run]
            c = colors[i % len(colors)]
            ax.plot(
                steps,
                [p[1] for p in run],
                label=f"run{i + 1} loss",
                color=c,
                linestyle="-",
                linewidth=1.0,
            )
            ax.plot(
                steps,
                [p[2] for p in run],
                label=f"run{i + 1} loss_action",
                color=c,
                linestyle="--",
                linewidth=1.0,
            )
            ax.plot(
                steps,
                [p[3] for p in run],
                label=f"run{i + 1} loss_video",
                color=c,
                linestyle=":",
                linewidth=1.0,
            )
            last_run, last_run_color = run, c
        if last_run:
            steps = [p[0] for p in last_run]
            tail = last_run[-1]
            _annotate_final_loss_values(
                ax,
                float(steps[-1]),
                tail[1],
                tail[2],
                tail[3],
                (last_run_color, last_run_color, last_run_color),
            )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss vs step")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("Wrote plot to %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    default_log = repo_root / "train.log"
    default_out = repo_root / "train_loss.png"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=default_log,
        help="Path to train.log (default: <repo>/train.log)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="Output image path (default: <repo>/train_loss.png)",
    )
    parser.add_argument(
        "--separate-runs",
        action="store_true",
        help="If log has multiple restarts, plot each run separately (same step axis).",
    )
    args = parser.parse_args()

    log_path = args.log.resolve()
    if not log_path.is_file():
        _LOG.error("Log file not found: %s", log_path)
        sys.exit(1)

    runs = parse_train_log(log_path)
    if not runs or all(not r for r in runs):
        _LOG.error("No [train] lines with loss fields found in %s", log_path)
        sys.exit(1)

    _LOG.info("Parsed %d run(s), total points=%d", len(runs), sum(len(r) for r in runs))
    plot_runs(
        runs,
        args.output.resolve(),
        merge_runs=not args.separate_runs,
    )


if __name__ == "__main__":
    main()
