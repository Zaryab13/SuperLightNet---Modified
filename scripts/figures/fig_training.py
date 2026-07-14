#!/usr/bin/env python3
"""Create self-distillation v3 loss and validation training curves.

CAPTION:
Post-training self-distillation dynamics of the headline model
(self_distill_v3). Left: training loss and its segmentation, distillation,
and feature components. Right: validation Dice for the full-modality and
drop-T1ce conditions; the shaded band spans three seeds (42, 43, 44) and the
dashed line marks the selected checkpoint. Curves are shown for the
self-distillation stage only.
"""

from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_figures"
SOURCES = {
    42: "main:results/03_self_kd/self_distill_v3/self_distill_v3_training_log.csv",
    43: "main:results/03_self_kd/self_distill_v3_seed43/self_distill_v3_seed43_training_log.csv",
    44: "main:results/03_self_kd/self_distill_v3_seed44/self_distill_v3_seed44_training_log.csv",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})


def read_git_csv(ref: str):
    completed = subprocess.run(
        ["git", "show", ref], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    reader = csv.DictReader(io.StringIO(completed.stdout))
    rows = list(reader)
    print(f"SOURCE {ref}")
    print(f"COLUMNS {ref}: {reader.fieldnames}")
    return rows, tuple(reader.fieldnames or ())


def numeric(row, column):
    value = row.get(column, "")
    if value is None or not str(value).strip():
        return None
    return float(value)


def validation_series(rows, column):
    return {
        int(row["epoch"]): numeric(row, column)
        for row in rows if numeric(row, column) is not None
    }


def selected_epoch(rows):
    eligible = []
    for row in rows:
        score = numeric(row, "drop_t1ce_tc_et")
        flag = str(row.get("full_constraint_met", "")).strip().lower()
        if score is not None and flag in ("true", "1"):
            eligible.append((score, int(row["epoch"])))
    if not eligible:
        eligible = [
            (numeric(row, "drop_t1ce_tc_et"), int(row["epoch"]))
            for row in rows if numeric(row, "drop_t1ce_tc_et") is not None
        ]
    if not eligible:
        raise RuntimeError("No validated epoch found in seed-42 training log")
    return max(eligible)[1]


def draw(loss_rows, all_runs, columns, wide):
    if wide:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.2))
    ax_loss, ax_val = axes

    loss_styles = {
        "train_loss": ("Total", "#332288", "-", 1.7),
        "seg_loss": ("Segmentation", "#0077BB", "--", 1.3),
        "kd_loss": ("KD", "#EE7733", "-.", 1.3),
        "feature_loss": ("Feature", "#009988", ":", 1.6),
    }
    plotted_values = []
    for column, (label, color, linestyle, linewidth) in loss_styles.items():
        if column not in columns[42]:
            print(f"SKIP missing component column in seed 42: {column}")
            continue
        points = [
            (int(row["epoch"]), numeric(row, column)) for row in loss_rows
            if numeric(row, column) is not None and numeric(row, column) > 0
        ]
        if not points:
            print(f"SKIP empty component column in seed 42: {column}")
            continue
        epochs, values = zip(*points)
        plotted_values.extend(values)
        ax_loss.plot(epochs, values, color=color, linestyle=linestyle,
                     linewidth=linewidth, label=label)
    if plotted_values and max(plotted_values) / min(plotted_values) > 25:
        ax_loss.set_yscale("log")
        print("LOSS_SCALE logarithmic")
    else:
        print("LOSS_SCALE linear")
    ax_loss.set_title("Training losses (seed 42)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(axis="both", linewidth=0.35, alpha=0.35)
    ax_loss.legend(frameon=False, ncol=2 if not wide else 1)

    metric_styles = {
        "all_mean": ("All modalities", "#0077BB", "-"),
        "drop_t1ce_tc_et": ("Missing T1ce", "#EE7733", "--"),
    }
    for column, (label, color, linestyle) in metric_styles.items():
        series = {seed: validation_series(rows, column) for seed, rows in all_runs.items()}
        common_epochs = sorted(set.intersection(*(set(values) for values in series.values())))
        if not common_epochs:
            print(f"SKIP no common validation epochs for {column}")
            continue
        matrix = np.asarray([
            [series[seed][epoch] for epoch in common_epochs] for seed in sorted(series)
        ], dtype=float)
        ax_val.fill_between(common_epochs, matrix.min(axis=0), matrix.max(axis=0),
                            color=color, alpha=0.16, linewidth=0,
                            label=f"{label} (seed range)")
        seed42_index = sorted(series).index(42)
        ax_val.plot(common_epochs, matrix[seed42_index], color=color, linestyle=linestyle,
                    linewidth=1.8, label=f"{label} (seed 42)")
    best_epoch = selected_epoch(loss_rows)
    print(f"SELECTED_BEST_EPOCH seed42={best_epoch}")
    ax_val.axvline(best_epoch, color="#AA3377", linestyle=":", linewidth=1.1,
                   label=f"Selected epoch {best_epoch}")
    ax_val.set_title("Validation Dice across seeds")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Dice")
    ax_val.set_ylim(0.35, 0.9)
    ax_val.grid(axis="both", linewidth=0.35, alpha=0.35)
    ax_val.legend(frameon=False)
    fig.tight_layout()
    return fig


def main():
    names = [
        "fig_training_selfdistill.pdf", "fig_training_selfdistill.png",
        "fig_training_selfdistill_wide.pdf", "fig_training_selfdistill_wide.png",
    ]
    existing = [OUT / name for name in names if (OUT / name).exists()]
    if existing:
        print(f"REPLACING previously generated figure variants: {existing}")
    runs = {}
    columns = {}
    for seed, ref in SOURCES.items():
        runs[seed], columns[seed] = read_git_csv(ref)
    OUT.mkdir(parents=True, exist_ok=True)
    figure = draw(runs[42], runs, columns, wide=False)
    figure.savefig(OUT / "fig_training_selfdistill.pdf", bbox_inches="tight")
    figure.savefig(OUT / "fig_training_selfdistill.png", bbox_inches="tight", dpi=600)
    plt.close(figure)
    figure = draw(runs[42], runs, columns, wide=True)
    figure.savefig(OUT / "fig_training_selfdistill_wide.pdf", bbox_inches="tight")
    figure.savefig(OUT / "fig_training_selfdistill_wide.png", bbox_inches="tight", dpi=600)
    plt.close(figure)
    print("CONFIRMED: only self_distill_v3 seed 42/43/44 logs were read.")
    print(f"Saved {OUT / 'fig_training_selfdistill.pdf'} and .png")
    print(f"Saved {OUT / 'fig_training_selfdistill_wide.pdf'} and .png")
    print("No existing files were modified.")


if __name__ == "__main__":
    main()
