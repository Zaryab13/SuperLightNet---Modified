"""Plot MACAF v3 training, validation, and selected-checkpoint test metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = ROOT / "results" / "macaf_experiments" / "base" / "v3"
TRAIN_LOG = V3_ROOT / "training" / "macaf_v3_training_log.csv"
TEST_SUMMARY = V3_ROOT / "sweep" / "test_all_macaf_v3_sweep.json"
OUTPUT_STEM = V3_ROOT / "training" / "macaf_v3_training_validation_test_curves"


def load_training_log() -> dict[str, np.ndarray]:
    with TRAIN_LOG.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training log is empty: {TRAIN_LOG}")
    columns = rows[0].keys()
    return {name: np.asarray([float(row[name]) for row in rows]) for name in columns}


def main() -> None:
    data = load_training_log()
    summary = json.loads(TEST_SUMMARY.read_text(encoding="utf-8"))
    epochs = data["epoch"].astype(int)
    val_dice = np.vstack([data["val_dice_wt"], data["val_dice_tc"], data["val_dice_et"]])
    val_mean = val_dice.mean(axis=0)
    best_index = int(np.argmax(val_mean))
    best_epoch = int(epochs[best_index])
    test_dice = np.asarray([
        summary["statistics"]["dice_wt"]["mean"],
        summary["statistics"]["dice_tc"]["mean"],
        summary["statistics"]["dice_et"]["mean"],
    ])
    test_mean = float(test_dice.mean())

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), constrained_layout=True)

    axes[0].plot(epochs, data["train_loss"], color="#2563a6", linewidth=1.6, label="Training loss")
    axes[0].plot(epochs, data["val_loss"], color="#d17a22", linewidth=1.4, label="Validation loss")
    axes[0].set(title="Optimization", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(frameon=False)

    colors = ("#2563a6", "#d17a22", "#16806a")
    for values, label, color in zip(val_dice, ("WT", "TC", "ET"), colors):
        axes[1].plot(epochs, values, linewidth=1.25, color=color, label=f"Validation {label}")
    axes[1].plot(epochs, val_mean, color="#202020", linewidth=1.8, label="Validation mean")
    axes[1].scatter([best_epoch], [test_mean], color="#8b3f8f", marker="D", s=34,
                    label="Test mean (selected checkpoint)", zorder=5)
    axes[1].set(title="Region Dice", xlabel="Epoch", ylabel="Dice", ylim=(0.0, 1.0))
    axes[1].legend(frameon=False, fontsize=7.5)

    val_error = 1.0 - val_mean
    axes[2].plot(epochs, val_error, color="#202020", linewidth=1.8, label="Validation Dice error")
    axes[2].scatter([best_epoch], [1.0 - test_mean], color="#8b3f8f", marker="D", s=34,
                    label="Test Dice error (selected checkpoint)", zorder=5)
    axes[2].set(title="Segmentation error", xlabel="Epoch", ylabel="1 - mean Dice", ylim=(0.0, 1.0))
    axes[2].legend(frameon=False, fontsize=7.5)

    for axis in axes:
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(int(epochs.min()), int(epochs.max()))

    fig.suptitle("MACAF v3 training and evaluation curves", fontsize=12)
    fig.text(0.5, -0.02, "Training log available for epochs 17-100; test evaluated once at the selected checkpoint.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Best logged epoch: {best_epoch}")
    print(f"Validation mean Dice: {val_mean[best_index]:.15f}")
    print(f"Test mean Dice: {test_mean:.15f}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
