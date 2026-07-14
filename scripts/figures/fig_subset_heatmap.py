"""Create the three-model, 15-subset Dice heatmap from per-case sweep CSVs.

CAPTION:
Per-subset Dice across all fifteen modality combinations for the three
variants. Rows are grouped by number of available modalities and split into
T1ce-present and T1ce-absent blocks. Tumour-core and enhancing-tumour
performance drops sharply and specifically in the T1ce-absent block,
consistent with the information-theoretic role of the contrast-enhanced
sequence.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("results/paper_figures")
SEEDS = (42, 43, 44)
MODELS = ("Self-KD v3\n(3-seed mean)", "MACAF v1", "Clean Swin-KD")
METRICS = ("dice_et", "dice_tc", "dice_wt")
METRIC_TITLES = {"dice_et": "Enhancing tumour (ET)", "dice_tc": "Tumour core (TC)", "dice_wt": "Whole tumour (WT)"}

# Cardinality ascending; T1ce-present subsets precede T1ce-absent subsets within each group.
SUBSETS = (
    "t1ce", "t1", "t2", "flair",
    "t1_t1ce", "t1ce_t2", "t1ce_flair", "t1_t2", "t1_flair", "t2_flair",
    "t1_t1ce_t2", "t1_t1ce_flair", "t1ce_t2_flair", "t1_t2_flair",
    "t1_t1ce_t2_flair",
)

LABELS = {
    "t1ce": "T1ce", "t1": "T1", "t2": "T2", "flair": "FLAIR",
    "t1_t1ce": "T1, T1ce", "t1ce_t2": "T1ce, T2", "t1ce_flair": "T1ce, FLAIR",
    "t1_t2": "T1, T2", "t1_flair": "T1, FLAIR", "t2_flair": "T2, FLAIR",
    "t1_t1ce_t2": "T1, T1ce, T2", "t1_t1ce_flair": "T1, T1ce, FLAIR",
    "t1ce_t2_flair": "T1ce, T2, FLAIR", "t1_t2_flair": "T1, T2, FLAIR",
    "t1_t1ce_t2_flair": "T1, T1ce, T2, FLAIR",
}


def git_csv(branch: str, path: str) -> pd.DataFrame:
    proc = subprocess.run(
        ["git", "show", f"{branch}:{path}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    frame = pd.read_csv(io.BytesIO(proc.stdout))
    if len(frame) != 126:
        raise RuntimeError(f"Expected 126 cases in {branch}:{path}; found {len(frame)}")
    return frame


def self_kd_frame(subset: str, seed: int) -> pd.DataFrame:
    if seed == 42:
        path = f"results/03_self_kd/self_distill_v3_sweep/test_{subset}_v3.csv"
    else:
        path = f"results/03_self_kd/self_distill_v3_seed{seed}_sweep/test_{subset}_v3_seed{seed}.csv"
    return git_csv("main", path)


def macaf_frame(subset: str) -> pd.DataFrame:
    stem = "all" if subset == "t1_t1ce_t2_flair" else subset
    return git_csv("macaf", f"results/macaf_experiments/base/v1/sweep/test_{stem}_macaf_v1_sweep.csv")


def swin_frame(subset: str) -> pd.DataFrame:
    return git_csv("main", f"results/04_kd_clean_swin/swin_kd_clean_A_sweep/test_{subset}_swin_kd_cleanA.csv")


def load_matrix() -> dict[str, np.ndarray]:
    matrices = {metric: np.zeros((len(SUBSETS), len(MODELS)), dtype=float) for metric in METRICS}
    for row, subset in enumerate(SUBSETS):
        seed_frames = [self_kd_frame(subset, seed) for seed in SEEDS]
        macaf = macaf_frame(subset)
        swin = swin_frame(subset)
        for metric in METRICS:
            seed_means = [frame[metric].mean() for frame in seed_frames]
            matrices[metric][row] = (np.mean(seed_means), macaf[metric].mean(), swin[metric].mean())
    return matrices


def print_matrix(matrices: dict[str, np.ndarray]) -> None:
    columns = pd.MultiIndex.from_product(
        [["ET", "TC", "WT"], [name.replace("\n", " ") for name in MODELS]], names=["metric", "model"]
    )
    data = np.concatenate([matrices[metric] for metric in METRICS], axis=1)
    table = pd.DataFrame(data, index=[LABELS[s] for s in SUBSETS], columns=columns)
    print("FULL 15 x 3 metrics x 3 models MATRIX (means recomputed from n=126 per-case CSVs)")
    print(table.to_string(float_format=lambda value: f"{value:.6f}"))


def add_structure(ax: plt.Axes) -> None:
    # Thick rules divide modality cardinalities; dotted rules divide T1ce+/- within each cardinality.
    for boundary in (3.5, 9.5, 13.5):
        ax.axhline(boundary, color="white", lw=2.2)
        ax.axhline(boundary, color="0.15", lw=0.55)
    for boundary in (0.5, 6.5, 12.5):
        ax.axhline(boundary, color="white", lw=1.5, ls="--")
        ax.axhline(boundary, color="0.25", lw=0.45, ls="--")

    blocks = (
        (0, 0, "T1ce+"), (1, 3, "T1ce−"),
        (4, 6, "T1ce+"), (7, 9, "T1ce−"),
        (10, 12, "T1ce+"), (13, 13, "T1ce−"),
        (14, 14, "T1ce+"),
    )
    for start, end, label in blocks:
        center = (start + end) / 2
        ax.text(-0.48, center, label, transform=ax.get_yaxis_transform(), rotation=90,
                ha="center", va="center", fontsize=7.5, color="#334E68", clip_on=False)


def main() -> None:
    targets = [OUT_DIR / "fig_subset_heatmap.pdf", OUT_DIR / "fig_subset_heatmap.png"]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matrices = load_matrix()
    print_matrix(matrices)

    plt.rcParams.update(
        {
            "font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 13.0), constrained_layout=True)
    image = None
    for ax, metric in zip(axes, METRICS):
        matrix = matrices[metric]
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        ax.set_title(METRIC_TITLES[metric], fontsize=9, fontweight="bold")
        ax.set_xticks(np.arange(len(MODELS)), MODELS)
        ax.set_yticks(np.arange(len(SUBSETS)), [LABELS[s] for s in SUBSETS])
        ax.tick_params(length=0)
        add_structure(ax)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                color = "white" if value < 0.36 or value > 0.72 else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.65, pad=0.035)
    colorbar.set_label("Mean Dice")
    fig.savefig(targets[0], bbox_inches="tight")
    fig.savefig(targets[1], dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {targets[0]}")
    print(f"SAVED {targets[1]}")
    print("No existing files were modified.")


if __name__ == "__main__":
    main()
