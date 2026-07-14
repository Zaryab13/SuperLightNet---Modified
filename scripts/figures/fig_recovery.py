"""Plot the valid drop-T1ce baseline and self-distillation recovery.

CAPTION:
Recovery of tumour-core (TC) and enhancing-tumour (ET) Dice under the
drop-T1ce condition. The baseline is the correctly evaluated RMD backbone
without distillation; the second bar is the same architecture after
post-training self-distillation (mean over three seeds, error bar = 1 s.d.).
Distillation accounts for the entire recovery.
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

SOURCES = {
    "eval_fixed_preKD": "results/01_base_model/leakage_safe_fixed/test_t1_t2_flair_fixed.csv",
    "seed42": "results/03_self_kd/self_distill_v3_sweep/test_t1_t2_flair_v3.csv",
    "seed43": "results/03_self_kd/self_distill_v3_seed43_sweep/test_t1_t2_flair_v3_seed43.csv",
    "seed44": "results/03_self_kd/self_distill_v3_seed44_sweep/test_t1_t2_flair_v3_seed44.csv",
}


def read_committed_csv(branch: str, path: str) -> pd.DataFrame:
    proc = subprocess.run(
        ["git", "show", f"{branch}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return pd.read_csv(io.BytesIO(proc.stdout))


def bracket(ax, x0: float, x1: float, y: float, text: str) -> None:
    height = 0.025
    ax.plot([x0, x0, x1, x1], [y, y + height, y + height, y], color="0.15", lw=0.7)
    ax.text((x0 + x1) / 2, y + height + 0.012, text, ha="center", va="bottom", fontsize=8)


def draw(width: float, stem: str, values: dict[str, np.ndarray], errors: dict[str, np.ndarray]) -> None:
    height = 2.75 if width >= 6 else 2.65
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True)
    labels = (
        ["Baseline", "Self-distilled"]
        if width < 6
        else ["Baseline\n(no distillation)", "After self-\ndistillation"]
    )
    fills = ["#EECC66", "#AA4499"]
    hatches = ["..", "xx"]
    x = np.arange(2)

    for ax, metric, title in zip(axes, ("dice_tc", "dice_et"), ("Tumour core (TC)", "Enhancing tumour (ET)")):
        vals = values[metric]
        errs = errors[metric]
        bars = ax.bar(
            x,
            vals,
            width=0.66,
            color=fills,
            edgecolor="0.1",
            linewidth=0.6,
            hatch=hatches,
            yerr=errs,
            error_kw={"ecolor": "0.05", "capsize": 2.5, "elinewidth": 0.8, "capthick": 0.8},
        )
        for bar, val, err in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + err + 0.018, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        gain = vals[1] - vals[0]
        bracket_y = max(vals[0], vals[1] + errs[1]) + 0.105
        bracket(ax, 0, 1, bracket_y, f"distillation gain\n+{gain:.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="0.85", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Dice")
    fig.tight_layout(w_pad=0.7)

    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    replacing = [target for target in (pdf, png) if target.exists()]
    if replacing:
        print(f"REPLACING named figure outputs: {replacing}")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {pdf}")
    print(f"SAVED {png}")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = {name: read_committed_csv("main", path) for name, path in SOURCES.items()}
    for name, frame in frames.items():
        if len(frame) != 126:
            raise RuntimeError(f"{name}: expected 126 rows, found {len(frame)}")

    print("DROP-T1CE SOURCES AND MEANS (n=126 each)")
    for name, path in SOURCES.items():
        frame = frames[name]
        print(
            f"{name}: main:{path} | "
            f"WT={frame['dice_wt'].mean():.12f} "
            f"TC={frame['dice_tc'].mean():.12f} "
            f"ET={frame['dice_et'].mean():.12f}"
        )

    values: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    for metric in ("dice_tc", "dice_et"):
        seed_means = np.array([frames[name][metric].mean() for name in ("seed42", "seed43", "seed44")])
        values[metric] = np.array(
            [frames["eval_fixed_preKD"][metric].mean(), seed_means.mean()]
        )
        errors[metric] = np.array([0.0, seed_means.std(ddof=0)])
        print(
            f"POSTKD {metric}: seed_means={seed_means.tolist()} "
            f"mean={seed_means.mean():.12f} population_sd={seed_means.std(ddof=0):.12f}"
        )

    draw(3.4, "fig_recovery", values, errors)
    draw(7.0, "fig_recovery_wide", values, errors)
    print("No existing files were modified.")


if __name__ == "__main__":
    main()
