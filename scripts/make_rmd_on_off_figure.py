import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "figure.dpi": 300,
})

df = pd.read_csv("results/05_analysis/paper_tables/rmd_on_vs_off_common5.csv")
subsets = ["all", "drop T1", "drop T1ce", "drop T2", "drop FLAIR"]
regions = [("dice_wt", "WT Dice"), ("dice_tc", "TC Dice"), ("dice_et", "ET Dice")]
colors = {"RMD ON": "#2a78d6", "RMD OFF": "#c0392b"}
x = np.arange(len(subsets))
width = 0.36

fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), sharey=True)
for ax, (metric, title) in zip(axes, regions):
    on = df[df.condition == "RMD ON"].set_index("subset").reindex(subsets)
    off = df[df.condition == "RMD OFF"].set_index("subset").reindex(subsets)
    ax.bar(x - width / 2, on[metric], width, label="RMD ON", color=colors["RMD ON"])
    ax.bar(x + width / 2, off[metric], width, label="RMD OFF", color=colors["RMD OFF"])
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)

axes[0].set_ylabel("Mean Dice (126 test cases)")
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.04))
fig.suptitle("RMD ON vs RMD OFF — five common subsets (legacy evaluation path)",
             y=0.96, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig("figures/rmd_on_vs_off_common5.pdf", bbox_inches="tight")
fig.savefig("figures/rmd_on_vs_off_common5.png", bbox_inches="tight")
print("Saved figures/rmd_on_vs_off_common5.pdf and .png")
