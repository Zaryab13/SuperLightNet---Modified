import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "figure.dpi": 300,
})
COLORS = {"WT": "#2a78d6", "TC": "#e0a02a", "ET": "#c0392b"}

df = pd.read_csv("results/paper_tables/fig_data_recovery_summary.csv")
panelA = df[df.subset == "all"].set_index("condition").reindex(
    ["eval_fixed_preKD", "eval_fixed_postKD"])
panelB = df[df.subset == "t1_t2_flair"].set_index("condition").reindex(
    ["original_buggy", "eval_fixed_preKD", "eval_fixed_postKD"])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 2.8))  # IEEE 2-col full width

def grouped_bars(ax, data, labels, title):
    regions = ["dice_wt", "dice_tc", "dice_et"]
    names = ["WT", "TC", "ET"]
    n = len(data)
    width = 0.8 / len(regions)
    x = range(n)
    for i, (col, name) in enumerate(zip(regions, names)):
        ax.bar([p + i * width for p in x], data[col].values, width,
               label=name, color=COLORS[name])
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Dice")
    ax.set_title(title, fontsize=8.5)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)

grouped_bars(axA, panelA, ["Baseline", "After\ndistillation"],
             "All 4 modalities present")
grouped_bars(axB, panelB, ["Original\n(buggy eval)", "Eval\ncorrected", "After\ndistillation"],
             "Missing T1ce (T1+T2+FLAIR)")
handles, labs = axB.get_legend_handles_labels()
fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
fig.savefig("figures/missing_modality_recovery.pdf", bbox_inches="tight")
fig.savefig("figures/missing_modality_recovery.png", bbox_inches="tight")
print("Saved figures/missing_modality_recovery.pdf and .png")
