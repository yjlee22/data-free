import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

DATASETS = ["dermamnist", "bloodmnist", "pathmnist"]
SEEDS = [0, 1, 2]
METHODS = ["FedAvg", "FedProx", "FedDyn", "SCAFFOLD", "FedSAM",
           "FedSpeed", "FedSMOO", "FedGamma", "FedLESAM", "FedWMSAM"]

MODEL_NAME = "convnextv2_nano"
TARGET_PRETRAIN = "True"
TARGET_OPT = "sgd"
TARGET_LR = "0.1"
THRESHOLD = "0.1"
PATIENCE_VALUES = [1, 2, 5, 10]
X_POS = np.arange(len(PATIENCE_VALUES))
SPLIT_RULE = "Dirichlet"
SPLIT_COEF = "0.1"
MODES = ["proposed", "validation"]

CURVE_STYLE = {
    ("dermamnist",  "proposed"):   dict(color="#F26522", marker="s", ls="-",  label="Skin lesion task (Proposed)"),
    ("dermamnist",  "validation"): dict(color="#F26522", marker="^", ls="--", label="Skin lesion task (Validation)"),
    ("bloodmnist",  "proposed"):   dict(color="#ED1C24", marker="s", ls="-",  label="Blood cell task (Proposed)"),
    ("bloodmnist",  "validation"): dict(color="#ED1C24", marker="^", ls="--", label="Blood cell task (Validation)"),
    ("pathmnist",   "proposed"):   dict(color="#007C97", marker="s", ls="-",  label="Colon pathology task (Proposed)"),
    ("pathmnist",   "validation"): dict(color="#007C97", marker="^", ls="--", label="Colon pathology task (Validation)"),
}

PLOT_ORDER = [
    ("dermamnist", "proposed"), ("dermamnist", "validation"),
    ("bloodmnist", "proposed"), ("bloodmnist", "validation"),
    ("pathmnist", "proposed"), ("pathmnist", "validation"),
]

X_OFFSETS = {
    ("dermamnist",  "proposed"): -0.06,
    ("dermamnist",  "validation"): -0.03,
    ("bloodmnist",  "proposed"):    0.00,
    ("bloodmnist",  "validation"):  0.03,
    ("pathmnist",   "proposed"):    0.06,
    ("pathmnist",   "validation"):  0.09,
}

def build_summary_path(ds, seed, sc, mode, method, patience):
    suffix = f"{MODEL_NAME}_{TARGET_PRETRAIN}_{TARGET_OPT}_{TARGET_LR}_{patience}_{THRESHOLD}"
    return os.path.join(
        ds,
        f"seed_{seed}",
        "summary",
        mode,
        f"{SPLIT_RULE}_{sc}",
        f"{method}_{suffix}.txt"
    )


def parse_summary(filepath):
    result = {}
    if not os.path.isfile(filepath):
        return result

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            m = re.match(
                r"Proposed Early Stop Round\s*:\s*(\d+|Not Triggered)", line)
            if m:
                result["proposed"] = int(m.group(1)) if m.group(
                    1) != "Not Triggered" else None

            m = re.match(
                r"Validation \(Loss\) Stop Round\s*:\s*(\d+|Not Triggered)", line)
            if m:
                result["val_loss"] = int(m.group(1)) if m.group(
                    1) != "Not Triggered" else None

            m = re.match(
                r"Validation \(Acc\) Stop Round\s*:\s*(\d+|Not Triggered)", line)
            if m:
                result["val_acc"] = int(m.group(1)) if m.group(
                    1) != "Not Triggered" else None

    return result


def get_stop_round(ds, seed, method, patience, mode):
    info = parse_summary(build_summary_path(
        ds, seed, SPLIT_COEF, mode, method, patience))

    if mode == "proposed":
        r = info.get("proposed")
        return float(r) if r is not None else np.nan

    rounds = [r for r in [info.get("val_loss"), info.get(
        "val_acc")] if r is not None]
    return float(min(rounds)) if rounds else np.nan

data = {}
for method in METHODS:
    for ds in DATASETS:
        for mode in MODES:
            means, stds = [], []
            for pat in PATIENCE_VALUES:
                seed_vals = [get_stop_round(
                    ds, seed, method, pat, mode) for seed in SEEDS]
                arr = np.array(seed_vals)

                if np.all(np.isnan(arr)):
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(float(np.nanmean(arr)))
                    stds.append(float(np.nanstd(arr)))

            data[(method, ds, mode)] = (np.array(means), np.array(stds))

with open("result.txt", "w", encoding="utf-8") as f:
    # 헤더 작성
    f.write(f"{'Method':<10} | {'Dataset':<12} | {'Mode':<10} | {'Patience':<8} | {'Mean':<8} | {'Std':<8}\n")
    f.write("-" * 65 + "\n")

    # 데이터 기록
    for method in METHODS:
        for ds in DATASETS:
            for mode in MODES:
                means, stds = data[(method, ds, mode)]
                for i, pat in enumerate(PATIENCE_VALUES):
                    mean_val = means[i]
                    std_val = stds[i]
                    f.write(
                        f"{method:<10} | {ds:<12} | {mode:<10} | {pat:<8} | {mean_val:<8.4f} | {std_val:<8.4f}\n")

print("Saved numeric data to result.txt")

fig, axs = plt.subplots(2, 5, figsize=(7.2, 3))

plt.subplots_adjust(
    wspace=0.20,
    hspace=0.28,
    bottom=0.20,
    top=0.93,
    left=0.06,
    right=0.99
)

axs = axs.flatten()

for i, method in enumerate(METHODS):
    ax = axs[i]
    ax.set_title(method, pad=2)

    for j, (ds, mode) in enumerate(PLOT_ORDER):
        means, stds = data[(method, ds, mode)]
        if np.all(np.isnan(means)):
            continue

        sty = CURVE_STYLE[(ds, mode)]
        x_adj = X_POS + X_OFFSETS[(ds, mode)]
        base_z = 10 + j

        ax.plot(
            x_adj,
            means,
            color=sty["color"],
            marker=sty["marker"],
            linestyle=sty["ls"],
            markersize=3,
            linewidth=1.0,
            label=sty["label"],
            zorder=base_z + 10
        )

        ax.fill_between(
            x_adj,
            means - stds,
            means + stds,
            color=sty["color"],
            alpha=0.15,
            linewidth=0,
            zorder=base_z
        )

    ax.set_xticks(X_POS)
    ax.set_xticklabels([str(p) for p in PATIENCE_VALUES])
    ax.tick_params(axis='both', which='major', pad=1.5)
    ax.set_xlim(X_POS[0] - 0.3, X_POS[-1] + 0.3)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5, color="grey")

    # y-ticks: integer only, around 3-4 ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    if i % 5 == 0:
        ax.set_ylabel("Stop round", labelpad=1)
    if i >= 5:
        ax.set_xlabel("Patience", labelpad=1)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles[:6],
    labels[:6],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.3,
    handlelength=1.5,
)

plt.savefig("result.pdf", format="pdf", dpi=300,
            bbox_inches='tight', pad_inches=0.02)
print("Saved result.pdf")
