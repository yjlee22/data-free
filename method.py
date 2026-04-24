import numpy as np
import matplotlib.pyplot as plt

# =========================
# Global font settings
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7
})

# =========================
# File paths
# =========================
files = [
    "./bloodmnist/seed_0/Dirichlet_0.1/FedAvg_convnextv2_nano_True_sgd_0.1.npy",
    "./bloodmnist/seed_1/Dirichlet_0.1/FedAvg_convnextv2_nano_True_sgd_0.1.npy",
    "./bloodmnist/seed_2/Dirichlet_0.1/FedAvg_convnextv2_nano_True_sgd_0.1.npy",
]

# =========================
# Column indices (확인 필요)
# =========================
DELTA_COL = 6
SHIFT_COL = 7
DISPERSION_COL = 8

# =========================
# Load data
# =========================
data_list = [np.load(f) for f in files]

# =========================
# Align length
# =========================
min_rounds = min(arr.shape[0] for arr in data_list)

# stack
delta_stack = np.stack([arr[:min_rounds, DELTA_COL] for arr in data_list])
shift_stack = np.stack([arr[:min_rounds, SHIFT_COL] for arr in data_list])
disp_stack = np.stack([arr[:min_rounds, DISPERSION_COL] for arr in data_list])

# =========================
# mean / std
# =========================


def get_mean_std(x):
    return x.mean(axis=0), x.std(axis=0)


delta_mean, delta_std = get_mean_std(delta_stack)
shift_mean, shift_std = get_mean_std(shift_stack)
disp_mean, disp_std = get_mean_std(disp_stack)

# =========================
# rounds
# =========================
rounds = np.arange(1, min_rounds + 1)
xticks = np.linspace(1, min_rounds, 6, dtype=int)

# =========================
# Plot (3 equal subplots)
# =========================
fig, axes = plt.subplots(3, 1, figsize=(7.2, 3.2), sharex=True)

plot_items = [
    ("Delta", delta_mean, delta_std),
    ("Shift", shift_mean, shift_std),
    ("Dispersion", disp_mean, disp_std),
]

for ax, (ylabel, mean, std) in zip(axes, plot_items):
    ax.plot(rounds, mean, linewidth=1.2)
    ax.fill_between(rounds, mean - std, mean + std, alpha=0.2)

    # Delta threshold line
    if ylabel == "Delta":
        ax.axhline(y=0.1, linestyle="--", linewidth=1.0)

    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_xticks(xticks)
    ax.tick_params(axis='both')
    ax.grid(True, alpha=0.3)

# x-label
axes[-1].set_xlabel("Global Round")

# y-axis alignment
fig.align_ylabels(axes)

plt.tight_layout()

# =========================
# Save
# =========================
plt.savefig("method.pdf", dpi=300, bbox_inches="tight")
plt.close()
