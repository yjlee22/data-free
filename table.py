import os
import re
import numpy as np

DATASETS = ["dermamnist", "bloodmnist", "pathmnist"]
SEEDS = [0, 1, 2]

SPLIT_COEFS = ["1.0", "0.1", "0.01", "0.001"]
SC_DISPLAY = {
    "1.0": "0",
    "0.1": r"\text{-}1",
    "0.01": r"\text{-}2",
    "0.001": r"\text{-}3",
}

METHODS = [
    "FedAvg", "FedProx", "FedDyn", "SCAFFOLD", "FedSAM",
    "FedSpeed", "FedSMOO", "FedGamma", "FedLESAM", "FedWMSAM",
]

MODEL_NAME = "convnextv2_nano"
TARGET_PRETRAIN = "True"
TARGET_OPT = "sgd"
TARGET_LR = "0.1"
PATIENCE = "10"
THRESHOLD = "0.1"
FIXED_ROUND = 500

def build_summary_path(ds, seed, sc, mode, method):
    suffix = (
        f"{MODEL_NAME}_{TARGET_PRETRAIN}_{TARGET_OPT}_{TARGET_LR}"
        f"_{PATIENCE}_{THRESHOLD}"
    )
    return os.path.join(
        ds, f"seed_{seed}", "summary", mode,
        f"Dirichlet_{sc}", f"{method}_{suffix}.txt"
    )

def parse_summary(filepath):
    result = {}
    if not os.path.isfile(filepath):
        return result

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            m = re.match(r"Proposed Early Stop Round\s*:\s*(\d+|Not Triggered)", line)
            if m:
                result["proposed"] = int(m.group(1)) if m.group(1) != "Not Triggered" else None

    return result

def compute_proposed(ds, seed, sc, method):
    info = parse_summary(build_summary_path(ds, seed, sc, "proposed", method))
    p = info.get("proposed")
    return float(p) if p is not None else np.nan

def compute_used_ratio(proposed, total=FIXED_ROUND):
    if np.isnan(proposed):
        return np.nan
    return (proposed / total) * 100.0

cell_data = {
    ds: {
        sc: {
            m: {"proposed": np.nan, "used": np.nan}
            for m in METHODS
        }
        for sc in SPLIT_COEFS
    }
    for ds in DATASETS
}

for ds in DATASETS:
    for sc in SPLIT_COEFS:
        for method in METHODS:
            vals = []
            for seed in SEEDS:
                vals.append(compute_proposed(ds, seed, sc, method))

            arr = np.array(vals, dtype=float)
            if np.any(~np.isnan(arr)):
                mean_p = float(np.nanmean(arr))

                display_round = round(mean_p, 1)
                used_ratio = compute_used_ratio(display_round)

                cell_data[ds][sc][method]["proposed"] = display_round
                cell_data[ds][sc][method]["used"] = used_ratio

def fmt_cell(proposed, used, highlight=False):
    if np.isnan(proposed):
        return "--"

    p_str = f"{proposed:.1f}"

    if np.isnan(used):
        core = rf"{p_str}\;(\text{{-}})"
    else:
        core = rf"{p_str}\;({used:.1f}\%)"

    if highlight:
        return rf"$\underline{{\mathbf{{{core}}}}}$"
    else:
        return rf"${core}$"

n_data_cols = len(METHODS)
col_spec = "cr" + "c" * n_data_cols

DS_SHORT = {
    "dermamnist": "Skin lesion",
    "bloodmnist": "Blood cell",
    "pathmnist": "Colon path.",
}

tex = []

tex.append(r"\begin{table*}[t]")
tex.append(r"\centering")
tex.append(r"\footnotesize")
tex.append(r"\renewcommand{\arraystretch}{1.4}")
tex.append(r"\setlength{\tabcolsep}{2pt}")
tex.append(rf"\begin{{tabular}}{{{col_spec}}}")
tex.append(r"\toprule")

# header
hdr = ["", r"$\log c$"] + [rf"\textbf{{{m}}}" for m in METHODS]
tex.append(" & ".join(hdr) + r" \\")
tex.append(r"\midrule")

# rows
for ds_idx, ds in enumerate(DATASETS):
    n_sc = len(SPLIT_COEFS)

    for sc_idx, sc in enumerate(SPLIT_COEFS):
        row = []

        # -------- 핵심: 최소 used% 찾기 --------
        used_vals = []
        for m in METHODS:
            u = cell_data[ds][sc][m]["used"]
            if not np.isnan(u):
                used_vals.append(u)

        min_used = min(used_vals) if used_vals else np.nan

        if sc_idx == 0:
            row.append(
                rf"\multirow{{{n_sc}}}{{*}}"
                rf"{{\rotatebox{{90}}{{{DS_SHORT[ds]}}}}}"
            )
        else:
            row.append("")

        row.append(f"${SC_DISPLAY[sc]}$")

        for m in METHODS:
            v_prop = cell_data[ds][sc][m]["proposed"]
            v_used = cell_data[ds][sc][m]["used"]

            highlight = (
                not np.isnan(v_used)
                and not np.isnan(min_used)
                and np.isclose(v_used, min_used)
            )

            row.append(fmt_cell(v_prop, v_used, highlight))

        tex.append(" & ".join(row) + r" \\")

    if ds_idx < len(DATASETS) - 1:
        tex.append(r"\midrule")

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(
    rf"\caption{{Early stopping round and usage ratio relative to {FIXED_ROUND} FL rounds. "
    rf"Each cell shows $\text{{round}}\;(\text{{used}}\%)$, where "
    rf"$\text{{used}}\% = \text{{round}}/{FIXED_ROUND}\times100$. "
    rf"The lowest usage per $\log c$ is underlined and bolded (ties included).}}"
)

tex.append(r"\label{tab:round_usage}")
tex.append(r"\end{table*}")

# write
with open("table.tex", "w") as f:
    f.write("\n".join(tex))

print("Saved -> table1.tex")