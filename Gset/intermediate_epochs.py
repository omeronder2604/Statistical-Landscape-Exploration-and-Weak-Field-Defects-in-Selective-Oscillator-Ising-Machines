# -*- coding: utf-8 -*-

import os
import glob
import re
from pathlib import Path
import numpy as np

# ============================================================
# 1. USER SETTINGS
# ============================================================

DATA_DIR = str(Path(__file__).resolve().parent.parent / "raw_data" / "Gset")

KEEP_FRACTION = 1.0 / 3.0

DH_OIM_FINAL = {
    "G1": 11624,
    "G6": 2177,
    "G11": 562,
    "G14": 3062,
    "G15": 3050,
    "G22": 13357,
    "G23": 13339,
    "G36": 7672,
    "G39": 2407,
}

# ============================================================
# 2. FUNCTIONS
# ============================================================

def phases_to_spins(phases):
    """
    Convert oscillator phases to Ising spins:
        s_i = sign(cos(phi_i))
    Exact zeros are assigned to +1.
    """
    spins = np.sign(np.cos(phases))
    spins[spins == 0] = 1.0
    return spins


def calculate_cuts(spins, J):
    """
    Same Max-Cut convention as the G-set solver.

    J[u,v] = -w, therefore W = -J.

    Cut(s) = 1/4 sum_ij W_ij (1 - s_i s_j)
    """
    W = -J
    total_edge_weight = 0.5 * np.sum(W)
    interaction = np.einsum("ri,ij,rj->r", spins, W, spins, optimize=True)
    cuts = 0.5 * total_edge_weight - 0.25 * interaction
    return cuts


def analyze_phase_configuration(phases, J, keep_fraction=1.0 / 3.0):
    spins = phases_to_spins(phases)
    cuts = calculate_cuts(spins, J)

    num_replicas = len(cuts)
    num_keep = max(1, int(num_replicas * keep_fraction))

    order = np.argsort(-cuts, kind="stable")
    selected = order[:num_keep]

    return {
        "num_replicas": num_replicas,
        "num_keep": num_keep,
        "max_cut": cuts[order[0]],
        "culling_lower_bound": cuts[selected[-1]],
        "mean_cut": np.mean(cuts),
        "median_cut": np.median(cuts),
        "cuts": cuts,
    }


def extract_graph_number(filename):
    match = re.search(r"G(\d+)", filename)
    if match:
        return int(match.group(1))
    return 999999


def extract_graph_name(filename):
    match = re.search(r"G(\d+)", filename)
    if match:
        return f"G{match.group(1)}"
    return filename


def print_percentage_progression(graph_max_cuts):
    print("\n" + "=" * 88)
    print("PERCENTAGE OF FINAL DH-OIM+ MAX-CUT")
    print("=" * 88)
    print(f"{'Graph':<8}{'Epoch 1':>12}{'Epoch 2':>12}{'Epoch 3':>12}{'Epoch 4':>12}{'DH-OIM+':>12}")
    print("-" * 88)

    for graph_name, max_cuts in graph_max_cuts.items():
        if graph_name not in DH_OIM_FINAL:
            continue

        max_cuts = np.asarray(max_cuts, dtype=float)
        if len(max_cuts) != 4:
            continue

        final_cut = DH_OIM_FINAL[graph_name]
        percentages = 100.0 * max_cuts / final_cut

        print(
            f"{graph_name:<8}"
            f"{percentages[0]:>11.3f}%"
            f"{percentages[1]:>11.3f}%"
            f"{percentages[2]:>11.3f}%"
            f"{percentages[3]:>11.3f}%"
            f"{100.0:>11.3f}%"
        )

    print("=" * 88)


# ============================================================
# 3. MAIN
# ============================================================

def main():
    npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))

    if not npz_files:
        print("No .npz files found in:")
        print(DATA_DIR)
        return

    npz_files = sorted(npz_files, key=lambda x: extract_graph_number(os.path.basename(x)))

    print("=" * 94)
    print("G-SET INTERMEDIATE EPOCH MAX-CUT ANALYSIS")
    print("=" * 94)

    graph_max_cuts = {}

    for filepath in npz_files:
        filename = os.path.basename(filepath)
        graph_name = extract_graph_name(filename)

        print()
        print("=" * 94)
        print(f"FILE: {filename}")
        print("=" * 94)

        data = np.load(filepath)

        print("NPZ keys:")
        print("   ", data.files)
        print()

        if "J" not in data:
            print("ERROR: J not found in file.")
            data.close()
            continue

        J = data["J"]

        epoch_keys = ["epoch_1_phis", "epoch_2_phis", "epoch_3_phis", "epoch_4_phis"]

        found_epoch = False
        max_cut_values = []

        print(
            f"{'Configuration':<20}"
            f"{'Replicas':>10}"
            f"{'Keep':>8}"
            f"{'Max Cut':>14}"
            f"{'Cull Lower':>14}"
            f"{'Mean Cut':>14}"
            f"{'Median':>14}"
        )
        print("-" * 94)

        for epoch_key in epoch_keys:
            if epoch_key not in data:
                continue

            found_epoch = True
            phases = data[epoch_key]
            result = analyze_phase_configuration(phases, J, KEEP_FRACTION)

            print(
                f"{epoch_key:<20}"
                f"{result['num_replicas']:>10d}"
                f"{result['num_keep']:>8d}"
                f"{result['max_cut']:>14.0f}"
                f"{result['culling_lower_bound']:>14.0f}"
                f"{result['mean_cut']:>14.2f}"
                f"{result['median_cut']:>14.2f}"
            )

            max_cut_values.append(result["max_cut"])

        if found_epoch:
            graph_max_cuts[graph_name] = np.asarray(max_cut_values, dtype=float)

        if not found_epoch and "replica_states" in data:
            phases = data["replica_states"]
            result = analyze_phase_configuration(phases, J, KEEP_FRACTION)

            print(
                f"{'replica_states':<20}"
                f"{result['num_replicas']:>10d}"
                f"{result['num_keep']:>8d}"
                f"{result['max_cut']:>14.0f}"
                f"{result['culling_lower_bound']:>14.0f}"
                f"{result['mean_cut']:>14.2f}"
                f"{result['median_cut']:>14.2f}"
            )

        elif not found_epoch:
            print("No epoch phase arrays or replica_states found.")

        data.close()

    if graph_max_cuts:
        print_percentage_progression(graph_max_cuts)

    print()
    print("=" * 94)
    print("DONE")
    print("=" * 94)


if __name__ == "__main__":
    main()