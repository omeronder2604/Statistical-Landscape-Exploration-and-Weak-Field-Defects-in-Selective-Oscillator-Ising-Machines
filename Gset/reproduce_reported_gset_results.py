"""Recompute G-set epoch cuts and a deterministic E4 diagnostic refinement.

Inputs are the nine graph texts and checkpoint NPZs committed at --ref.
The best-known cuts below are published reference values, not simulation data.
No oscillator search is run and no checkpoint is written.
"""

import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
from numba import njit, prange


ROOT = Path(__file__).resolve().parents[1]
BEST_KNOWN = {
    "G1": 11624, "G6": 2178, "G11": 564, "G14": 3064,
    "G15": 3050, "G22": 13359, "G23": 13344, "G36": 7680,
    "G39": 2408,
}
REPLICAS = 448
PROPOSALS = 10_000
SEED = 42


def committed_bytes(ref, path):
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT
    )


def graph_edges(data):
    lines = data.decode("ascii").splitlines()
    n, count = map(int, lines[0].split())
    edges = np.array([list(map(int, line.split())) for line in lines[1:] if line.strip()], dtype=np.int64)
    if edges.shape != (count, 3):
        raise ValueError(f"Graph header says {count} edges; found {len(edges)}")
    u, v, w = edges.T
    u, v = u - 1, v - 1
    if np.any(u < 0) or np.any(v < 0) or np.any(u >= n) or np.any(v >= n):
        raise ValueError("Graph contains an out-of-range vertex")
    return n, u, v, w


def cuts_from_phases(phases, u, v, w):
    spins = np.sign(np.cos(phases))
    spins[spins == 0] = 1
    return cuts_from_spins(spins, u, v, w)


def cuts_from_spins(spins, u, v, w):
    # Each edge appears once in the graph text; signed weights are retained.
    return np.sum((spins[:, u] != spins[:, v]) * w, axis=1, dtype=np.int64)


@njit(fastmath=True)
def refine_one(J, n, spins, seed):
    # Same proposal order, temperature interpolation, and RNG convention as
    # Gset_OIM.py. A proposal is a single attempted spin flip.
    np.random.seed(int(seed))
    spins = spins.copy()
    for step in range(PROPOSALS):
        p = step / (PROPOSALS - 1)
        temperature = 0.1 * (1 - p) + 0.01 * p
        i = np.random.randint(0, n)
        local_field = np.dot(J[i], spins)
        delta_energy = 2.0 * spins[i] * local_field
        if delta_energy <= 0 or np.random.random() < np.exp(-delta_energy / temperature):
            spins[i] *= -1.0
    return spins


@njit(parallel=True)
def refine_population(J, phases, seeds):
    count, n = phases.shape
    final = np.empty((count, n), dtype=np.float64)
    for r in prange(count):
        spins = np.sign(np.cos(phases[r]))
        for i in range(n):
            if spins[i] == 0:
                spins[i] = 1.0
        final[r] = refine_one(J, n, spins, seeds[r])
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="origin/main", help="Git commit/ref containing the checkpoint inputs")
    args = parser.parse_args()
    seeds = np.random.SeedSequence(SEED + 5000).generate_state(REPLICAS)
    print(f"Input Git ref: {args.ref}; 448 replicas; 10,000 proposals; T=0.1 -> 0.01; seed root={SEED}", flush=True)
    print("Graph,Stage,Max-Cut,Mean,Top-third threshold,Best known,Percent of best known", flush=True)

    for graph, best in BEST_KNOWN.items():
        n, u, v, w = graph_edges(committed_bytes(args.ref, f"raw_data/Gset/{graph}.txt"))
        raw = committed_bytes(args.ref, f"raw_data/Gset/checkpoint_{graph}.txt.npz")
        with np.load(io.BytesIO(raw), allow_pickle=False) as checkpoint:
            if int(checkpoint["N"]) != n:
                raise ValueError(f"{graph}: checkpoint N differs from graph")
            J = checkpoint["J"]
            expected_J = np.zeros((n, n), dtype=np.float64)
            expected_J[u, v] = -w
            expected_J[v, u] = -w
            if not np.array_equal(J, expected_J):
                raise ValueError(f"{graph}: checkpoint J differs from graph weights")
            epochs = [checkpoint[f"epoch_{k}_phis"] for k in range(1, 5)]
            if any(phases.shape != (REPLICAS, n) for phases in epochs):
                raise ValueError(f"{graph}: unexpected epoch shape")
            if not np.array_equal(checkpoint["replica_states"], epochs[-1]):
                raise ValueError(f"{graph}: replica_states differs from E4")

        for k, phases in enumerate(epochs, 1):
            cuts = cuts_from_phases(phases, u, v, w)
            threshold = np.partition(cuts, REPLICAS - 149)[REPLICAS - 149]
            print(f"{graph},E{k},{int(cuts.max())},{cuts.mean():.2f},{int(threshold)},{best},", flush=True)
        final = refine_population(J, epochs[-1], seeds)
        dh = int(cuts_from_spins(final, u, v, w).max())
        percent = 100 * dh / best
        print(f"{graph},DH,{dh},,,{best},{percent:.2f}%", flush=True)


if __name__ == "__main__":
    main()
