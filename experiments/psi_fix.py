'''Do the two principled repairs of Fix-8's 1e-12 clamp actually remove it, and what do they do to Ψ?'''
# The clamp is a numerical patch, not I2V. Two repairs were proposed (supervisor, 2026-09-13):
#   (a) normalize Ω over N(u)            -> `psi_qnorm`
#       and, because that alone does NOT bound λ, normalize Δ too -> `psi_pqnorm`, a true KL, λ >= 0 by Gibbs
#   (b) shift λ so the graph minimum is 0 -> `psi_shift`, order-preserving, the floor then bites ONE node not a third
# This script measures the repairs on the BUILD side only; it never trains and fits nothing. `floored_frac` is the
# number the whole exercise is about: the share of non-isolated nodes whose λ still hits the 1e-12 floor.
#   the k=1 catch  under full normalization a node with ONE neighbour has p=q=1 exactly, so λ=0 and it floors anyway.
#                  `deg1_frac` is printed next to it, because that degeneracy is a property of the repair, not a bug.
#   the confound   build() samples inside tie classes, so two builds of ONE signal already disagree; `overlap_ratio`
#                  divides by that self-overlap floor, as in psi_clamp.py / role_overlap.py.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo.virtual_graph import PSI_ARMS, VirtualGraph
from experiments.run_core import split

ARMS = ["psi", "psi_qnorm", "psi_pqnorm", "psi_shift"]            # shipped, fix (a) partial, fix (a) full, fix (b)
ALT_SEED = 43                                                     # second RNG draw: same construction, different tie-class sample


# Jaccard over edge SETS; nx edges are unordered pairs, so frozensets make the comparison direction-free.
def jaccard(A, B):
    '''Edge-set Jaccard of two graphs on the same nodes.'''
    a = {frozenset(e) for e in A.edges}
    b = {frozenset(e) for e in B.edges}
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def measure(ds, k, split_seed=None):
    '''One row per dataset x repair: does the floor still fire, and how far does the repair move Ψ and its role graph?'''
    G = graph_io.load_graph(split(ds, split_seed) / "train.edgelist" if split_seed else cfg.dataset(ds)["edgelist"])
    vg = VirtualGraph(G, seed=cfg.REPRO["seed"])
    nodes, base = vg.signatures("psi")
    neigh = vg.core.node_neighbors()
    live = np.array([len(neigh[n]) > 0 for n in nodes])
    deg = [G.degree(n) for n in nodes]
    P = vg.build("psi", k)
    floor = jaccard(P, VirtualGraph(G, seed=ALT_SEED).build("psi", k))
    rows = []
    for arm in ARMS:
        poisson, omega, norm, shift = PSI_ARMS[arm]
        lam = np.array([vg.psi_lambda_of(n, neigh, vg.core.degree_distribution(), vg.core.eigenvector_centrality(),
                                         vg.core.degree_node(), omega, norm) for n in nodes])
        if shift:
            lam = np.where(live, lam - lam[live].min(), 0.0)
        _, X = vg.signatures(arm)
        A = vg.build(arm, k)
        rows.append({
            "dataset": ds, "arm": arm, "nodes": G.number_of_nodes(),
            # THE number: share of non-isolated nodes whose λ still hits the 1e-12 floor.
            "floored_frac": round(float(((lam <= 1e-12) & live).sum() / max(live.sum(), 1)), 4),
            "deg1_frac": round(float((np.array(deg) == 1).sum() / max(live.sum(), 1)), 4),
            "lambda_min": float(f"{lam[live].min():.4g}") if live.any() else float("nan"),
            "rho_vs_psi": round(float(spearmanr(X[:, 0], base[:, 0]).statistic), 4),
            "rho_vs_degree": round(float(spearmanr(X[:, 0], deg).statistic), 4),
            "overlap_ratio": round(float(jaccard(A, P) / floor), 4) if floor else float("nan"),
        })
    return rows


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for ds in args.datasets for r in measure(ds, args.k, args.split_seed)]
    df = pd.DataFrame(rows)
    out = f"psi_fix_train_s{args.split_seed}.csv" if args.split_seed else "psi_fix.csv"
    df.to_csv(cfg.RESULTS_DIR / out, index=False)
    print("\nPER-DATASET x REPAIR (build side only; floored_frac is the share of nodes the 1e-12 floor still catches)")
    print(df.to_string(index=False))
    print("\nmean floored_frac by repair:")
    print(df.groupby("arm")["floored_frac"].mean().round(4).to_string())
    print("\nmean |rho| vs degree by repair:")
    print(df.groupby("arm")["rho_vs_degree"].apply(lambda c: c.abs().mean()).round(4).to_string())
    print(f"\n{len(df):3d} rows -> results/{out}")


# Defines command-line options (mirrors psi_clamp.py / psi_omega.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure whether the proposed repairs actually remove the 1e-12 clamp.")
    p.add_argument('--datasets', nargs='+', default=["roman_empire", "cora", "enzymes", "amherst41"],
                   help='Datasets. Default: four graphs spanning the clamp range (0%%, 12%%, 47%%, 99%%).')
    p.add_argument('--k', type=int, default=cfg.VG_K[1], help='Top-K of the role graph. Default 10 (the locked setting).')
    p.add_argument('--split-seed', type=int, default=None, help="Measure on this LP seed's 70%% train graph instead of the full graph.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
