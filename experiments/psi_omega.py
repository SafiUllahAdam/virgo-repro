'''Sensitivity test: Eq. 3-4 divides λ by ω, "the structural attributes of v1", which the paper never writes out.'''
# Fix 4A shipped ω = raw degree + Ω and the choice was left OPEN at the 2026-06-24 review (notes.md:152, :209).
# Four readings are compared here, all on the SAME λ so only the divisor moves:
#   deg_ev    degree + Ω   - as shipped (`psi`)
#   deg       degree       - drops Ω from the divisor
#   ev        Ω            - drops degree from the divisor
#   delta_ev  Δ + Ω        - the paper's OWN two named structural properties (3.2.1), and the reviewer's alternative
# ω > 0 in every arm, so sign(λ/ω) = sign(λ): the arm CANNOT change which nodes the Fix-8 clamp hits. `clamped_frac`
# is printed anyway, as the check that this holds. What the arm does change is magnitude, hence the top-K neighbourhoods.

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
from experiments.run_core import score, split, tag

ARMS = ["psi", "psi_w_deg", "psi_w_ev", "psi_w_delta"]            # the Poisson form throughout, so only ω varies
ALT_SEED = 43                                                     # second RNG draw: same construction, different tie-class sample


# Jaccard over edge SETS; nx edges are unordered pairs, so frozensets make the comparison direction-free.
def jaccard(A, B):
    '''Edge-set Jaccard of two graphs on the same nodes.'''
    a = {frozenset(e) for e in A.edges}
    b = {frozenset(e) for e in B.edges}
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def measure(ds, k, split_seed=None):
    '''One row per dataset x ω arm: what the divisor does to the signature and to the role graph it builds.'''
    G = graph_io.load_graph(split(ds, split_seed) / "train.edgelist" if split_seed else cfg.dataset(ds)["edgelist"])
    vg = VirtualGraph(G, seed=cfg.REPRO["seed"])
    nodes, base = vg.signatures("psi")
    deg = [G.degree(n) for n in nodes]
    live = np.array([G.degree(n) > 0 for n in nodes])              # isolated nodes score 0.0 in every arm, not a clamp
    lam = vg.signatures("psi_lambda")[1][:, 0]                     # λ/ω itself: its SIGN is what the clamp reads, and ω>0 cannot change it
    P = vg.build("psi", k)
    floor = jaccard(P, VirtualGraph(G, seed=ALT_SEED).build("psi", k))
    rows = []
    for arm in ARMS:
        _, X = vg.signatures(arm)
        v = X[:, 0]
        A = vg.build(arm, k)
        rows.append({
            "dataset": ds, "arm": arm, "omega": PSI_ARMS[arm][1], "nodes": G.number_of_nodes(),
            # Constant down this column by construction: ω only rescales, so it cannot move the clamp. Measured, not assumed.
            "clamped_frac": round(float(((lam <= 0) & live).sum() / max(live.sum(), 1)), 4),
            "sig_min": round(float(v.min()), 2), "sig_max": round(float(v.max()), 2),
            "nonfinite": int((~np.isfinite(X[:, 0])).sum()),
            "rho_vs_psi": round(float(spearmanr(v, base[:, 0]).statistic), 4),
            "rho_vs_degree": round(float(spearmanr(v, deg).statistic), 4),
            "jaccard_vs_psi": round(float(jaccard(A, P)), 4),
            "overlap_ratio": round(float(jaccard(A, P) / floor), 4) if floor else float("nan"),
        })
    return rows


# psi and one rival are scored on the SAME split per seed, so split variance cancels in a per-seed difference.
def paired(ds, k, seeds, task, arm, encoder="graphsage_edge"):
    '''Per-seed psi minus one ω arm on the cached embeddings, with the standard error of the DIFFERENCE.'''
    per = {}
    for sim in ("psi", arm):
        for seed in seeds:
            emb = cfg.NB3_DIR / task / ds / f"k{k}" / sim / f"{tag(encoder, 'all')}_s{seed}.emb"
            if emb.exists():
                per.setdefault(seed, {})[sim] = float(score(ds, emb, task, seed))
    both = [v for v in per.values() if len(v) == 2]
    if not both:
        return {"dataset": ds, "arm": arm, "n_seeds": 0}
    d = np.array([v[arm] - v["psi"] for v in both])
    sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"dataset": ds, "arm": arm, "omega": PSI_ARMS[arm][1], "n_seeds": len(d),
            "psi": round(float(np.mean([v["psi"] for v in both])), 4),
            "arm_score": round(float(np.mean([v[arm] for v in both])), 4),
            "gap": round(float(d.mean()), 4), "paired_sem": round(sem, 4),
            "paired_ratio": round(float(d.mean() / sem), 2) if sem and sem == sem else float("nan")}


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.paired:
        arms = [a for a in cfg.VG_SIMS if a.startswith("psi_")]    # every non-published Ψ arm that has embeddings on disk
        df = pd.DataFrame([paired(ds, args.k, args.seeds, args.task, a) for ds in args.datasets for a in arms])
        df = df[df["n_seeds"] > 0]
        df.to_csv(cfg.RESULTS_DIR / "psi_omega_paired.csv", index=False)
        print("\nPAIRED arm - psi (same split per seed; POSITIVE gap = the arm BEATS the shipped ω)")
        print(df.to_string(index=False))
        print(f"\n{len(df):3d} rows -> results/psi_omega_paired.csv")
        return
    rows = [r for ds in args.datasets for r in measure(ds, args.k, args.split_seed)]
    df = pd.DataFrame(rows)
    out = f"psi_omega_train_s{args.split_seed}.csv" if args.split_seed else "psi_omega.csv"
    df.to_csv(cfg.RESULTS_DIR / out, index=False)
    print("\nPER-DATASET x ω ARM (same λ throughout; only the divisor changes)")
    print(df.to_string(index=False))
    bad = df.groupby("dataset")["clamped_frac"].nunique()
    print(f"\nclamp invariance (must be 1 for every dataset): {dict(bad)}")
    print(f"non-finite signatures: {int(df['nonfinite'].sum())} across all arms")
    print(f"\n{len(df):3d} rows -> results/{out}")


# Defines command-line options (mirrors psi_clamp.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Compare readings of Eq. 3-4's omega, on the build and on the score.")
    p.add_argument('--datasets', nargs='+', default=["cora", "twitch_ptbr", "squirrel_filtered"], help='Datasets. Default: the three smoke-test graphs.')
    p.add_argument('--k', type=int, default=cfg.VG_K[1], help='Top-K of the role graph. Default 10 (the locked setting).')
    p.add_argument('--split-seed', type=int, default=None, help="Measure on this LP seed's 70%% train graph instead of the full graph.")
    p.add_argument('--paired', action='store_true', help='Score cached arm-vs-psi embeddings PAIRED per seed instead of measuring the build.')
    p.add_argument('--seeds', type=int, nargs='+', default=cfg.VG_SEEDS, help='Seeds to score in --paired mode; missing embeddings are skipped, never trained.')
    p.add_argument('--task', default='link_prediction', help='Task to score in --paired mode. Default link_prediction.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
