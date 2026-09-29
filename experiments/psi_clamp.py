'''Diagnostic: how much does Fix-8's 1e-12 clamp change the psi role graph, measured against psi_lambda (same λ, no wrapper)?'''
# psi = k*log(λ) - λ - log(k!) needs λ>0, so it clamps λ at 1e-12. But λ = Σ Δ log(Δ/Ω) / (deg+Ω) is NOT a true KL -
# Δ and Ω are never normalized over N(u) - so λ goes negative, and every clamped node collapses to k*log(1e-12) - log(k!),
# a pure function of degree. psi_lambda uses the same λ directly: no clamp, and no -log(k!) degree term either.
# This script measures the size of that difference. It builds role graphs, never trains, and fits nothing.
#   the confound  build() SAMPLES inside a tie class, so two builds of ONE signal already disagree; a raw psi-vs-psi_lambda
#                 Jaccard is meaningless without that floor. `overlap_ratio` = cross / self-overlap, as in role_overlap.py.
#   the invariant `sig_spearman` needs no sampling; `rho_degree` says how far each signature is from being degree itself.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import frozen_rules as fr
from virgo import graph_io
from virgo.virtual_graph import VirtualGraph
from experiments.run_core import score, split, tag

ALT_SEED = 43                                                     # second RNG draw: same construction, different tie-class sample


# Jaccard over edge SETS; nx edges are unordered pairs, so frozensets make the comparison direction-free.
def jaccard(A, B):
    '''Edge-set Jaccard of two graphs on the same nodes.'''
    a = {frozenset(e) for e in A.edges}
    b = {frozenset(e) for e in B.edges}
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def measure(ds, k, split_seed=None):
    '''One row per dataset: how often the clamp fires, and how far the two signatures and role graphs move apart.'''
    # LP trains on the seed's 70% train graph, so the clamp rate there is what the trained cells actually saw.
    G = graph_io.load_graph(split(ds, split_seed) / "train.edgelist" if split_seed else cfg.dataset(ds)["edgelist"])
    vg = VirtualGraph(G, seed=cfg.REPRO["seed"])
    nodes, Xp = vg.signatures("psi")
    _, Xl = vg.signatures("psi_lambda")
    deg = [G.degree(n) for n in nodes]
    lam = Xl[:, 0]
    nonisolated = np.array([G.degree(n) > 0 for n in nodes])
    clamped = (lam <= 1e-12) & nonisolated                        # isolated nodes return 0.0 from both, not a clamp
    P, L = vg.build("psi", k), vg.build("psi_lambda", k)
    P2 = VirtualGraph(G, seed=ALT_SEED).build("psi", k)
    L2 = VirtualGraph(G, seed=ALT_SEED).build("psi_lambda", k)
    cross, floor = jaccard(P, L), np.mean([jaccard(P, P2), jaccard(L, L2)])
    return {
        "dataset": ds, "k": k, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        # THE headline: fraction of non-isolated nodes whose psi signature is k*log(1e-12) - log(k!), i.e. degree alone.
        "clamped_frac": round(float(clamped.sum() / max(nonisolated.sum(), 1)), 4),
        "lambda_min": round(float(lam.min()), 4), "lambda_max": round(float(lam.max()), 4),
        "sig_spearman": round(float(spearmanr(Xp[:, 0], lam).statistic), 4),
        "rho_degree_psi": round(float(spearmanr(Xp[:, 0], deg).statistic), 4),
        "rho_degree_lambda": round(float(spearmanr(lam, deg).statistic), 4),
        "jaccard_cross": round(float(cross), 4),
        "overlap_ratio": round(float(cross / floor), 4) if floor else float("nan"),
    }


# psi and psi_lambda are scored on the SAME split per seed, so the split variance cancels in a per-seed difference
# (the tie_break.py argument). Reads embeddings already on disk and never trains; a missing seed is skipped, not run.
def paired(ds, k, seeds, task, encoder="graphsage_edge"):
    '''Per-seed psi minus psi_lambda on the cached embeddings, with the standard error of the DIFFERENCE.'''
    per = {}
    for sim in ("psi", "psi_lambda"):
        for seed in seeds:
            emb = cfg.NB3_DIR / task / ds / f"k{k}" / sim / f"{tag(encoder, 'all')}_s{seed}.emb"
            if emb.exists():
                per.setdefault(seed, {})[sim] = float(score(ds, emb, task, seed))
    both = [v for v in per.values() if len(v) == 2]
    if not both:
        return {"dataset": ds, "n_seeds": 0}
    d = np.array([v["psi"] - v["psi_lambda"] for v in both])
    sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"dataset": ds, "n_seeds": len(d),
            "psi": round(float(np.mean([v["psi"] for v in both])), 4),
            "psi_lambda": round(float(np.mean([v["psi_lambda"] for v in both])), 4),
            "gap": round(float(d.mean()), 4), "paired_sem": round(sem, 4),
            "paired_ratio": round(float(d.mean() / sem), 2) if sem and sem == sem else float("nan")}


def report(df):
    '''Print the per-dataset table and the one-line reading of it.'''
    print("\nPER-DATASET (K locked; --split-seed measures the 70%% train graph the LP cells were actually built from)")
    print(df.drop(columns=["k"]).to_string(index=False))
    print(f"\nclamped_frac: mean {df['clamped_frac'].mean():.3f}, max {df['clamped_frac'].max():.3f} "
          f"({df.loc[df['clamped_frac'].idxmax(), 'dataset']}), zero on {(df['clamped_frac'] == 0).sum()}/{len(df)}")
    print(f"|rho| with degree: psi {df['rho_degree_psi'].abs().mean():.3f} vs psi_lambda {df['rho_degree_lambda'].abs().mean():.3f} (mean)")
    print("\nTHE QUESTION THIS ANSWERS: if overlap_ratio is ~1 everywhere the clamp is cosmetic and no retraining is needed.")
    print("Where it is well below 1, the two build genuinely different role graphs and only training can say which is better.")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.paired:                                                # scoring mode: compare the two variants already trained
        df = pd.DataFrame([paired(ds, args.k, args.seeds, args.task) for ds in args.datasets])
        df.to_csv(cfg.RESULTS_DIR / "psi_clamp_paired.csv", index=False)
        print("\nPAIRED psi - psi_lambda (same split per seed; positive gap = the CLAMPED psi scores higher)")
        print(df.to_string(index=False))
        print(f"\n{len(df):3d} rows -> results/psi_clamp_paired.csv")
        return
    rows = []
    for ds in args.datasets:
        rows.append(measure(ds, args.k, args.split_seed))
        print(f"measured {ds}: clamped {rows[-1]['clamped_frac']}  ratio {rows[-1]['overlap_ratio']}  rho {rows[-1]['sig_spearman']}", flush=True)
    df = pd.DataFrame(rows)
    out = f"psi_clamp_train_s{args.split_seed}.csv" if args.split_seed else "psi_clamp.csv"
    df.to_csv(cfg.RESULTS_DIR / out, index=False)
    report(df)
    print(f"\n{len(df):3d} rows -> results/{out}")


# Defines command-line options (mirrors role_overlap.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure how far the 1e-12 clamp moves the psi role graph.")
    p.add_argument('--datasets', nargs='+', default=fr.STRATEGY_PANEL,
                   help='Datasets to measure. Default: the stage-2 strategy panel.')
    p.add_argument('--k', type=int, default=cfg.VG_K[1], help='Top-K of the role graph. Default 10 (the locked setting).')
    p.add_argument('--split-seed', type=int, default=None, help="Measure on this LP seed's 70%% train graph instead of the full graph (what the LP cells were built from).")
    p.add_argument('--paired', action='store_true', help='Score cached psi vs psi_lambda embeddings PAIRED per seed instead of measuring the build.')
    p.add_argument('--seeds', type=int, nargs='+', default=cfg.VG_SEEDS, help='Seeds to score in --paired mode; missing embeddings are skipped, never trained.')
    p.add_argument('--task', default='link_prediction', help='Task to score in --paired mode. Default link_prediction.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
