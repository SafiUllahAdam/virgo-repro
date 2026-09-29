'''Mechanism study: how much do the DEGREE and PSI role graphs actually differ, and does the difference track which one wins?'''
# Modules 10 and 11 asked "which graph property predicts a degree win" twice and got nothing twice (paper_log 13, 18).
# The standing explanation was never measured: psi is a Poisson/KL score over degree-based signatures, so on most graphs
# the two constructions may build near-identical role graphs, in which case no property CAN separate them.
# This script measures that directly. It builds role graphs, never trains, and fits nothing.
#   the confound  the 1-D signature ties massively and build() SAMPLES inside a tie class, so two builds of the SAME
#                 signal already disagree. A raw degree-vs-psi overlap is meaningless without that floor.
#   the control   rebuild each signal under a second RNG seed -> self-overlap = how much disagreement sampling alone
#                 produces. `overlap_ratio` = cross-overlap / self-overlap: ~1 means the two signals are as similar as
#                 two draws of one signal, i.e. indistinguishable by construction; << 1 means they genuinely differ.
#   the invariant `sig_spearman`, the rank correlation of the two signatures, needs no sampling and cannot be confounded.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo.virtual_graph import VirtualGraph

VERDICTS = cfg.RESULTS_DIR / "degree_rule_corpus_verdicts.csv"
ALT_SEED = 43                                                     # second RNG draw: same construction, different tie-class sample


# One role graph plus its signature, built on the FULL original graph (the Module-4 convention for role-graph properties).
def role(G, sim, k, seed):
    '''Role graph for one signal and one RNG seed, with the signature vector it was built from.'''
    vg = VirtualGraph(G, seed=seed)
    nodes, X = vg.signatures(sim)
    return vg.build(sim, k), dict(zip(nodes, X[:, 0]))


# Jaccard over edge SETS; nx edges are unordered pairs, so frozensets make the comparison direction-free.
def jaccard(A, B):
    '''Edge-set Jaccard of two graphs on the same nodes.'''
    a = {frozenset(e) for e in A.edges}
    b = {frozenset(e) for e in B.edges}
    return len(a & b) / len(a | b) if (a or b) else float("nan")


# What each construction connects a node TO, described by degree rather than identity: the sampling confound moves node
# ids around inside a tie class but not their degrees, so this compares the two role graphs where they are comparable.
def profile_gap(G, A, B):
    '''Mean |sorted neighbour-degrees(A) - sorted neighbour-degrees(B)| per node, in units of the graph mean degree.'''
    deg = dict(G.degree)
    mean_deg = float(np.mean(list(deg.values()))) or 1.0
    gaps = []
    for v in G.nodes:
        a, b = sorted(deg[u] for u in A[v]), sorted(deg[u] for u in B[v])
        m = min(len(a), len(b))
        if m:
            gaps.append(float(np.mean(np.abs(np.array(a[:m]) - np.array(b[:m])))))
    return round(float(np.mean(gaps)) / mean_deg, 4) if gaps else float("nan")


def measure(ds, k):
    '''One row per dataset: cross-signal overlap, the sampling floor it must be read against, and signature agreement.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    D, sig_d = role(G, "degree", k, cfg.REPRO["seed"])
    P, sig_p = role(G, "psi", k, cfg.REPRO["seed"])
    D2, _ = role(G, "degree", k, ALT_SEED)
    P2, _ = role(G, "psi", k, ALT_SEED)
    nodes = list(G.nodes)
    cross, floor_d, floor_p = jaccard(D, P), jaccard(D, D2), jaccard(P, P2)
    floor = np.mean([floor_d, floor_p])
    rho = spearmanr([sig_d[n] for n in nodes], [sig_p[n] for n in nodes]).statistic
    return {
        "dataset": ds, "k": k, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "edges_degree_vg": D.number_of_edges(), "edges_psi_vg": P.number_of_edges(),
        "jaccard_cross": round(cross, 4),
        "jaccard_floor_degree": round(floor_d, 4), "jaccard_floor_psi": round(floor_p, 4),
        # The headline. 1.0 = the two signals differ no more than one signal differs from itself under resampling.
        "overlap_ratio": round(float(cross / floor), 4) if floor else float("nan"),
        "sig_spearman": round(float(rho), 4) if rho == rho else float("nan"),
        "profile_gap_cross": profile_gap(G, D, P),
        "profile_gap_floor": profile_gap(G, D, D2),
    }


def report(df):
    '''Print the per-dataset table, then the same numbers grouped by which signal actually won.'''
    print("\nPER-DATASET (role graphs built on the FULL original graph; the LP runs built them on each seed's 70% train graph)")
    print(df[["dataset", "nodes", "edges", "jaccard_cross", "jaccard_floor_degree", "jaccard_floor_psi",
              "overlap_ratio", "sig_spearman", "profile_gap_cross", "profile_gap_floor"]].to_string(index=False))
    if "signal" not in df:
        return
    print("\nGROUPED BY THE SIGNAL THAT ACTUALLY WON (paired band; 'tie' = augments but names no single signal)")
    g = df.groupby("signal").agg(n=("dataset", "size"), jaccard_cross=("jaccard_cross", "mean"),
                                 overlap_ratio=("overlap_ratio", "mean"), sig_spearman=("sig_spearman", "mean"),
                                 profile_gap_cross=("profile_gap_cross", "mean"))
    print(g.round(4).to_string())
    print("\nTHE QUESTION THIS ANSWERS: if degree wins where the two role graphs genuinely DIVERGE, `overlap_ratio` and")
    print("`sig_spearman` should be lower on the degree rows than on the tie rows. Read the group sizes before the means -")
    print("degree wins alone on five graphs in the whole study, so these are descriptions, not tests.")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        rows.append(measure(ds, args.k))
        print(f"measured {ds}: ratio {rows[-1]['overlap_ratio']}  rho {rows[-1]['sig_spearman']}", flush=True)
    df = pd.DataFrame(rows)
    if VERDICTS.exists():                                          # label each row with the signal its seeds named, when known
        v = pd.read_csv(VERDICTS).query("band == 'paired'").set_index("dataset")
        df["winner_signals"] = [v["winner_signals"].get(d, "") for d in df["dataset"]]
        df["beats_original"] = [bool(v["beats_original"].get(d, False)) for d in df["dataset"]]
        df["signal"] = [("no augmentation" if not b else s if "|" not in s else "tie")
                        for s, b in zip(df["winner_signals"], df["beats_original"])]
    df.to_csv(cfg.RESULTS_DIR / "role_overlap.csv", index=False)
    report(df)
    print(f"\n{len(df):3d} rows -> results/role_overlap.csv")


# Defines command-line options (mirrors degree_rule.py / tie_break.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure how much the degree and psi role graphs differ, against the sampling floor.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to measure. Default: the degree corpus.')
    p.add_argument('--k', type=int, default=10, help='Top-K of the role graph. Default 10 (the locked setting).')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
