'''DEGREE-SPECIFIC properties: a separate tier, measured on the degree role graph itself, never mixed with the 15 general ones.'''
# Modules 10, 11 and the mechanism study (paper_log 13, 17, 18, 19) all asked the degree question with the SAME 15
# graph-level properties and all returned nothing. This module answers a different objection: those 15 describe the
# ORIGINAL graph and say nothing about the object actually under selection - the degree role graph. Four families are
# measured here, on the user's instruction (2026-09-04), and kept in their own tier so they can never silently join the
# general screen or move a rule fitted on it:
#   1 tie structure     the degree signature is an INTEGER, so it ties massively and build() samples inside a tie class
#   2 degree resolution how many distinct degree values the graph offers the construction at all
#   3 stability         how much the degree role graph changes when only the RNG seed changes
#   4 divergence        how far the degree role graph sits from the psi one it is competing against
# Every one of these needs the role graph BUILT, which is allowed - building is not training (CLAUDE.md 7) - but it does
# mean they cannot be read off a dataset the way the 15 can. That cost is real and is reported with any rule fitted here.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from experiments.role_overlap import jaccard, profile_gap, role

FEATURES_CSV = cfg.RESULTS_DIR / "degree_features.csv"

# THE degree tier. Deliberately a separate list from characterize.PREDICTORS_PRIMARY / PREDICTORS_EXPLORATORY: a screen
# must ask for it by name (`--tier degree`), so no existing result can change because this file exists.
DEGREE_TIER = ["degree_tie_ratio", "distinct_degrees", "distinct_degree_frac",
               "degree_vg_stability", "psi_vg_stability",
               "deg_psi_jaccard", "deg_psi_overlap_ratio", "deg_psi_profile_gap", "deg_psi_sig_spearman"]

DEG_SEEDS = [42, 43, 44]        # degree signatures are trivial to compute, so stability gets three draws
PSI_SEEDS = [42, 43]            # psi needs eigenvector centrality per build, so its floor gets two - stated, not hidden


# Reproduces build()'s own tie keying exactly, so "tied" here means what it means inside the builder.
def tie_ratio(G, k):
    '''Mean fraction of a node's K role slots that its OWN degree tie class could fill - 1.0 = the neighbourhood is an arbitrary draw.'''
    vals = np.array([d for _, d in G.degree], dtype=float)
    span = float(vals.max() - vals.min())
    tol = graph_io.GRAPH_POLICY["sig_tol"]
    keys = np.round(vals / (span * tol)) if span > 0 else np.zeros(len(vals))
    sizes = pd.Series(keys).map(pd.Series(keys).value_counts()).to_numpy()
    return round(float(np.mean(np.minimum(sizes - 1, k) / k)), 4)


# Mean pairwise edge Jaccard over seeds: the same construction, different tie-class draws.
def stability(G, sim, k, seeds):
    '''How much a role graph stays the same when ONLY the RNG seed changes; 1.0 = fully determined, 0 = arbitrary.'''
    builds = [role(G, sim, k, s)[0] for s in seeds]
    pairs = [jaccard(builds[i], builds[j]) for i in range(len(builds)) for j in range(i + 1, len(builds))]
    return round(float(np.mean(pairs)), 4), builds[0]


def measure(ds, k):
    '''One row: the four degree-specific families for one dataset.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    deg_stab, D = stability(G, "degree", k, DEG_SEEDS)
    psi_stab, P = stability(G, "psi", k, PSI_SEEDS)
    _, sig_d = role(G, "degree", k, cfg.REPRO["seed"])
    _, sig_p = role(G, "psi", k, cfg.REPRO["seed"])
    nodes = list(G.nodes)
    cross = jaccard(D, P)
    rho = spearmanr([sig_d[n] for n in nodes], [sig_p[n] for n in nodes]).statistic
    uniq = len({d for _, d in G.degree})
    return {
        "dataset": ds, "k": k, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        # 1 tie structure
        "degree_tie_ratio": tie_ratio(G, k),
        # 2 degree resolution
        "distinct_degrees": uniq, "distinct_degree_frac": round(uniq / G.number_of_nodes(), 6),
        # 3 stability under reseeding
        "degree_vg_stability": deg_stab, "psi_vg_stability": psi_stab,
        # 4 divergence from the psi role graph it competes with; the ratio is normalized by DEGREE's own floor, so it
        #   reads as "how close is psi to degree, relative to how close degree is to itself"
        "deg_psi_jaccard": round(cross, 4),
        "deg_psi_overlap_ratio": round(float(cross / deg_stab), 4) if deg_stab else float("nan"),
        "deg_psi_profile_gap": profile_gap(G, D, P),
        "deg_psi_sig_spearman": round(float(rho), 4) if rho == rho else float("nan"),
    }


# Reads what is already on disk and measures only what is missing (same contract as strategy_select.properties).
def degree_features(datasets, k=10):
    '''The degree tier for these datasets, cached in results/degree_features.csv.'''
    have = pd.read_csv(FEATURES_CSV) if FEATURES_CSV.exists() else pd.DataFrame(columns=["dataset"])
    miss = [d for d in datasets if d not in set(have["dataset"])]
    if miss:
        have = pd.concat([have, pd.DataFrame([measure(d, k) for d in miss])], ignore_index=True)
        have.to_csv(FEATURES_CSV, index=False)
    return have[have["dataset"].isin(datasets)][["dataset"] + DEGREE_TIER].sort_values("dataset").reset_index(drop=True)


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        rows.append(measure(ds, args.k))
        print(f"measured {ds}: tie_ratio {rows[-1]['degree_tie_ratio']}  stability {rows[-1]['degree_vg_stability']}  "
              f"ratio {rows[-1]['deg_psi_overlap_ratio']}", flush=True)
    df = pd.DataFrame(rows)
    old = pd.read_csv(FEATURES_CSV) if FEATURES_CSV.exists() else pd.DataFrame(columns=df.columns)
    pd.concat([old[~old["dataset"].isin(df["dataset"])], df], ignore_index=True).to_csv(FEATURES_CSV, index=False)
    print("\nDEGREE TIER (separate from the 15 general properties; a screen must ask for it with --tier degree)")
    print(df[["dataset"] + DEGREE_TIER].to_string(index=False))
    print(f"\n{len(df):3d} rows -> results/degree_features.csv")


# Defines command-line options (mirrors role_overlap.py / degree_rule.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure the DEGREE-specific property tier: tie structure, degree resolution, role-graph stability, divergence from psi.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to measure. Default: the degree corpus.')
    p.add_argument('--k', type=int, default=10, help='Top-K of the role graph. Default 10 (the locked setting).')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
