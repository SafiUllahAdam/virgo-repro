'''How much of a graph does a FIXED K=10 role graph overwrite? A confound GATE first, a screen only if it passes.'''
# paper_log 33 measured that role augmentation helps high-degree endpoints and hurts low-degree ones, and proposed why:
# K is fixed, so ten role edges outnumber a degree-1 node's single real edge ten to one and OVERWRITE its neighbourhood,
# while for a degree-100 node they are a small perturbation. This script measures that relative perturbation directly.
#   rewrite_share  the share of nodes with degree < K - the fraction of the graph whose neighbourhood the augmentation
#                  replaces rather than extends. This is the quantity section 33's mechanism implicates, and it is NOT a
#                  mean of anything.
#   k_over_degree  mean(K / degree) over non-isolated nodes. Reported because it is the literal form of the idea, and
#                  flagged because it equals K divided by the HARMONIC mean degree - i.e. another summary of the degree
#                  sequence, in the same family as avg_degree, density, degree_gini, degree_skew and distinct_degrees,
#                  all of which were screened and failed in sections 25-27.
# THE GATE, and it runs BEFORE any screening. Section 34 died exactly this way: a new measure that correlated +0.79 with
# avg_degree scored identically to avg_degree (9 exceptions, P 0.72 both). So a candidate here must clear |rho| < 0.7
# against every already-screened degree-distribution property, or it is the sparsity axis again under a new name and no
# test is spent on it.
# THE DIRECTION must be declared: --declare-side is REQUIRED before the screen runs at all. "The relative change is
# suitable" is non-monotone and a single cut cannot express it, so a side has to be chosen in advance, not read off the
# numbers. Without the flag this script measures and gates, and stops.
# Read off the ORIGINAL graph with no labels and no role graph.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from experiments.characterize import loo_threshold, rho, threshold
from experiments.degree_consistency import VERDICTS, shuffle_price
from experiments.strategy_select import loo_majority, properties

REWRITE_CSV = cfg.RESULTS_DIR / "degree_rewrite_share.csv"
GATE_CSV = cfg.RESULTS_DIR / "degree_rewrite_gate.csv"
RULES_CSV = cfg.RESULTS_DIR / "degree_rewrite_rules.csv"

K = 10                                     # the locked top-K; the whole point is that it is FIXED
CANDIDATES = ["rewrite_share", "k_over_degree"]
# Every already-screened summary of the degree sequence. A candidate correlating with any of them at this level is that
# property wearing a new name, and section 34 showed such a measure scores exactly what the property it proxies scores.
SCREENED = ["avg_degree", "density", "degree_gini", "degree_skew", "distinct_degrees"]
MAX_CONFOUND_RHO = 0.7


def measure(ds, k=K):
    '''One row: how much of this graph a fixed-K role graph overwrites, plus the degree-sequence summaries it must clear.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    deg = np.array([d for _, d in G.degree], dtype=float)
    live = deg > 0
    return {"dataset": ds, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "k": k,
            # an isolated node is overwritten completely, so it counts here; it has no K/degree, so it does not count there
            "rewrite_share": round(float((deg < k).mean()), 6),
            "k_over_degree": round(float(np.mean(k / deg[live])), 6),
            "k_over_degree_median": round(float(np.median(k / deg[live])), 6),
            "isolate_share": round(float((~live).mean()), 6),
            "distinct_degrees": int(len(np.unique(deg)))}


# Reads what is already on disk and measures only what is missing (same contract as degree_consistency.py).
def rewrite(datasets, k=K):
    '''The overwrite measures per dataset, cached in results/degree_rewrite_share.csv.'''
    have = pd.read_csv(REWRITE_CSV) if REWRITE_CSV.exists() else pd.DataFrame(columns=["dataset"])
    miss = [d for d in datasets if d not in set(have.get("dataset", []))]
    if miss:
        rows = [measure(d, k) for d in miss]
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(REWRITE_CSV, index=False)
    return have[have["dataset"].isin(datasets)].sort_values("dataset").reset_index(drop=True)


def cells(datasets, k=K):
    '''Augmenting graphs only, labelled by whether DEGREE won alone, joined to the already-screened properties.'''
    v = pd.read_csv(VERDICTS).query("band == 'paired'")[["dataset", "winner_signals", "beats_original"]]
    d = v[v["dataset"].isin(datasets) & v["beats_original"]].merge(rewrite(datasets, k), on="dataset")
    d = d.merge(properties(list(d["dataset"])), on="dataset", suffixes=("", "_p"))
    d["degree_sole"] = d["winner_signals"] == "degree"
    return d.sort_values("dataset").reset_index(drop=True)


# THE GATE. Nothing is screened until this passes, because a proxy scores whatever the property it proxies scores.
def gate(d):
    '''Per candidate: its correlation with every already-screened degree-sequence property, and whether it clears the bar.'''
    rows = []
    for c in CANDIDATES:
        r = {"candidate": c, "n": len(d)}
        for p in SCREENED:
            r[f"rho_vs_{p}"] = rho(d[c], d[p])[1]
        worst = max(abs(r[f"rho_vs_{p}"]) for p in SCREENED if r[f"rho_vs_{p}"] == r[f"rho_vs_{p}"])
        r["worst_abs_rho"] = round(float(worst), 4)
        r["worst_against"] = max(SCREENED, key=lambda p: abs(r[f"rho_vs_{p}"]) if r[f"rho_vs_{p}"] == r[f"rho_vs_{p}"] else -1)
        r["passes"] = bool(worst < MAX_CONFOUND_RHO)
        rows.append(r)
    return pd.DataFrame(rows)


# Only reached when the gate passes AND a side was declared on the command line.
def screen(d, candidate, side):
    '''The declared test for one candidate that cleared the gate.'''
    x, y = d[candidate].to_numpy(dtype=float), d["degree_sole"].to_numpy()
    t, fitted, err, acc, lo, hi = threshold(x, y)
    l_acc, l_ok, l_n, _, _ = loo_threshold(x, y)
    l_maj, _ = loo_majority(x, y)
    p_shuf, _ = shuffle_price(x, y)
    major = round(float(max(y.mean(), 1 - y.mean())), 4)
    return pd.DataFrame([{
        "candidate": candidate, "declared_side": side, "n": int(len(x)), "n_degree": int(y.sum()),
        "spearman_rho": rho(pd.Series(x), pd.Series(y.astype(float)))[1],
        "threshold": t, "fitted_side": fitted, "matches_declared_side": bool(fitted == side),
        "n_exceptions": err, "accuracy": acc, "interval_lo": lo, "interval_hi": hi,
        "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
        "majority_baseline": major, "loo_majority": l_maj, "p_shuffle_as_good": p_shuf,
        "rule": f"use degree when {candidate} is {fitted} ({'>' if fitted == 'high' else '<'} {t})",
        "survives": bool(p_shuf < 0.05 and fitted == side and l_acc == l_acc and l_acc > major and l_acc > l_maj),
    }])


def report(d, g, s):
    '''Print what the augmentation overwrites, then the gate, then the screen if one was authorised.'''
    print(f"\nHOW MUCH A FIXED K={K} ROLE GRAPH OVERWRITES - share of nodes with degree < K, ORIGINAL graph, no labels")
    print(d[["dataset", "winner_signals", "degree_sole", "rewrite_share", "k_over_degree", "k_over_degree_median",
             "avg_degree", "distinct_degrees"]].to_string(index=False))
    w, o = d[d["degree_sole"]], d[~d["degree_sole"]]
    print(f"\n{len(d)} augmenting graphs  |  {len(w)} name DEGREE alone")
    print(f"  degree winners: mean rewrite_share {w['rewrite_share'].mean():.4f}  k_over_degree {w['k_over_degree'].mean():.4f}")
    print(f"  the rest      : mean rewrite_share {o['rewrite_share'].mean():.4f}  k_over_degree {o['k_over_degree'].mean():.4f}")
    print(f"\nCONFOUND GATE - a candidate must clear |rho| < {MAX_CONFOUND_RHO} against every already-screened degree-sequence property")
    print(g[["candidate"] + [f"rho_vs_{p}" for p in SCREENED] + ["worst_abs_rho", "worst_against", "passes"]].to_string(index=False))
    if s.empty:
        blocked = [r["candidate"] for _, r in g.iterrows() if not r["passes"]]
        print("\nSCREEN NOT RUN: " + (f"{', '.join(blocked)} failed the gate" if blocked else
                                      "no side declared - pass --declare-side high|low to authorise the test"))
        return
    print("\nDECLARED TEST")
    print(s[["candidate", "declared_side", "n_degree", "spearman_rho", "threshold", "fitted_side",
             "matches_declared_side", "n_exceptions", "loo_accuracy", "majority_baseline", "loo_majority",
             "p_shuffle_as_good", "survives"]].to_string(index=False))


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d = cells(args.datasets, args.k)
    g = gate(d)
    g.to_csv(GATE_CSV, index=False)
    ok = [r["candidate"] for _, r in g.iterrows() if r["passes"]]
    s = pd.concat([screen(d, c, args.declare_side) for c in ok], ignore_index=True) if (ok and args.declare_side) else pd.DataFrame()
    if not s.empty:
        s.to_csv(RULES_CSV, index=False)
    report(d, g, s)
    print(f"\n{len(d):3d} augmenting graphs  ->  results/{REWRITE_CSV.name}, results/{GATE_CSV.name}")


# Defines command-line options (mirrors degree_dispersion.py / psi_refinement.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure how much a fixed-K role graph overwrites, gate it against the already-screened degree properties, and screen only if a side is declared.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to measure. Default: the degree corpus.')
    p.add_argument('--k', type=int, default=K, help=f'The fixed top-K under study. Default {K} (the locked setting).')
    p.add_argument('--declare-side', choices=['high', 'low'], default=None,
                   help="REQUIRED to run the screen: which side of the cut means 'use degree'. Declaring it after seeing "
                        "the gate output is fitting, not testing, so the default is to gate and stop.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
