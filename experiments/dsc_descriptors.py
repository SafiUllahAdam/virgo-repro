'''The two UNTESTED DSC descriptors: does "same degree = same role" hold for CLUSTERING profile or a local role vector?'''
# paper_log 29 tested one descriptor - the neighbour-DEGREE histogram - in five mathematical forms, and it failed with a
# diagnosable reason (its sign flipped once the noisiest groups were removed). Two descriptors from the original list
# were never written: a node can have the identical neighbour-degree histogram as another and a completely different
# triangle structure, which is exactly the failure case the hypothesis describes and the degree histogram is blind to.
# This script tests those two, and re-runs the degree histogram beside them as the control.
# TERMS, fixed BEFORE running (user, 2026-09-07), because ten variants have already been spent on this question:
#   1  exactly two new descriptors, one run each - no re-tuning of bins, weighting or grouping
#   2  ONE form only: separation = within-group similarity - between-group similarity. No null variants.
#   3  each runs at ALL groups and at degree >= 5, and a descriptor is DEAD if its sign flips between them - the test
#      that killed the degree histogram, applied up front instead of after the fact
#   4  direction PRE-DECLARED: high separation => degree. An inversion is a failure of the hypothesis, not a result
#   5  the bar is P(shuffle as good) < 0.05 against 12 cumulative variants, not the 0.135 near-miss that came before
# Everything is read off the ORIGINAL graph with no labels, so a call would exist before any encoder runs.

import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from experiments.characterize import loo_threshold, rho, threshold
from experiments.degree_consistency import BINS, MIN_GROUP_DEGREE, PAIRS, SEED, VERDICTS, group_similarity, js_similarity, shuffle_price
from experiments.strategy_select import loo_majority

DESCRIPTORS_CSV = cfg.RESULTS_DIR / "dsc_descriptors.csv"
RULES_CSV = cfg.RESULTS_DIR / "dsc_descriptors_rules.csv"

# nbr_degree is the paper_log-29 descriptor, re-run here as the control so the three sit in one table.
DESCRIPTORS = ["nbr_degree", "clustering", "role_vector"]
PASSES = [1, MIN_GROUP_DEGREE]          # all groups, then the degree >= 5 correction; disagreement in SIGN kills a descriptor
DECLARED_SIDE = "high"                  # term 4: the hypothesis says high separation => degree. Written down before the run.


# Percentile-rank histogram of a per-node quantity over the fixed ladder 0-10 ... 90-100, exactly as paper_log 29 bins
# neighbour degrees: every graph gets the same `bins` bins, so no graph is excluded and the numbers are comparable.
def profile(G, idx, value, bins):
    '''Row-normalized histogram of each node's NEIGHBOURS' percentile rank in `value`.'''
    uv, uc = np.unique(value, return_counts=True)
    pct = 100.0 * (np.cumsum(uc) - uc + (uc + 1) / 2.0) / len(value)
    b = np.clip((np.interp(value, uv, pct) / (100.0 / bins)).astype(int), 0, bins - 1)
    H = np.zeros((len(value), bins))
    u = np.array([idx[x] for x, _ in G.edges], dtype=np.int64)
    v = np.array([idx[y] for _, y in G.edges], dtype=np.int64)
    np.add.at(H, (u, b[v]), 1.0)
    np.add.at(H, (v, b[u]), 1.0)
    return H / np.maximum(H.sum(1, keepdims=True), 1e-12)


# The three descriptors. 'hist' rows are probability vectors compared by Jensen-Shannon; 'vec' rows are z-normalized
# features compared by cosine, rescaled to [0,1] so both similarity scales read the same way.
def descriptor(G, name, clus, bins=BINS):
    '''(matrix, kind, degree) for one descriptor, all measured on the ORIGINAL graph without labels.'''
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    deg = np.array([d for _, d in G.degree], dtype=np.int64)
    u = np.array([idx[x] for x, _ in G.edges], dtype=np.int64)
    v = np.array([idx[y] for _, y in G.edges], dtype=np.int64)
    if name == "nbr_degree":
        return profile(G, idx, deg.astype(float), bins), "hist", deg
    if name == "clustering":                                       # how densely a node's NEIGHBOURS are themselves clustered
        return profile(G, idx, clus, bins), "hist", deg
    # role_vector: what a node looks like BEYOND its degree. Degree itself is excluded on purpose - it is constant inside
    # a degree group, so including it would inflate within-group similarity for free.
    s1 = np.zeros(len(nodes)); s2 = np.zeros(len(nodes)); mx = np.zeros(len(nodes))
    d = deg.astype(float)
    np.add.at(s1, u, d[v]); np.add.at(s1, v, d[u])
    np.add.at(s2, u, d[v] ** 2); np.add.at(s2, v, d[u] ** 2)
    np.maximum.at(mx, u, d[v]); np.maximum.at(mx, v, d[u])
    k = np.maximum(d, 1.0)
    mean_nd = s1 / k
    sd_nd = np.sqrt(np.maximum(s2 / k - mean_nd ** 2, 0.0))
    X = np.c_[clus, np.log1p(mean_nd), np.log1p(sd_nd), np.log1p(mx)]
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return X, "vec", deg


def similarity(A, B, kind):
    '''Row-wise similarity in [0,1]: Jensen-Shannon for histograms, rescaled cosine for feature vectors.'''
    if kind == "hist":
        return js_similarity(A, B)
    n = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return 0.5 * (1.0 + np.sum(n(A) * n(B), axis=1))


# THE one form (term 2): within-group similarity minus between-group similarity, node-weighted over exact-degree groups.
def separation(M, kind, deg, min_degree, pairs=PAIRS, seed=SEED):
    '''(within, between, separation, coverage) for one descriptor at one minimum group degree.'''
    n = len(deg)
    rng = np.random.default_rng(seed)
    order = np.argsort(deg, kind="stable")
    starts = {int(x): int(s) for x, s in zip(*np.unique(deg[order], return_index=True))}
    win, btw, w = [], [], []
    for x in np.unique(deg):
        if x < min_degree:
            continue
        members = np.flatnonzero(deg == x)
        nd = len(members)
        if nd < 2 or nd == n:
            continue                                               # no within pair, or no outside to compare against
        if kind == "hist":
            win.append(group_similarity(M[members], rng, pairs))
        else:                                                      # same sampling, cosine instead of JS
            i = rng.integers(0, nd, pairs)
            j = (i + rng.integers(1, nd, pairs)) % nd
            win.append(float(np.mean(similarity(M[members[i]], M[members[j]], kind))))
        t = rng.integers(0, n - nd, pairs)
        outside = order[np.where(t < starts[int(x)], t, t + nd)]
        btw.append(float(np.mean(similarity(M[members[rng.integers(0, nd, pairs)]], M[outside], kind))))
        w.append(nd)
    if not w:
        return float("nan"), float("nan"), float("nan"), 0.0
    w = np.array(w, dtype=float)
    within = float(np.average(win, weights=w))
    between = float(np.average(btw, weights=w))
    return round(within, 6), round(between, 6), round(within - between, 6), round(float(w.sum() / n), 4)


def measure(ds, bins=BINS, pairs=PAIRS, seed=SEED):
    '''One row: separation for all three descriptors, at every pass.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    clus = np.array(list(nx.clustering(G).values()), dtype=float)   # ONE pass, shared by the clustering and role-vector descriptors
    row = {"dataset": ds, "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
    for name in DESCRIPTORS:
        M, kind, deg = descriptor(G, name, clus, bins)
        for p in PASSES:
            wi, bt, sp, cov = separation(M, kind, deg, p, pairs, seed)
            tag = "all" if p == 1 else f"d{p}"
            row[f"{name}_within_{tag}"], row[f"{name}_between_{tag}"] = wi, bt
            row[f"{name}_sep_{tag}"], row[f"{name}_cov_{tag}"] = sp, cov
    return row


# Reads what is already on disk and measures only what is missing (same contract as degree_consistency.py).
def descriptors(datasets, bins=BINS, pairs=PAIRS, seed=SEED):
    '''The three descriptors' separations for these datasets, cached in results/dsc_descriptors.csv.'''
    have = pd.read_csv(DESCRIPTORS_CSV) if DESCRIPTORS_CSV.exists() else pd.DataFrame(columns=["dataset"])
    miss = [d for d in datasets if d not in set(have.get("dataset", []))]
    if miss:
        rows = []
        for d in miss:
            rows.append(measure(d, bins, pairs, seed))
            print("measured " + d + ": " + "  ".join(
                f"{n} {rows[-1][f'{n}_sep_all']:+.4f}/{rows[-1][f'{n}_sep_d{MIN_GROUP_DEGREE}']:+.4f}" for n in DESCRIPTORS), flush=True)
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(DESCRIPTORS_CSV, index=False)
    return have[have["dataset"].isin(datasets)].sort_values("dataset").reset_index(drop=True)


def cells(datasets):
    '''Augmenting graphs only, labelled by whether DEGREE won alone under the paired band.'''
    v = pd.read_csv(VERDICTS).query("band == 'paired'")[["dataset", "winner_signals", "beats_original"]]
    d = v[v["dataset"].isin(datasets) & v["beats_original"]].merge(descriptors(datasets), on="dataset")
    d["degree_sole"] = d["winner_signals"] == "degree"
    return d.sort_values("dataset").reset_index(drop=True)


# Term 3: the all-groups pass is the hypothesis, the degree>=5 pass is the KILL TEST. A descriptor whose separation
# changes sign, or whose fitted side changes, is not measuring one thing and is reported dead whatever its p-value.
def screen(d):
    '''One row per descriptor: the pre-declared test, its price, and whether it survives the degree>=5 pass.'''
    rows = []
    for name in DESCRIPTORS:
        a, b = f"{name}_sep_all", f"{name}_sep_d{MIN_GROUP_DEGREE}"
        ok = d[a].notna() & d[b].notna()
        x, y = d.loc[ok, a].to_numpy(dtype=float), d.loc[ok, "degree_sole"].to_numpy()
        if len(x) < 4 or not y.any() or y.all():
            continue
        t, side, err, acc, lo, hi = threshold(x, y)
        l_acc, l_ok, l_n, _, _ = loo_threshold(x, y)
        l_maj, _ = loo_majority(x, y)
        x5 = d.loc[ok, b].to_numpy(dtype=float)
        side5 = threshold(x5, y)[1]
        r, r5 = rho(pd.Series(x), pd.Series(y.astype(float)))[1], rho(pd.Series(x5), pd.Series(y.astype(float)))[1]
        p_shuf, _ = shuffle_price(x, y)
        rows.append({
            "descriptor": name, "n_augmenting": int(len(x)), "n_degree": int(y.sum()),
            "spearman_rho": r, "threshold": t, "degree_side": side, "n_exceptions": err,
            "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
            "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 4), "loo_majority": l_maj,
            "p_shuffle_as_good": p_shuf,
            # term 4: was the direction the one declared before the run?
            "matches_declared_side": bool(side == DECLARED_SIDE),
            # term 3: the kill test, decided on the degree>=5 pass
            "rho_d5": r5, "side_d5": side5, "self_rho_across_passes": rho(pd.Series(x), pd.Series(x5))[1],
            "sign_stable": bool(side == side5 and r == r and r5 == r5 and np.sign(r) == np.sign(r5)),
            "rule": f"use degree when {name} separation is {side} ({'>' if side == 'high' else '<'} {t})" if side else "",
            # term 5: below 0.05 AND stable AND in the declared direction. Anything else is reported, never promoted.
            "survives": bool(p_shuf < 0.05 and side == side5 and side == DECLARED_SIDE
                             and l_acc == l_acc and l_acc > max(y.mean(), 1 - y.mean())),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("p_shuffle_as_good").reset_index(drop=True) if not out.empty else out


def report(d, s):
    '''Print the per-graph separations, then the three pre-declared tests and their kill test.'''
    cols = ["dataset", "winner_signals", "degree_sole"] + [f"{n}_sep_{t}" for n in DESCRIPTORS
                                                           for t in ("all", f"d{MIN_GROUP_DEGREE}")]
    print("\nSEPARATION per descriptor - within-group minus between-group similarity, ORIGINAL graph, no labels")
    print(d[cols].to_string(index=False))
    print(f"\n{len(d)} augmenting graphs  |  {int(d['degree_sole'].sum())} name DEGREE alone")
    if s.empty:
        print("\nnothing screenable")
        return
    print(f"\nPRE-DECLARED TEST: high separation => degree.  Bar: P(shuffle) < 0.05, sign stable across passes, declared side.")
    print(s[["descriptor", "n_degree", "spearman_rho", "threshold", "degree_side", "n_exceptions", "loo_accuracy",
             "majority_baseline", "p_shuffle_as_good", "matches_declared_side", "rho_d5", "side_d5",
             "self_rho_across_passes", "sign_stable", "survives"]].to_string(index=False))
    good = s[s["survives"]]
    print("\nSURVIVES ALL FIVE TERMS: " + ("none" if good.empty else ""))
    for _, r in good.iterrows():
        print(f"  {r['rule']}   [rho {r['spearman_rho']}, LOO {r['loo_correct']}/{r['loo_folds']} vs "
              f"{r['majority_baseline']}, P(shuffle) {r['p_shuffle_as_good']}, stable across passes]")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d = cells(args.datasets)
    s = screen(d)
    if not s.empty:
        s.to_csv(RULES_CSV, index=False)
    report(d, s)
    print(f"\n{len(d):3d} cells  ->  results/{DESCRIPTORS_CSV.name}, results/{RULES_CSV.name}")


# Defines command-line options (mirrors degree_consistency.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Test the two untested DSC descriptors: clustering profile and local role vector.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to measure. Default: the degree corpus.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
