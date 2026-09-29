'''DEGREE STRUCTURAL CONSISTENCY: among nodes of the SAME degree, do their neighbourhoods actually look alike?'''
# Degree augmentation rests on one assumption - nodes with similar degree occupy similar structural roles - and that
# assumption is not always true: two degree-10 nodes can sit in completely different neighbourhoods, one wired to hubs
# and one to leaves. Every degree screen so far (paper_log 13, 17-18, 20-21, 24-27) tested GENERIC properties -
# clustering, skew, assortativity - none of which measures that assumption. This one measures it directly.
#   dsc        for each node, the histogram of its NEIGHBOURS' degrees, each neighbour recorded by its PERCENTILE RANK
#              in the degree distribution and dropped into the fixed ladder 0-10, 10-20 ... 90-100, so every graph has
#              the same 10 bins and none is excluded; group nodes by EXACT degree (build()'s own tie classes); mean
#              pairwise Jensen-Shannon SIMILARITY inside each group; node-weighted mean over groups.
#   dsc_between the same similarity between a group member and a node of a DIFFERENT degree.
#   dsc_sep    THE measure: dsc - dsc_between. Absolute within-group similarity is not enough - if same-degree nodes are
#              0.9 alike and different-degree nodes are also 0.9 alike, degree has explained nothing. Separation is what
#              the assumption "same degree => same role" actually claims.
#   DSC_adj    the same number with its chance floor removed, because raw within-group similarity rises with density and
#              size - the confound that disqualified four features in paper_log 20. The null is the CONFIGURATION MODEL,
#              computed analytically rather than by rewiring: in it a degree-d node draws d neighbours i.i.d. from the
#              degree-weighted endpoint distribution, so DSC_null is an expectation over two independent Multinomial(d, q)
#              draws. Same construction homophily_adjusted uses for the same reason.
# Read off the ORIGINAL graph, with NO labels, so a call is available before any encoder runs.
# The tier is deliberately its own list: a screen must ask for it by name, so no published result can move because this
# file exists (same discipline as degree_features.py).

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from experiments.characterize import GATES, loo_threshold, rho, threshold
from experiments.strategy_select import loo_majority, properties

CONSISTENCY_CSV = cfg.RESULTS_DIR / "degree_consistency.csv"
RULES_CSV = cfg.RESULTS_DIR / "degree_consistency_rules.csv"
VERDICTS = cfg.RESULTS_DIR / "degree_rule_corpus_verdicts.csv"

CONSISTENCY_TIER = ["dsc_sep", "dsc", "dsc_between", "dsc_adj", "dsc_null"]   # dsc_sep is THE measure; the rest are its parts, screened so the price is honest
BINS = 10            # global degree-quantile bins: fine enough to see neighbourhood shape, coarse enough that two
                     # low-degree nodes still share support (raw degree values leave them near-disjoint and JS saturates)
PAIRS = 500          # sampled pairs per degree group; a mean over 500 draws is far tighter than the between-graph spread
# Degree groups below this are EXCLUDED from every average. A degree-1 node's neighbour histogram is a one-hot vector, so
# the JS similarity of two of them is a coin flip on whether their single neighbour lands in the same percentile bin -
# noise, and on sparse graphs it carries 66-84% of the node weight. Deliberately a constant with no CLI flag: this is one
# declared correction, not a knob (user, 2026-09-07). group_coverage reports what it costs on each graph.
MIN_GROUP_DEGREE = 5
SEED = 42
CONFOUNDS = ["density", "nodes", "avg_degree"]   # what a within-group similarity could silently be measuring instead


# Jensen-Shannon SIMILARITY in [0,1]: 1 = identical neighbour-degree profiles, 0 = disjoint. Rows are probability vectors.
def js_similarity(P, Q):
    '''1 - JSD/ln2, row-wise; JSD uses natural log so its maximum is exactly ln 2.'''
    M = 0.5 * (P + Q)
    kl = lambda A, B: np.sum(np.where(A > 0, A * np.log(np.where(A > 0, A, 1.0) / np.where(B > 0, B, 1.0)), 0.0), axis=-1)
    return 1.0 - (0.5 * kl(P, M) + 0.5 * kl(Q, M)) / np.log(2.0)


# Mean pairwise similarity inside one group of histograms, sampling pairs when the group is large enough to make
# every pair expensive (a degree-1 class can hold thousands of nodes, i.e. millions of pairs).
def group_similarity(H, rng, pairs=PAIRS):
    '''Mean pairwise JS similarity over up to `pairs` sampled pairs of rows; nan for a group of one.'''
    n = len(H)
    if n < 2:
        return float("nan")
    if n * (n - 1) // 2 <= pairs:                                  # small group: every pair, no sampling
        i, j = np.triu_indices(n, 1)
    else:
        i = rng.integers(0, n, pairs)
        j = (i + rng.integers(1, n, pairs)) % n                    # j != i by construction, uniform over the other members
    return float(np.mean(js_similarity(H[i], H[j])))


# THE measurement. One pass over the edges builds every node's neighbour-degree histogram; the null needs no rewiring.
def consistency(G, bins=BINS, pairs=PAIRS, seed=SEED):
    '''(DSC, DSC_null, DSC_adj) plus coverage: the observed within-degree similarity and its configuration-model floor.'''
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    deg = np.array([d for _, d in G.degree], dtype=np.int64)
    # A neighbour's degree is recorded as its PERCENTILE RANK in this graph's degree distribution, then dropped into a
    # FIXED ladder of bins 0-10, 10-20, ... 90-100. Every graph therefore has exactly `bins` bins and the histograms are
    # directly comparable, with no graph excluded - the earlier version cut quantiles on raw degree, and on heavy-tailed
    # integer degrees those edges collided (bins_used ran 4-10 across the panel), which made DSC track its own resolution.
    # Midrank percentiles: a tied degree gets the average rank of its class, the standard percentile-rank definition.
    ud, uc = np.unique(deg, return_counts=True)
    pct = 100.0 * (np.cumsum(uc) - uc + (uc + 1) / 2.0) / len(deg)
    b = np.clip((np.interp(deg, ud, pct) / (100.0 / bins)).astype(int), 0, bins - 1)
    B = bins
    H = np.zeros((len(nodes), B))
    u = np.array([idx[x] for x, _ in G.edges], dtype=np.int64)
    v = np.array([idx[y] for _, y in G.edges], dtype=np.int64)
    np.add.at(H, (u, b[v]), 1.0)                                   # each endpoint counts the other's degree bin
    np.add.at(H, (v, b[u]), 1.0)
    H = H / np.maximum(H.sum(1, keepdims=True), 1e-12)             # isolated node -> all-zero row, excluded below
    # The configuration model's endpoint distribution: pick a random edge END, land in bin q_c with prob (degree mass in c)/2m.
    q = np.bincount(b, weights=deg.astype(float), minlength=B)
    q = q / q.sum()
    rng = np.random.default_rng(seed)
    order = np.argsort(deg, kind="stable")                         # degree groups are contiguous here, so the cross-group
    starts = {int(d): int(s) for d, s in zip(*np.unique(deg[order], return_index=True))}   # draw below is O(1) per sample
    obs, cross, null, w = [], [], [], []
    for d in np.unique(deg):
        if d < MIN_GROUP_DEGREE:
            continue                                               # too few neighbours for the histogram to be a distribution
        members = np.flatnonzero(deg == d)
        if len(members) < 2:
            continue                                               # a degree with one node carries no within-group evidence
        obs.append(group_similarity(H[members], rng, pairs))
        # The BETWEEN-group term the assumption actually needs: same-degree nodes must look alike RELATIVE to nodes of a
        # DIFFERENT degree. Without it a graph whose histograms are all alike scores high while degree explains nothing.
        # Pairs are (a member, a node outside this degree class), drawn by index-shifting in `order` so no rejection loop.
        s, nd, n = starts[int(d)], len(members), len(nodes)
        if nd < n:
            t = rng.integers(0, n - nd, pairs)
            outside = order[np.where(t < s, t, t + nd)]
            cross.append(float(np.mean(js_similarity(H[members[rng.integers(0, nd, pairs)]], H[outside]))))
        else:
            cross.append(float("nan"))                             # every node has this degree: no outside to compare to
        # Same statistic under the null: two independent degree-d nodes each draw d neighbours i.i.d. from q.
        m = min(pairs, max(len(members) * (len(members) - 1) // 2, 1))
        A = rng.multinomial(int(d), q, size=m) / float(d)
        C = rng.multinomial(int(d), q, size=m) / float(d)
        null.append(float(np.mean(js_similarity(A, C))))
        w.append(len(members))                                     # node-weighted: a 4000-node class counts more than a 2-node one
    w = np.array(w, dtype=float)
    dsc = float(np.average(obs, weights=w)) if len(w) else float("nan")
    dsc_null = float(np.average(null, weights=w)) if len(w) else float("nan")
    adj = (dsc - dsc_null) / (1.0 - dsc_null) if dsc_null == dsc_null and dsc_null < 1 else float("nan")
    ok = np.isfinite(cross)
    dsc_between = float(np.average(np.array(cross)[ok], weights=w[ok])) if ok.any() else float("nan")
    # SEPARATION: how much more alike two same-degree nodes are than a same-degree and a different-degree node.
    # ~0 means degree explains nothing about the neighbourhood, however high the raw within-group similarity is.
    dsc_sep = float(np.average((np.array(obs) - np.array(cross))[ok], weights=w[ok])) if ok.any() else float("nan")
    return {"dsc": round(dsc, 6), "dsc_between": round(dsc_between, 6), "dsc_sep": round(dsc_sep, 6),
            "dsc_null": round(dsc_null, 6), "dsc_adj": round(adj, 6),
            "bins_used": B,                                            # always `bins` now, kept so the column still means something
            # How many of the fixed bins any neighbour actually falls into: a DIAGNOSTIC of how much the ties concentrate
            # the mass, reported so a low-resolution graph is visible. Never screened as a predictor.
            "bins_occupied": int((H.sum(0) > 0).sum()),
            "degree_groups": len(w), "nodes_in_groups": int(w.sum()),
            "group_coverage": round(float(w.sum() / len(nodes)), 4)}


def measure(ds, bins=BINS, pairs=PAIRS, seed=SEED):
    '''One row: DSC and its null for one dataset, off the ORIGINAL graph.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    return {"dataset": ds, "bins": bins, "pairs": pairs, "seed": seed,
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), **consistency(G, bins, pairs, seed)}


# Reads what is already on disk and measures only what is missing (same contract as degree_features.degree_features).
def degree_consistency(datasets, bins=BINS, pairs=PAIRS, seed=SEED):
    '''The consistency tier for these datasets, cached in results/degree_consistency.csv.'''
    have = pd.read_csv(CONSISTENCY_CSV) if CONSISTENCY_CSV.exists() else pd.DataFrame(columns=["dataset", "seed"])
    miss = [d for d in datasets if d not in set(have.get("dataset", []))]
    if miss:
        rows = []
        for d in miss:
            rows.append(measure(d, bins, pairs, seed))
            print(f"measured {d}: within {rows[-1]['dsc']:.4f}  between {rows[-1]['dsc_between']:.4f}  "
                  f"SEPARATION {rows[-1]['dsc_sep']:+.4f}  ({rows[-1]['degree_groups']} groups, "
                  f"{100 * rows[-1]['group_coverage']:.1f}% of nodes)", flush=True)
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(CONSISTENCY_CSV, index=False)
    return have[have["dataset"].isin(datasets)].sort_values("dataset").reset_index(drop=True)


# THE target, as specified: degree wins ALONE among the graphs augmentation helps. Not "degree vs psi" - that only asks
# which of two candidates wins once both are on the table, and the deployable question is when stage 2 should pick degree.
def cells(datasets):
    '''One row per augmenting graph: the paired-band winner, whether DEGREE won alone, and the consistency tier.'''
    v = pd.read_csv(VERDICTS).query("band == 'paired'")[["dataset", "winner_signals", "beats_original"]]
    c = degree_consistency(datasets).drop(columns=["nodes", "edges"])   # properties() owns those two names; keep one copy
    d = v[v["dataset"].isin(datasets) & v["beats_original"]].merge(c, on="dataset")
    d = d.merge(properties(list(d["dataset"])), on="dataset")
    d["degree_sole"] = d["winner_signals"] == "degree"
    return d.sort_values("dataset").reset_index(drop=True)


# The price. A threshold search on 40 cells with 9 positives finds SOMETHING; the only honest question is how often it
# finds something this good when the labels mean nothing. Same instrument paper_log 26-27 used, so the rows compare.
def shuffle_price(x, y, draws=4000, seed=SEED):
    '''P(the identical search on shuffled labels does at least this well), by exception count.'''
    err = threshold(x, y)[2]
    rng = np.random.default_rng(seed)
    hits = sum(threshold(x, rng.permutation(y))[2] <= err for _ in range(draws))
    return round(hits / draws, 4), int(err)


def screen(d, predictors=CONSISTENCY_TIER):
    '''One row per predictor: does it separate the graphs where degree wins alone from the other augmenting graphs?'''
    rows = []
    for p in predictors:
        ok = d[p].notna().to_numpy()
        x, y = d.loc[ok, p].to_numpy(dtype=float), d.loc[ok, "degree_sole"].to_numpy()
        if len(x) < GATES["min_cells"] or not y.any() or y.all() or len(np.unique(x)) < 2:
            continue
        t, side, err, acc, lo, hi = threshold(x, y)
        l_acc, l_ok, l_n, _, _ = loo_threshold(x, y)
        l_maj, _ = loo_majority(x, y)
        major = round(float(max(y.mean(), 1 - y.mean())), 4)
        p_shuf, _ = shuffle_price(x, y)
        r = rho(pd.Series(x), pd.Series(y.astype(float)))[1]
        rows.append({
            "predictor": p, "tier": "consistency", "n_augmenting": int(len(x)), "n_degree": int(y.sum()),
            "degree_datasets": "|".join(sorted(d.loc[ok, "dataset"][y])),
            "degree_range": f"{x[y].min():.4f}-{x[y].max():.4f}", "rest_range": f"{x[~y].min():.4f}-{x[~y].max():.4f}",
            "spearman_rho": r, "threshold": t, "degree_side": side, "n_exceptions": err, "accuracy": acc,
            "interval_lo": lo, "interval_hi": hi,
            "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
            "majority_baseline": major, "loo_majority": l_maj,
            # The number that decides whether this is a finding: how often shuffled labels split at least this cleanly.
            "p_shuffle_as_good": p_shuf,
            **{f"rho_vs_{c}": rho(pd.Series(x), d.loc[ok, c].reset_index(drop=True))[1] for c in CONFOUNDS},
            "rule": f"use degree when {p} is {side} ({'>' if side == 'high' else '<'} {t})" if side else "",
            "credible": bool(r == r and abs(r) >= GATES["min_abs_rho"] and 0 <= err <= GATES["max_exceptions"]
                             and l_acc == l_acc and l_acc > major and l_acc > l_maj and p_shuf < 0.05),
        })
    return pd.DataFrame(rows).sort_values("p_shuffle_as_good").reset_index(drop=True)


def report(d, s):
    '''Print the measured consistency per graph, then the screen with its price and its confounds.'''
    print("\nDEGREE STRUCTURAL CONSISTENCY - measured on the ORIGINAL graph, no labels used")
    print(d[["dataset", "winner_signals", "degree_sole", "dsc_sep", "dsc", "dsc_between", "dsc_null", "dsc_adj",
             "degree_groups", "bins_occupied"]].to_string(index=False))
    print(f"\n{len(d)} augmenting graphs  |  {int(d['degree_sole'].sum())} name DEGREE alone "
          f"({', '.join(d['dataset'][d['degree_sole']]) or 'none'})")
    if s.empty:
        print("\nnothing screenable - too few cells, or the target has no variation")
        return
    print("\nSCREEN - does consistency separate the degree winners? (p_shuffle_as_good < 0.05 is the bar)")
    print(s[["predictor", "n_augmenting", "n_degree", "degree_range", "rest_range", "spearman_rho", "threshold",
             "degree_side", "n_exceptions", "loo_accuracy", "loo_folds", "majority_baseline", "loo_majority",
             "p_shuffle_as_good"] + [f"rho_vs_{c}" for c in CONFOUNDS] + ["credible"]].to_string(index=False))
    good = s[s["credible"]]
    print("\nCREDIBLE: " + ("none - every cut is at the rate shuffled labels supply" if good.empty else ""))
    for _, r in good.iterrows():
        print(f"  {r['rule']}   [interval {r['interval_lo']}-{r['interval_hi']}, rho {r['spearman_rho']}, "
              f"LOO {r['loo_correct']}/{r['loo_folds']} vs {r['majority_baseline']}, P(shuffle) {r['p_shuffle_as_good']}]")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d = cells(args.datasets)
    s = screen(d)
    if not s.empty:
        s.to_csv(RULES_CSV, index=False)
    report(d, s)
    print(f"\n{len(d):3d} cells  ->  results/{CONSISTENCY_CSV.name}, results/{RULES_CSV.name}")


# Defines command-line options (mirrors degree_features.py / degree_rule.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure degree structural consistency and screen it as a stage-2 degree rule.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to measure. Default: the degree corpus.')
    p.add_argument('--bins', type=int, default=BINS, help=f'Global degree-quantile bins for the neighbour histograms. Default {BINS}.')
    p.add_argument('--pairs', type=int, default=PAIRS, help=f'Sampled pairs per degree group. Default {PAIRS}.')
    p.add_argument('--seed', type=int, default=SEED, help=f'Sampling seed. Default {SEED}.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
