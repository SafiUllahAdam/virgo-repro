'''DEGREE ROLE DISPERSION: are same-degree nodes FAR APART in the original graph, i.e. does degree wire long-range shortcuts?'''
# A different mechanism from paper_log 29-30, which asked whether same-degree nodes are locally ALIKE and answered no on
# three descriptors. This asks whether degree augmentation wins for the opposite reason: not because it groups
# structurally identical nodes, but because it connects distant ones and adds useful long-range structure.
# The estimator needs no role graph. build() links a node to K SAMPLED members of its exact-degree tie class, so the
# distance distribution of same-degree pairs IS an unbiased estimate of the degree role graph's edge lengths.
#   DRD = mean shortest-path distance between SAME-DEGREE pairs / mean distance between RANDOM pairs
# Both terms come from the SAME breadth-first searches, so the ratio is paired and scale-free: 1.0 means degree classes
# are spread exactly like random pairs, > 1 means degree wires genuinely distant nodes, < 1 means its edges are local and
# redundant with the original graph. Raw hop counts are not comparable across graphs, which is why it is a ratio.
# TERMS, fixed BEFORE running (user, 2026-09-07), unchanged from paper_log 30 except the kill test:
#   1  ONE predictor - drd. No variants, no re-tuning of the source count or the sampling.
#   2  ONE form - the ratio above.
#   3  KILL TEST: the whole measurement is repeated under a second BFS-sampling seed, and drd is DEAD if the fitted side
#      flips between them. Declared up front, not applied after the fact.
#   4  direction PRE-DECLARED: high dispersion => degree. An inversion is a failure of the hypothesis, not a result.
#   5  the bar is P(shuffle as good) < 0.05. Nothing is frozen without a further pre-registered test.
# Unreachable same-degree pairs are excluded from the distance and reported as a separate DIAGNOSTIC column - counting
# them as "maximally distant" would re-import fragmentation, which Module 2 already screened.
# Read off the ORIGINAL graph with no labels, so a call would exist before any encoder runs.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from experiments.characterize import loo_threshold, rho, threshold
from experiments.degree_consistency import VERDICTS, shuffle_price
from experiments.strategy_select import loo_majority

DISPERSION_CSV = cfg.RESULTS_DIR / "degree_dispersion.csv"
RULES_CSV = cfg.RESULTS_DIR / "degree_dispersion_rules.csv"

# The SECOND DRD form (user, 2026-09-07), pre-declared and tested ONCE: instead of a mean distance, the share of
# same-degree pairs that are genuinely far. A mean compresses on a small-world graph where every distance sits between
# 2 and 6 hops; a tail probability does not. Screened as a RATIO against the same tail probability for random pairs,
# because the raw share would mostly measure diameter - the size confound that disqualified four features in section 20.
# The raw share is emitted beside it as a diagnostic and is never screened. If this fails, DRD stops.
FAR_HOPS = 3               # "far" = at least this many hops. A declared constant with no CLI flag, so it cannot be tuned
SOURCES = 300              # BFS sources per graph; uniform over non-isolated nodes, so groups are node-weighted for free
BATCH = 20                 # sources per dijkstra call: keeps the distance block small on a 400k-node graph
SEEDS = [42, 43]           # term 3: the second one is the KILL TEST, not a tuning knob
DECLARED_SIDE = "high"     # term 4: written down before the run - for both forms, more long-range => degree
PREDICTORS = ["drd_far", "drd"]   # drd_far is the new pre-declared test; drd is section 31's, re-reported for context


# One graph: BFS from a uniform sample of sources, and read both distance terms off the SAME searches.
def dispersion(G, sources=SOURCES, seed=42):
    '''(DRD, same-degree mean distance, random-pair mean distance, unreachable share, sources used).'''
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    deg = np.array([d for _, d in G.degree], dtype=np.int64)
    r = [idx[u] for u, v in G.edges] + [idx[v] for u, v in G.edges]
    c = [idx[v] for u, v in G.edges] + [idx[u] for u, v in G.edges]
    A = sp.csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(len(nodes), len(nodes)))
    live = np.flatnonzero(deg > 0)                                 # an isolated node reaches nothing and has no distance
    rng = np.random.default_rng(seed)
    pick = rng.choice(live, size=min(sources, len(live)), replace=False)
    same, rand, miss, far_s, far_r = [], [], [], [], []
    for lo in range(0, len(pick), BATCH):
        batch = pick[lo:lo + BATCH]
        D = dijkstra(A, unweighted=True, indices=batch)             # BFS distances, inf where unreachable
        for row, s in zip(D, batch):
            reach = np.isfinite(row)
            reach[s] = False                                        # a node is not its own pair
            peers = deg == deg[s]
            peers[s] = False
            if not peers.any() or not (peers & reach).any():
                continue                                            # a unique degree, or none of its class reachable
            same.append(float(row[peers & reach].mean()))
            rand.append(float(row[reach].mean()))
            far_s.append(float((row[peers & reach] >= FAR_HOPS).mean()))   # the tail form: share of pairs genuinely far
            far_r.append(float((row[reach] >= FAR_HOPS).mean()))
            miss.append(float(1.0 - (peers & reach).sum() / peers.sum()))
    if not same:
        return {}
    s_mean, r_mean = float(np.mean(same)), float(np.mean(rand))
    fs, fr = float(np.mean(far_s)), float(np.mean(far_r))
    return {"drd": round(s_mean / r_mean, 6) if r_mean else float("nan"),
            "same_dist": round(s_mean, 6), "rand_dist": round(r_mean, 6),
            "drd_far": round(fs / fr, 6) if fr else float("nan"),
            "far_same": round(fs, 6), "far_rand": round(fr, 6),          # raw shares: diagnostic only, never screened
            "unreachable_same": round(float(np.mean(miss)), 6), "n_sources": len(same)}


def measure(ds, sources=SOURCES):
    '''One row: DRD under both seeds - the second is the kill test, so it is measured with the first, never later.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    row = {"dataset": ds, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "sources": sources}
    for s in SEEDS:
        row.update({f"{k}_s{s}": v for k, v in dispersion(G, sources, s).items()})
    return row


# Reads what is already on disk and measures only what is missing (same contract as degree_consistency.py).
def dispersions(datasets, sources=SOURCES):
    '''DRD for these datasets, cached in results/degree_dispersion.csv.'''
    have = pd.read_csv(DISPERSION_CSV) if DISPERSION_CSV.exists() else pd.DataFrame(columns=["dataset"])
    miss = [d for d in datasets if d not in set(have.get("dataset", []))]
    if miss:
        rows = []
        for d in miss:
            rows.append(measure(d, sources))
            print(f"measured {d}: DRD {rows[-1]['drd_s42']:.4f}/{rows[-1]['drd_s43']:.4f}  "
                  f"FAR {rows[-1]['drd_far_s42']:.4f}/{rows[-1]['drd_far_s43']:.4f}  "
                  f"(raw far share {100 * rows[-1]['far_same_s42']:.1f}% vs random {100 * rows[-1]['far_rand_s42']:.1f}%)", flush=True)
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(DISPERSION_CSV, index=False)
    return have[have["dataset"].isin(datasets)].sort_values("dataset").reset_index(drop=True)


def cells(datasets):
    '''Augmenting graphs only, labelled by whether DEGREE won alone under the paired band.'''
    v = pd.read_csv(VERDICTS).query("band == 'paired'")[["dataset", "winner_signals", "beats_original"]]
    d = v[v["dataset"].isin(datasets) & v["beats_original"]].merge(dispersions(datasets), on="dataset")
    d["degree_sole"] = d["winner_signals"] == "degree"
    return d.sort_values("dataset").reset_index(drop=True)


def screen(d):
    '''The pre-declared test for each form, its price, and the second-seed kill test.'''
    rows = []
    for pred in PREDICTORS:
        a, b = f"{pred}_s{SEEDS[0]}", f"{pred}_s{SEEDS[1]}"
        ok = d[a].notna() & d[b].notna()
        x, y = d.loc[ok, a].to_numpy(dtype=float), d.loc[ok, "degree_sole"].to_numpy()
        if len(x) < 4 or not y.any() or y.all():
            continue
        t, side, err, acc, lo, hi = threshold(x, y)
        l_acc, l_ok, l_n, _, _ = loo_threshold(x, y)
        l_maj, _ = loo_majority(x, y)
        x2 = d.loc[ok, b].to_numpy(dtype=float)
        t2, side2 = threshold(x2, y)[:2]
        r, r2 = rho(pd.Series(x), pd.Series(y.astype(float)))[1], rho(pd.Series(x2), pd.Series(y.astype(float)))[1]
        p_shuf, _ = shuffle_price(x, y)
        major = round(float(max(y.mean(), 1 - y.mean())), 4)
        rows.append({
            "predictor": pred, "n_augmenting": int(len(x)), "n_degree": int(y.sum()),
            "degree_range": f"{x[y].min():.4f}-{x[y].max():.4f}", "rest_range": f"{x[~y].min():.4f}-{x[~y].max():.4f}",
            "spearman_rho": r, "threshold": t, "degree_side": side, "n_exceptions": err, "accuracy": acc,
            "interval_lo": lo, "interval_hi": hi,
            "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
            "majority_baseline": major, "loo_majority": l_maj, "p_shuffle_as_good": p_shuf,
            "matches_declared_side": bool(side == DECLARED_SIDE),
            "rho_seed2": r2, "side_seed2": side2, "threshold_seed2": t2,
            "self_rho_across_seeds": rho(pd.Series(x), pd.Series(x2))[1],
            "side_stable": bool(side == side2),
            "rule": f"use degree when {pred} is {side} ({'>' if side == 'high' else '<'} {t})" if side else "",
            "survives": bool(p_shuf < 0.05 and side == side2 and side == DECLARED_SIDE
                             and l_acc == l_acc and l_acc > major and l_acc > l_maj),
        })
    return pd.DataFrame(rows)


def report(d, s):
    '''Print the per-graph dispersion in both forms, then each pre-declared test and its kill test.'''
    print("\nDEGREE ROLE DISPERSION - how far apart the degree role graph's endpoints sit in the ORIGINAL graph")
    print(d[["dataset", "winner_signals", "degree_sole", "drd_s42", "drd_far_s42", "drd_far_s43",
             "far_same_s42", "far_rand_s42", "unreachable_same_s42"]].to_string(index=False))
    print(f"\n{len(d)} augmenting graphs  |  {int(d['degree_sole'].sum())} name DEGREE alone  "
          f"|  far = at least {FAR_HOPS} hops")
    if s.empty:
        print("\nnothing screenable")
        return
    print("\nPRE-DECLARED TEST: high => degree.  Bar: P(shuffle) < 0.05, side stable across BFS seeds, declared side.")
    print(s[["predictor", "n_augmenting", "n_degree", "degree_range", "rest_range", "spearman_rho", "threshold",
             "degree_side", "n_exceptions", "loo_accuracy", "majority_baseline", "loo_majority", "p_shuffle_as_good",
             "matches_declared_side", "rho_seed2", "side_seed2", "self_rho_across_seeds", "side_stable",
             "survives"]].to_string(index=False))
    good = s[s["survives"]]
    print("\nSURVIVES ALL FIVE TERMS: " + ("none" if good.empty else ""))
    for _, r in good.iterrows():
        print(f"  {r['rule']}   [rho {r['spearman_rho']}, LOO {r['loo_correct']}/{r['loo_folds']} vs "
              f"{r['majority_baseline']}, P(shuffle) {r['p_shuffle_as_good']}]")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d = cells(args.datasets)
    s = screen(d)
    if not s.empty:
        s.to_csv(RULES_CSV, index=False)
    report(d, s)
    print(f"\n{len(d):3d} cells  ->  results/{DISPERSION_CSV.name}, results/{RULES_CSV.name}")


# Defines command-line options (mirrors degree_consistency.py / dsc_descriptors.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure degree role dispersion and screen it as a stage-2 degree rule.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to measure. Default: the degree corpus.')
    p.add_argument('--sources', type=int, default=SOURCES, help=f'BFS sources per graph. Default {SOURCES}.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
