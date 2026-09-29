'''Stage 2 without a screen: score each role graph with an UNTRAINED probe and let the probe pick the signal.'''
# Every degree attempt so far (paper_log 13, 17-18, 20-21, 24-27) asked the same question - which SCALAR GRAPH PROPERTY
# predicts a degree win - and every one returned what shuffled labels return. 27 closed that route and named the only
# opening left: "a feature space that is not hand-designed graph properties at all". This is that feature space.
# The idea is to stop describing the graph and instead SIMULATE the comparison the encoder makes, at zero training cost:
#   1  split the LP train graph again, 70:30 -> an INNER train graph and inner held-out edges (leakage-free: the real
#      test split is never read, so a probe call can be written before the encoder runs, exactly as stage 1 requires)
#   2  build each signal's role graph on the inner train graph, with the SAME VirtualGraph.build() the study uses
#   3  embed with the encoder's own four structural features, mean-aggregated over the role graph for `layers` hops and
#      NOT trained - GraphSAGE at initialisation, identity weights
#   4  score the inner held-out edges by cosine, exactly as eval/linkpred does
# The contrast probe(degree) - probe(psi) is then a per-graph number that needs no labels, no encoder and no threshold.
# What it costs, stated because it is the same cost that retired the Module-4 gate: the role graphs must be BUILT (never
# trained), so unlike homophily this is not readable off a dataset card. Building is not training (CLAUDE.md 7).
# Status: FITTED ON NOTHING. The sign rule has no free parameter; the band is measured from repeated inner splits, not
# from any outcome. Numbers and their price are in docs/paper_log.md 28.

import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

import pandas as pd

from virgo import config as cfg
from virgo import graph_io
from virgo.virtual_graph import VirtualGraph
from virgo.data.prepare_linkpred import sample_non_edges, split_edges

PROBE_CSV = cfg.RESULTS_DIR / "role_probe.csv"
CALLS_CSV = cfg.RESULTS_DIR / "role_probe_calls.csv"

SIMS = ["degree", "psi", "centrality", "original"]      # the signals plus the do-not-augment control
INNER_SEED = 1234                                       # the inner split's seed; deliberately NOT a study seed, so it can never coincide with a training seed
LAYERS = 2                                              # GNN_PARAMS["layers"]: the probe mirrors the encoder's depth
MAX_PAIRS = 40_000                                      # cap per class; the AUC is already stable well below this
MIN_INNER_TEST = 200                                    # fewer inner held-out edges than this and the probe reports nothing rather than a noisy AUC
# Two independent inner splits move the contrast by ~0.0032 (sd over three seeds, 14 datasets), so anything inside twice
# that is a draw the instrument cannot call. Measured from re-splitting, never from a label - it is a noise floor, not a fit.
BAND = 0.0064


# Untrained GraphSAGE: the encoder's own z-normalized features, mean-aggregated over the role graph, nothing learned.
def embed(G, V, layers=LAYERS):
    '''Mean-aggregate the four structural features over the role graph for `layers` hops; return L2-normalized rows.'''
    nodes = list(G.nodes)
    vg = VirtualGraph(G)
    pn, ps = vg.signatures('psi')
    psi = dict(zip(pn, ps[:, 0]))
    deg, ev, clus = vg.core.degree_node(), vg.core.eigenvector_centrality(), nx.clustering(G)
    X = np.array([[deg[n], ev[n], float(psi[n]), clus[n]] for n in nodes])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)                       # same normalization the encoder applies
    i = {n: j for j, n in enumerate(nodes)}
    r = [i[u] for u, v in V.edges] + [i[v] for u, v in V.edges]
    c = [i[v] for u, v in V.edges] + [i[u] for u, v in V.edges]
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(nodes), len(nodes)))
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1                                                 # an isolated node keeps its own features, never divides by zero
    D = sp.diags(1.0 / d)
    Z = X
    for _ in range(layers):
        Z = np.hstack([Z, D @ (A @ Z)])                           # concat, as SAGEConv concatenates self and neighbourhood
    return i, Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


# One dataset: split the train graph again, build each role graph on the inner train graph, score the inner held-out edges.
def probe(ds, k=10, inner_seed=INNER_SEED, layers=LAYERS):
    '''One row: untrained inner-split AUC per signal, plus the contrasts the stage-2 call reads.'''
    G = graph_io.load_graph(cfg.LP_SPLITS_VG / ds / "seed_42" / "train.edgelist")
    tr, te = split_edges(G, cfg.REPRO["linkpred_test_frac"], inner_seed)
    # A near-forest keeps almost every edge in the spanning forest, so the inner split can hold out too few pairs to
    # score. Report the graph as unprobeable rather than returning an AUC computed on a handful of edges.
    if len(te) < MIN_INNER_TEST:
        return {"dataset": ds, "k": k, "inner_seed": inner_seed, "nodes": G.number_of_nodes(), "inner_edges": len(tr),
                "blocked_by": f"only {len(te)} inner held-out edges (< {MIN_INNER_TEST}) - the graph is near-acyclic"}
    neg = sample_non_edges(G, len(te), inner_seed, {frozenset(e) for e in tr + te})
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    H.add_edges_from(tr)                                          # the inner TRAIN graph: role graphs see only this
    rng = np.random.default_rng(0)
    m = min(len(te), len(neg), MAX_PAIRS)
    pairs = [[te[j] for j in rng.choice(len(te), m, replace=False)],
             [neg[j] for j in rng.choice(len(neg), m, replace=False)]]
    y = np.r_[np.ones(m), np.zeros(m)]
    row = {"dataset": ds, "k": k, "inner_seed": inner_seed, "nodes": G.number_of_nodes(), "inner_edges": len(tr)}
    for sim in SIMS:
        i, E = embed(H, VirtualGraph(H, seed=cfg.REPRO["seed"]).build(sim, k), layers)
        s = [(E[[i[u] for u, v in p]] * E[[i[v] for u, v in p]]).sum(1) for p in pairs]
        row[f"probe_{sim}"] = round(float(roc_auc_score(y, np.r_[s[0], s[1]])), 6)
    row["probe_degree_psi"] = round(row["probe_degree"] - row["probe_psi"], 6)
    row["probe_degree_rival"] = round(row["probe_degree"] - max(row["probe_psi"], row["probe_centrality"]), 6)
    row["probe_augment"] = round(max(row[f"probe_{s}"] for s in SIMS[:3]) - row["probe_original"], 6)
    return row


# Reads what is already on disk and measures only what is missing (same contract as degree_features.degree_features).
def role_probe(datasets, k=10, inner_seeds=(INNER_SEED,)):
    '''The probe rows for these datasets x inner seeds, cached in results/role_probe.csv.'''
    have = pd.read_csv(PROBE_CSV) if PROBE_CSV.exists() else pd.DataFrame(columns=["dataset", "inner_seed"])
    done = set(zip(have.get("dataset", []), have.get("inner_seed", [])))
    miss = [(d, s) for s in inner_seeds for d in datasets if (d, s) not in done]
    if miss:
        rows = []
        for d, s in miss:
            rows.append(probe(d, k, s))
            print(f"probed {d} (inner seed {s}): " + (rows[-1]["blocked_by"] if "blocked_by" in rows[-1] else
                  f"degree-psi {rows[-1]['probe_degree_psi']:+.4f}  augment {rows[-1]['probe_augment']:+.4f}"), flush=True)
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(PROBE_CSV, index=False)
    return have[have["dataset"].isin(datasets) & have["inner_seed"].isin(inner_seeds)].sort_values(["dataset", "inner_seed"]).reset_index(drop=True)


# The call itself. No threshold is fitted: the sign decides, and the band is the instrument's own noise floor.
def calls(df, band=BAND):
    '''Per dataset: the mean contrast over inner seeds, its spread, and the signal / augment call the probe makes.'''
    df = df[df["probe_degree_psi"].notna()]                        # a blocked graph carries no contrast to average
    g = df.groupby("dataset").agg(inner_seeds=("inner_seed", "nunique"),
                                  degree_psi=("probe_degree_psi", "mean"), degree_psi_sd=("probe_degree_psi", "std"),
                                  degree_rival=("probe_degree_rival", "mean"),
                                  augment=("probe_augment", "mean")).reset_index()
    g["signal_call"] = np.where(g["degree_psi"] > band, "degree",
                                np.where(g["degree_psi"] < -band, "psi", "undetermined"))
    g["augment_call"] = np.where(g["augment"] > 0, "augment", "keep original")
    return g.round(6)


# Scores the calls against the paired-band winner labels, when they exist. Reporting only - nothing here is refitted.
def score(g, verdicts=cfg.RESULTS_DIR / "degree_rule_corpus_verdicts.csv"):
    '''Join the probe calls to the labels the seeds actually name, restricted to the cells the contrast is defined on.'''
    if not Path(verdicts).exists():
        return pd.DataFrame()
    v = pd.read_csv(verdicts).query("band == 'paired'")[["dataset", "winner_signals", "beats_original"]]
    d = g.merge(v, on="dataset")
    d = d[d["beats_original"] & d["winner_signals"].isin(["degree", "psi"])].copy()
    d["correct"] = np.where(d["signal_call"] == "undetermined", None, d["signal_call"] == d["winner_signals"])
    return d


def report(g, s):
    '''Print the contrasts, the calls, and - where labels exist - how often the call was right.'''
    print("\nUNTRAINED ROLE-GRAPH PROBE - inner-split AUC contrasts (no encoder was run)")
    print(g[["dataset", "inner_seeds", "degree_psi", "degree_psi_sd", "degree_rival", "augment",
             "signal_call", "augment_call"]].to_string(index=False))
    print(f"\nsignal calls: " + "  ".join(f"{k} {v}" for k, v in g["signal_call"].value_counts().items()))
    if s.empty:
        print("no paired verdicts on disk -> the calls stand unscored, which is what a pre-registration looks like")
        return
    c = [b for b in s["correct"] if isinstance(b, (bool, np.bool_))]
    print(f"\nSCORED against the paired-band labels ({len(s)} degree-or-psi cells, {len(s) - len(c)} undetermined)")
    print(s[["dataset", "winner_signals", "degree_psi", "signal_call", "correct"]].to_string(index=False))
    print(f"  {sum(c)}/{len(c)} correct   |   a constant predictor scores {max(s['winner_signals'].value_counts()) }/{len(s)} on the same cells")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = role_probe(args.datasets, args.k, tuple(args.inner_seeds))
    g = calls(df, args.band)
    s = score(g)
    g.to_csv(CALLS_CSV, index=False)
    report(g, s)
    print(f"\n{len(df):3d} probe rows -> results/{PROBE_CSV.name}   |   {len(g)} calls -> results/{CALLS_CSV.name}")


# Defines command-line options (mirrors degree_features.py / tie_break.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Pick the stage-2 signal with an untrained role-graph probe instead of a property screen.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to probe. Default: the degree corpus.')
    p.add_argument('--k', type=int, default=10, help='Top-K of the role graph. Default 10 (the locked setting).')
    p.add_argument('--inner-seeds', type=int, nargs='+', default=[INNER_SEED],
                   help='Inner-split seeds. Pass several to measure the contrast\'s own spread; the call uses their mean.')
    p.add_argument('--band', type=float, default=BAND,
                   help=f'Contrasts inside +/-band are UNDETERMINED. Default {BAND}, twice the measured cross-split sd.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
