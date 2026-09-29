'''WHO gains from degree augmentation? Test whether degree's advantage is concentrated on LOW-DEGREE nodes.'''
# paper_log 29-32 asked what is special about degree as a SIGNAL - are same-degree nodes alike (no), are they far apart
# (no) - and closed both. This asks a different question, and the first one about OUTCOMES rather than about the graph:
# on the graphs where degree wins, WHICH nodes actually gain? The hypothesis comes from the GNN degree-bias literature
# (GraphPatcher; SAug's hub/tail imbalance): low-degree nodes have thin neighbourhoods, message passing serves them
# badly, and augmentation may win precisely by giving them context. If that is why degree wins, its advantage should
# grow as node degree falls - and it should do so on the degree-winning graphs and NOT on the others.
# THE CONFOUND this design exists to defeat: a role graph adds K=10 edges per node, which is an enormous relative change
# for a degree-2 node and nothing for a degree-500 node. A tail gradient will therefore appear for EVERY signal. Finding
# one for degree alone proves nothing, so the test is a DIFFERENCE - degree against its best rival, and the degree
# winners against the other augmenting graphs.
# TERMS, fixed BEFORE running (user, 2026-09-07):
#   unit      every test pair, positive and negative, keyed by min(deg(u), deg(v)) on the TRAIN graph - the weakest
#             endpoint. Nothing is binned for the test and no node is dropped, so the tail cannot be selected away.
#   advantage per positive edge, its exact AUC contribution (the share of negatives it outranks) under degree minus the
#             same under the best rival signal, PAIRED on the same split and seed. Degree vs original is context, not the test.
#   statistic Spearman rho(advantage, min endpoint degree). Scale-free, so it compares across graphs whose advantage
#             magnitudes differ tenfold; the OLS slope on log degree is reported beside it for interpretation.
#             NEGATIVE = degree's advantage grows as degree falls = the hypothesis.
#   VERDICT   all three required: (a) negative on the 9 degree-winning graphs, (b) NOT negative on the other augmenting
#             graphs - a difference-in-differences, (c) the sign holds on at least 7 of the 9 individually, because
#             6 of 9 happens a quarter of the time by chance and "one dataset carried it" is how the DSC near-miss happened.
# Trains nothing: it re-scores the embeddings already on disk, exactly as tie_break.py does.

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from experiments.degree_consistency import VERDICTS
from experiments.run_core import split, tag
from experiments.strategy_select import SIGNAL

GAIN_CSV = cfg.RESULTS_DIR / "degree_tail_gain.csv"
TEST_CSV = cfg.RESULTS_DIR / "degree_tail_gain_test.csv"

SEEDS = list(range(42, 52))       # every seed that may exist; missing embeddings are skipped, never trained
ENCODER = "graphsage_edge"
DECILES = 10                      # for the printed table ONLY - the test statistic is continuous
MIN_WINNER_CONSISTENCY = 7        # of the 9 degree winners (term c)


def pairs(path):
    '''Node-id pairs from a split file.'''
    return np.array([[int(a), int(b)] for a, b in (l.split() for l in open(path) if l.strip())])


# Each positive edge's exact contribution to AUC: the share of negatives it outranks. Mean over positives = the AUC.
def contribution(pos_score, neg_score):
    '''Per-positive AUC contribution, ties counted as half, via one sort of the negatives.'''
    s = np.sort(neg_score)
    lo = np.searchsorted(s, pos_score, side="left")
    hi = np.searchsorted(s, pos_score, side="right")
    return (lo + hi) / (2.0 * len(s))


# One (dataset, seed): per-positive advantage of degree over its rival and over original, keyed by weakest endpoint degree.
def advantage(ds, seed, k, pick):
    '''(min endpoint degree, degree-vs-rival advantage, degree-vs-original advantage) for every test positive.'''
    embs = {r: cfg.NB3_DIR / "link_prediction" / ds / f"k{k}" / v / f"{tag(ENCODER, 'all')}_s{seed}.emb"
            for r, v in pick.items()}
    if not all(e.exists() for e in embs.values()):
        return None                                                # this seed was never trained for one of the variants
    d = split(ds, seed)                                            # rebuilds a deleted split dir; the build is deterministic
    G = graph_io.load_graph(d / "train.edgelist")
    deg = dict(G.degree)
    P, N = pairs(d / "test_pos.txt"), pairs(d / "test_neg.txt")
    out = {}
    for role, emb in embs.items():
        kv = KeyedVectors.load_word2vec_format(str(emb))
        M = np.array([kv[str(n)] if str(n) in kv else np.zeros(kv.vector_size) for n in G.nodes])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        i = {n: j for j, n in enumerate(G.nodes)}
        cos = lambda X: (M[[i[a] for a in X[:, 0]]] * M[[i[b] for b in X[:, 1]]]).sum(1)
        out[role] = contribution(cos(P), cos(N))
        del kv, M
    lo = np.array([min(deg.get(a, 0), deg.get(b, 0)) for a, b in P], dtype=float)
    return lo, out["degree"] - out["rival"], out["degree"] - out["original"]


# A signal's representative is its best variant by MEAN auc, chosen once per graph - never per degree bin, which would
# let the comparison pick whichever rival happens to be weakest among the tail edges.
def representatives(ds, board):
    '''{"degree": variant, "rival": variant, "original": "original"} for one dataset.'''
    b = board[(board["dataset"] == ds) & (board["encoder"] == ENCODER) & (board["top_K_neighbors"] == 10)
              & (board["task"] == "link prediction (AUC)")]
    m = b.assign(signal=b["graph_variant"].map(SIGNAL)).groupby(["signal", "graph_variant"])["mean"].mean().reset_index()
    best = {s: g.loc[g["mean"].idxmax()] for s, g in m.groupby("signal")}
    if "degree" not in best:
        return None
    rival = max((s for s in ("psi", "centrality") if s in best), key=lambda s: best[s]["mean"], default=None)
    return None if rival is None else {"degree": best["degree"]["graph_variant"],
                                       "rival": best[rival]["graph_variant"], "original": "original"}


def measure(ds, k, board):
    '''One row: the tail gradient for one dataset, averaged over every seed whose embeddings are on disk.'''
    pick = representatives(ds, board)
    if pick is None:
        return None
    rows = []
    for seed in SEEDS:
        got = advantage(ds, seed, k, pick)
        if got is None:
            continue
        lo, adv_r, adv_o = got
        ok = lo > 0
        rows.append({"rho_rival": spearmanr(lo[ok], adv_r[ok]).statistic,
                     "rho_original": spearmanr(lo[ok], adv_o[ok]).statistic,
                     "slope_rival": float(np.polyfit(np.log(lo[ok]), adv_r[ok], 1)[0]),
                     "mean_adv": float(adv_r[ok].mean()),
                     # readability only: mean advantage in the lowest and highest decile of weakest-endpoint degree
                     "adv_low": float(adv_r[ok][lo[ok] <= np.quantile(lo[ok], 1 / DECILES)].mean()),
                     "adv_high": float(adv_r[ok][lo[ok] >= np.quantile(lo[ok], 1 - 1 / DECILES)].mean())})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    return {"dataset": ds, "n_seeds": len(r), "degree_variant": pick["degree"], "rival_variant": pick["rival"],
            **{c: round(float(r[c].mean()), 6) for c in r.columns},
            "rho_rival_sem": round(float(r["rho_rival"].std(ddof=1) / np.sqrt(len(r))), 6) if len(r) > 1 else float("nan")}


# Reads what is already on disk and measures only what is missing (same contract as degree_consistency.py).
def tail_gain(datasets, k=10):
    '''The tail gradient per dataset, cached in results/degree_tail_gain.csv.'''
    have = pd.read_csv(GAIN_CSV) if GAIN_CSV.exists() else pd.DataFrame(columns=["dataset"])
    miss = [d for d in datasets if d not in set(have.get("dataset", []))]
    if miss:
        board, rows = pd.read_csv(cfg.SCOREBOARD_CSV), []
        for d in miss:
            r = measure(d, k, board)
            if r is None:
                print(f"skipped {d}: no scoreboard row or no embeddings on disk", flush=True)
                continue
            rows.append(r)
            print(f"measured {d}: rho(advantage, min degree) {r['rho_rival']:+.4f}  "
                  f"low decile {r['adv_low']:+.4f} vs high decile {r['adv_high']:+.4f}  ({r['n_seeds']} seeds)", flush=True)
        if rows:
            have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
            have.to_csv(GAIN_CSV, index=False)
    return have[have["dataset"].isin(datasets)].sort_values("dataset").reset_index(drop=True)


def cells(datasets, k=10):
    '''Augmenting graphs only, split into the degree winners and the rest.'''
    v = pd.read_csv(VERDICTS).query("band == 'paired'")[["dataset", "winner_signals", "beats_original"]]
    d = v[v["dataset"].isin(datasets) & v["beats_original"]].merge(tail_gain(datasets, k), on="dataset")
    d["degree_sole"] = d["winner_signals"] == "degree"
    return d.sort_values(["degree_sole", "dataset"], ascending=[False, True]).reset_index(drop=True)


# The pre-declared verdict. A permutation over the winner/non-winner labels prices the difference-in-differences, so the
# group gap is read against what an arbitrary split of the same 39 numbers produces.
def verdict(d, draws=20000, seed=42):
    '''The three declared criteria, each scored, plus the permutation price on the group difference.'''
    w = d[d["degree_sole"]]["rho_rival"].to_numpy()
    o = d[~d["degree_sole"]]["rho_rival"].to_numpy()
    gap = float(w.mean() - o.mean())
    x, n = np.r_[w, o], len(w)
    rng = np.random.default_rng(seed)
    p = float(np.mean([rng.permutation(x)[:n].mean() - rng.permutation(x)[n:].mean() <= gap for _ in range(draws)]))
    consistent = int((w < 0).sum())
    return pd.DataFrame([{
        "n_winners": len(w), "n_others": len(o),
        "mean_rho_winners": round(float(w.mean()), 6), "mean_rho_others": round(float(o.mean()), 6),
        "group_gap": round(gap, 6), "p_permutation": round(p, 5),
        "a_winners_negative": bool(w.mean() < 0),
        "b_others_not_negative": bool(o.mean() >= 0),
        "c_consistent": f"{consistent}/{len(w)}",
        "c_met": bool(consistent >= MIN_WINNER_CONSISTENCY),
        "VERIFIED": bool(w.mean() < 0 and o.mean() >= 0 and consistent >= MIN_WINNER_CONSISTENCY),
    }])


def report(d, t):
    '''Print the per-graph gradient, then the three declared criteria.'''
    print("\nWHO GAINS FROM DEGREE AUGMENTATION - rho(per-edge advantage over the best rival, weakest endpoint degree)")
    print("negative rho = degree's advantage grows as node degree falls = the hypothesis\n")
    print(d[["dataset", "winner_signals", "degree_sole", "n_seeds", "rho_rival", "rho_rival_sem", "slope_rival",
             "mean_adv", "adv_low", "adv_high", "rho_original"]].to_string(index=False))
    if t.empty:
        return
    r = t.iloc[0]
    print(f"\nDECLARED CRITERIA")
    print(f"  (a) degree winners have a NEGATIVE mean rho .......... {r['mean_rho_winners']:+.4f}  -> {r['a_winners_negative']}")
    print(f"  (b) other augmenting graphs do NOT ................... {r['mean_rho_others']:+.4f}  -> {r['b_others_not_negative']}")
    print(f"  (c) sign holds on >= {MIN_WINNER_CONSISTENCY} of the {r['n_winners']} winners individually ... {r['c_consistent']}  -> {r['c_met']}")
    print(f"  difference-in-differences {r['group_gap']:+.4f}, permutation P = {r['p_permutation']}")
    print(f"\nVERIFIED: {'YES - proceed to the tail-imbalance characteristic' if r['VERIFIED'] else 'NO - measured negative, no characteristic is built'}")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d = cells(args.datasets, args.k)
    t = verdict(d) if d["degree_sole"].any() and (~d["degree_sole"]).any() else pd.DataFrame()
    if not t.empty:
        t.to_csv(TEST_CSV, index=False)
    report(d, t)
    print(f"\n{len(d):3d} augmenting graphs  ->  results/{GAIN_CSV.name}, results/{TEST_CSV.name}")


# Defines command-line options (mirrors degree_dispersion.py / tie_break.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Test whether degree augmentation's advantage is concentrated on low-degree nodes.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to measure. Default: the degree corpus.')
    p.add_argument('--k', type=int, default=10, help='Top-K the virtual graphs were built with. Default 10 (locked).')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
