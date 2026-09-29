'''When does Psi add nothing beyond DEGREE? If degree already explains Psi, the simpler signal may be sufficient.'''
# The last selector direction left after paper_log 29-33 closed three mechanisms. It inverts the question: instead of
# asking what is special about degree, ask when its RIVAL is redundant. Psi is a Poisson/KL score built on degree-based
# signatures (section 19 measured rho ~ -0.80 between them on 33 of 34 graphs), so on some graphs Psi is nearly a
# relabelling of degree and on others it genuinely refines it. Where it does not refine, the simpler signal may win.
#   psi_explained_by_degree = the correlation ratio eta^2 of RANK-TRANSFORMED Psi across exact-degree classes:
#   the share of Psi's variation that degree already accounts for. HIGH = Psi adds little beyond degree.
# Ranks, not raw Psi, because Psi has extreme tails (its log(p/Omega) term reaches ~52 on some graphs) and a
# variance ratio on raw values would be an outlier statistic.
# ADJUSTED against its own chance floor, exactly as homophily_adjusted is: eta^2 rises mechanically with the number of
# groups, and (c-1)/(n-1) is 0.26 on a 131-node graph with 35 degree classes but 0.006 on a 20k-node graph. Comparing
# raw eta^2 across the panel would mostly compare class counts.
# TWO RISKS, recorded BEFORE the run:
#   1  a NEAR-NEIGHBOUR already failed. Section 24/27 screened deg_psi_sig_spearman - the GLOBAL rank correlation of the
#      two signatures - at 8 exceptions and P(shuffle) 0.960. This is a different quantity: a global correlation of -0.9
#      is perfectly compatible with Psi varying wildly INSIDE a degree class, which is what eta^2 measures.
#   2  it may predict TIES, not degree wins. Section 19 found degree and Psi tie exactly where their role graphs are most
#      alike. "Psi adds nothing" is plausibly the tie regime, and the target here is degree winning ALONE. That is the
#      kill test below, declared in advance rather than discovered afterwards.
# TERMS: one predictor, one form, direction pre-declared HIGH => degree, bar P(shuffle) < 0.05, plus the two kill tests.
# Read off the ORIGINAL graph with no labels and no role graph, so a call exists before any encoder runs.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo.virtual_graph import VirtualGraph
from experiments.characterize import loo_threshold, rho, threshold
from experiments.degree_consistency import VERDICTS, shuffle_price
from experiments.strategy_select import loo_majority, properties

REFINEMENT_CSV = cfg.RESULTS_DIR / "psi_refinement.csv"
RULES_CSV = cfg.RESULTS_DIR / "psi_refinement_rules.csv"

PREDICTOR = "psi_explained_by_degree"
DECLARED_SIDE = "high"                                     # high = degree already explains Psi => degree wins
CONFOUNDS = ["density", "nodes", "avg_degree"]             # kill test (i): a proxy for any of these at |rho| >= 0.7 is dead
MAX_CONFOUND_RHO = 0.7


# Correlation ratio: the share of a variable's total variation that a grouping already accounts for.
def eta_squared(value, group):
    '''(raw eta^2, chance floor, adjusted eta^2) of `value` across `group`; the floor is the null E[eta^2] = (c-1)/(n-1).'''
    n, order = len(value), np.argsort(group, kind="stable")
    v, g = value[order], group[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    sizes = np.diff(np.r_[starts, n]).astype(float)
    means = np.add.reduceat(v, starts) / sizes
    total = float(((v - v.mean()) ** 2).sum())
    between = float((sizes * (means - v.mean()) ** 2).sum())
    c = len(starts)
    raw = between / total if total > 0 else float("nan")
    floor = (c - 1) / (n - 1) if n > 1 else float("nan")
    adj = (raw - floor) / (1 - floor) if floor == floor and floor < 1 else float("nan")
    return round(raw, 6), round(floor, 6), round(adj, 6)


def measure(ds):
    '''One row: how much of Psi's variation the degree partition already explains, on the ORIGINAL graph.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    nodes, X = VirtualGraph(G, seed=cfg.REPRO["seed"]).signatures("psi")
    deg = np.array([d for _, d in G.degree(nodes)], dtype=np.int64)
    psi = rankdata(X[:, 0])                                # ranks: Psi's raw tail reaches ~52 and would dominate a variance ratio
    raw, floor, adj = eta_squared(psi, deg)
    return {"dataset": ds, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "degree_classes": int(len(np.unique(deg))), "psi_eta2_raw": raw, "psi_eta2_floor": floor,
            PREDICTOR: adj}


# Reads what is already on disk and measures only what is missing (same contract as degree_consistency.py).
def refinement(datasets):
    '''Psi-refinement per dataset, cached in results/psi_refinement.csv.'''
    have = pd.read_csv(REFINEMENT_CSV) if REFINEMENT_CSV.exists() else pd.DataFrame(columns=["dataset"])
    miss = [d for d in datasets if d not in set(have.get("dataset", []))]
    if miss:
        rows = []
        for d in miss:
            rows.append(measure(d))
            print(f"measured {d}: degree explains {100 * rows[-1]['psi_eta2_raw']:.1f}% of Psi's rank variation "
                  f"(chance floor {100 * rows[-1]['psi_eta2_floor']:.1f}%, adjusted {rows[-1][PREDICTOR]:+.4f}, "
                  f"{rows[-1]['degree_classes']} classes)", flush=True)
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(REFINEMENT_CSV, index=False)
    return have[have["dataset"].isin(datasets)].sort_values("dataset").reset_index(drop=True)


def cells(datasets):
    '''Augmenting graphs only, with the degree-sole label and the tie label the second kill test needs.'''
    v = pd.read_csv(VERDICTS).query("band == 'paired'")[["dataset", "winner_signals", "beats_original"]]
    d = v[v["dataset"].isin(datasets) & v["beats_original"]].merge(refinement(datasets), on="dataset")
    d = d.merge(properties(list(d["dataset"])), on="dataset", suffixes=("", "_p"))
    d["degree_sole"] = d["winner_signals"] == "degree"
    d["is_tie"] = [len(s.split("|")) > 1 for s in d["winner_signals"]]     # kill test (ii): does it really predict TIES?
    return d.sort_values("dataset").reset_index(drop=True)


def one_screen(x, y, label):
    '''Threshold, exceptions, LOO and the shuffle price for one binary target.'''
    t, side, err, acc, lo, hi = threshold(x, y)
    l_acc, l_ok, l_n, _, _ = loo_threshold(x, y)
    l_maj, _ = loo_majority(x, y)
    p_shuf, _ = shuffle_price(x, y)
    return {"target": label, "n": int(len(x)), "n_positive": int(y.sum()),
            "spearman_rho": rho(pd.Series(x), pd.Series(y.astype(float)))[1],
            "threshold": t, "positive_side": side, "n_exceptions": err, "accuracy": acc,
            "interval_lo": lo, "interval_hi": hi, "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
            "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 4), "loo_majority": l_maj,
            "p_shuffle_as_good": p_shuf}


def screen(d):
    '''The declared test on degree-vs-rest, plus the tie target the second kill test compares it against.'''
    x = d[PREDICTOR].to_numpy(dtype=float)
    rows = [one_screen(x, d["degree_sole"].to_numpy(), "degree wins alone"),
            one_screen(x, d["is_tie"].to_numpy(), "the winner is a TIE (kill test ii)")]
    out = pd.DataFrame(rows)
    for c in CONFOUNDS:
        out[f"rho_vs_{c}"] = rho(pd.Series(x), d[c].reset_index(drop=True))[1]
    out["rho_vs_degree_classes"] = rho(pd.Series(x), d["degree_classes"].reset_index(drop=True))[1]
    main = out.iloc[0]
    confounded = max(abs(out.loc[0, f"rho_vs_{c}"]) for c in CONFOUNDS + ["degree_classes"][:0]) if CONFOUNDS else 0.0
    confounded = max(confounded, abs(out.loc[0, "rho_vs_degree_classes"]))
    out["kill_i_confounded"] = [bool(confounded >= MAX_CONFOUND_RHO)] * len(out)
    out["kill_ii_predicts_ties_better"] = [bool(rows[1]["p_shuffle_as_good"] < rows[0]["p_shuffle_as_good"])] * len(out)
    out["survives"] = [bool(main["p_shuffle_as_good"] < 0.05 and main["positive_side"] == DECLARED_SIDE
                            and main["loo_accuracy"] == main["loo_accuracy"]
                            and main["loo_accuracy"] > main["majority_baseline"]
                            and main["loo_accuracy"] > main["loo_majority"]
                            and confounded < MAX_CONFOUND_RHO
                            and rows[1]["p_shuffle_as_good"] >= rows[0]["p_shuffle_as_good"]), False]
    return out


def report(d, s):
    '''Print the measured refinement, then the declared test and both kill tests.'''
    print("\nHOW MUCH OF PSI DOES DEGREE ALREADY EXPLAIN - correlation ratio on ranks, ORIGINAL graph, no labels")
    print(d[["dataset", "winner_signals", "degree_sole", "is_tie", "psi_eta2_raw", "psi_eta2_floor", PREDICTOR,
             "degree_classes"]].to_string(index=False))
    print(f"\n{len(d)} augmenting graphs  |  {int(d['degree_sole'].sum())} name DEGREE alone  |  "
          f"{int(d['is_tie'].sum())} are ties")
    if s.empty:
        return
    print("\nPRE-DECLARED TEST: high => degree.  Bar: P(shuffle) < 0.05, declared side, LOO above baseline, both kill tests passed.")
    print(s[["target", "n", "n_positive", "spearman_rho", "threshold", "positive_side", "n_exceptions",
             "loo_accuracy", "majority_baseline", "loo_majority", "p_shuffle_as_good"]].to_string(index=False))
    r = s.iloc[0]
    print(f"\n  confounds: " + "  ".join(f"{c} {s.loc[0, f'rho_vs_{c}']:+.3f}" for c in CONFOUNDS)
          + f"  degree_classes {s.loc[0, 'rho_vs_degree_classes']:+.3f}")
    print(f"  kill test (i)  proxy for an already-screened property .... {'FAILED' if r['kill_i_confounded'] else 'passed'}")
    print(f"  kill test (ii) predicts TIES better than degree wins ..... "
          f"{'FAILED' if r['kill_ii_predicts_ties_better'] else 'passed'}")
    print(f"\nSURVIVES: {'YES - ' + str(r['threshold']) + ' is a candidate cut' if r['survives'] else 'no'}")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d = cells(args.datasets)
    s = screen(d) if d["degree_sole"].any() and not d["degree_sole"].all() else pd.DataFrame()
    if not s.empty:
        s.to_csv(RULES_CSV, index=False)
    report(d, s)
    print(f"\n{len(d):3d} augmenting graphs  ->  results/{REFINEMENT_CSV.name}, results/{RULES_CSV.name}")


# Defines command-line options (mirrors degree_consistency.py / degree_dispersion.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure how much of Psi degree already explains, and screen it as a stage-2 degree rule.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to measure. Default: the degree corpus.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
