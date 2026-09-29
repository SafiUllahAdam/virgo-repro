'''Module 3, step 2 - SCORE. After the pipeline has trained on the held-out datasets, join the predictions saved BEFORE
training to the keep/augment verdict the training actually produced, and report pred-vs-actual.

Never fits: reads the frozen scoreboard through gaps(frozen_inputs(...)) and never calls threshold() or candidate_rules().
Both frozen rules are link-prediction rules, so the pre-registered verdict is scored against the LP cell.'''

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import frozen_rules as fr
from virgo import graph_io
from virgo.virtual_graph import VirtualGraph
from experiments.characterize import frozen_inputs, gaps, rho
from experiments import predict_module3 as pm

SCORED_CSV = cfg.RESULTS_DIR / "module3_scored.csv"
EVIDENCE_CSV = cfg.RESULTS_DIR / "module3_evidence.csv"
DISCOVERY_CSV = cfg.RESULTS_DIR / "candidate_rules.csv"     # Module-2 evidence, READ-ONLY here
CHARACTERIZATION_CSV = cfg.RESULTS_DIR / "dataset_characterization.csv"     # Module-2 properties, READ-ONLY here
CORR_CSV = cfg.RESULTS_DIR / "module3_property_correlation.csv"
AGREE_CSV = cfg.RESULTS_DIR / "module3_decision_agreement.csv"


def actual_lp_verdicts(datasets):
    '''The keep/augment verdict the trained pipeline produced for each held-out LINK-PREDICTION cell (from the frozen scoreboard).'''
    inp = frozen_inputs(datasets)
    lp = gaps(inp)
    lp = lp[lp["task_family"] == "link prediction"].copy()
    # The two seed spreads behind the verdict, plus the pooled 3-seed noise it was thresholded on (compare(): hypot/sqrt2).
    std = inp.set_index(["dataset", "task", "graph_variant"])["std"]
    lp["original_std"] = [std.get((d, t, "original"), float("nan")) for d, t in zip(lp["dataset"], lp["task"])]
    lp["best_augmented_std"] = [std.get((d, t, v), float("nan")) for d, t, v in zip(lp["dataset"], lp["task"], lp["best_variant"])]
    lp["noise"] = (np.hypot(lp["original_std"], lp["best_augmented_std"]) / np.sqrt(2)).round(4)
    return lp[["dataset", "metric", "original", "original_std", "best_augmented", "best_augmented_std", "best_variant",
               "gap_abs", "noise", "gap_rel", "gap_sigma", "verdict", "usable"]]


def nc_verdicts(datasets):
    '''Held-out NODE-CLASSIFICATION cells. No frozen rule covers them - this is the boundary check ("never augment"), not a scored rule.'''
    g = gaps(frozen_inputs(datasets))
    nc = g[g["task_family"] == "node classification"]
    return nc[["dataset", "metric", "original", "best_augmented", "best_variant", "gap_sigma", "verdict", "usable"]]


def score(datasets=None):
    '''Join the pre-training predictions to the actual held-out LP verdicts; returns the pred-vs-actual table. Never fits.'''
    assert pm.PRED_CSV.exists(), f"{pm.PRED_CSV} missing - run experiments/predict_module3.py BEFORE training first."
    pred = pd.read_csv(pm.PRED_CSV)
    dsets = datasets or pred["dataset"].tolist()
    act = actual_lp_verdicts(dsets).rename(columns={"verdict": "actual_verdict"})
    m = pred.merge(act, on="dataset", how="left")
    # A cell only scores a rule when the experiment actually DECIDED: the verdict must be augment or keep original AND the
    # metric must resolve the variants. Module 2 fitted on exactly those cells (ties dropped, unusable dropped), so counting a
    # tie as a miss would invent an error the data cannot support - a tie means the 3-seed noise covers the gap, not that the rule was wrong.
    decided = m["actual_verdict"].isin(["augment", "keep original"]) & m["usable"].fillna(False)
    # Score EACH rule on its own, never the combined "rules disagree" verdict: a disagreement is not dropped, it just shows
    # which rule was right. Four states, kept apart: "pending" = not trained yet; "no decision" = tie or unusable cell;
    # None = the rule did not fire (property missing, e.g. homophily on an unlabelled graph); True/False = actually scored.
    for r in fr.FROZEN_RULES:
        m[f"{r.name}_correct"] = ["pending" if not isinstance(av, str) else "no decision" if not ok
                                  else (bool(p == av) if p in ("augment", "keep original") else None)
                                  for p, av, ok in zip(m[f"{r.name}_pred"], m["actual_verdict"], decided)]
    m["actual_verdict"] = m["actual_verdict"].fillna("pending")     # untrained held-out datasets read "pending", not NaN
    cols = (["dataset", "homophily_adjusted", "largest_component_frac", "rule1_pred", "rule1_correct",
             "rule2_pred", "rule2_correct", "predicted_verdict", "actual_verdict", "original", "best_augmented", "best_variant"])
    return m[[c for c in cols if c in m.columns]]


def evidence(datasets=None):
    '''One wide row per held-out dataset: frozen property + prediction beside what training measured. Reporting only, never fits.'''
    m = score(datasets)
    a = actual_lp_verdicts(datasets or m["dataset"].tolist())
    e = m.merge(a.drop(columns=["original", "best_augmented", "best_variant"]), on="dataset", how="left")
    return e[["dataset", "homophily_adjusted", "largest_component_frac", "rule1_pred", "rule2_pred",
              "original", "original_std", "best_augmented", "best_augmented_std", "best_variant",
              "gap_abs", "noise", "gap_sigma", "actual_verdict", "rule1_correct", "rule2_correct"]]


def rule_properties(datasets=None):
    '''The two frozen rule properties for every dataset, discovery panel + held-out, from the frozen CSVs. Never fits.'''
    props = ["homophily_adjusted", "largest_component_frac"]
    disc = pd.read_csv(CHARACTERIZATION_CSV)
    disc = disc[disc["dataset"].isin(fr.DISCOVERY_PANEL)]  # the CSV regains dropped datasets if characterize refits with defaults
    held = pd.read_csv(pm.PRED_CSV)
    both = pd.concat([disc.assign(panel="discovery"), held.assign(panel="held-out")], ignore_index=True)
    if datasets:
        both = both[both["dataset"].isin(datasets)]
    return both[["dataset", "panel"] + props]


def property_correlation(datasets=None):
    '''Redundancy DIAGNOSTIC: 2x2 Spearman matrix of the two frozen rule properties across ALL datasets (discovery panel +
    held-out - review request 2026-08-08), plus each property's variation. Never fits.'''
    both = rule_properties(datasets)
    props = ["homophily_adjusted", "largest_component_frac"]
    n, r, p = rho(both[props[0]], both[props[1]])          # rho() floors p at 2/n!, so small-n ranks are never overstated
    mat = pd.DataFrame([[1.0, r], [r, 1.0]], index=props, columns=props)
    mat.index.name = "property"
    # A correlation alone cannot show rule 2's real weakness - the ceiling pile-up - so the variation counts ride along.
    mat["n_values"] = [int(both[c].notna().sum()) for c in props]      # ogbl_ddi is unlabelled -> no homophily value
    mat["distinct_values"] = [int(both[c].nunique()) for c in props]
    mat["n_at_1.0"] = [int((both[c] == 1.0).sum()) for c in props]
    mat["n_datasets"], mat["spearman_p"] = n, p            # pair-level (non-missing pairs), repeated on both rows for one flat table
    return mat


def decision_agreement(datasets=None):
    '''Redundancy DIAGNOSTIC: R1 x R2 decisions for EVERY dataset, discovery panel + held-out (review request
    2026-08-08), and in a disagreement which rule the LP verdict backed. Never fits - calls come from the frozen points;
    discovery rows are IN-SAMPLE (the rules were fitted there), so their correct-columns describe fit, not validation.'''
    m = rule_properties(datasets)
    for r in fr.FROZEN_RULES:
        m[f"{r.name}_pred"] = [fr.predict_one(r, v) for v in m[r.predictor]]
    pred = pd.read_csv(pm.PRED_CSV).set_index("dataset")   # integrity check, not a fit: a recomputed held-out call must
    for ds, p1, p2 in zip(m["dataset"], m["rule1_pred"], m["rule2_pred"]):     # equal the write-once pre-registration
        if ds in pred.index:
            assert (p1, p2) == (pred.loc[ds, "rule1_pred"], pred.loc[ds, "rule2_pred"]), f"{ds}: call drifted from the frozen prediction"
    act = actual_lp_verdicts(m["dataset"].tolist()).rename(columns={"verdict": "actual_verdict"})
    m = m.merge(act[["dataset", "actual_verdict", "usable"]], on="dataset", how="left")
    m["actual_verdict"] = m["actual_verdict"].fillna("no LP cell")             # ogbn_arxiv is node-classification only
    decided = m["actual_verdict"].isin(["augment", "keep original"]) & m["usable"].fillna(False)
    for r in fr.FROZEN_RULES:
        m[f"{r.name}_correct"] = ["n/a" if p == "n/a" else "no decision" if not ok else bool(p == av)
                                  for p, av, ok in zip(m[f"{r.name}_pred"], m["actual_verdict"], decided)]
    # None (not False) when R1 cannot fire: an unlabelled graph is a coverage hole, not a disagreement.
    m["rules_agree"] = [None if "n/a" in (p1, p2) else bool(p1 == p2) for p1, p2 in zip(m["rule1_pred"], m["rule2_pred"])]
    # `is True` because the correct-columns mix bool with "n/a"/"no decision" strings (object dtype, == is unreliable).
    m["disagreement_result"] = ["R1 n/a (unlabelled)" if a is None
                                else "rules agree" if a
                                else "R1 correct, R2 wrong" if c1 is True
                                else "R2 correct, R1 wrong" if c2 is True
                                else f"disagree, undecided ({av})"
                                for a, c1, c2, av in zip(m["rules_agree"], m["rule1_correct"], m["rule2_correct"], m["actual_verdict"])]
    return m[["dataset", "panel", "homophily_adjusted", "largest_component_frac", "rule1_pred", "rule2_pred",
              "actual_verdict", "rule1_correct", "rule2_correct", "rules_agree", "disagreement_result"]]


def uniformity(datasets, k=10, sims=("psi", "degree", "centrality")):
    '''Post-hoc DIAGNOSTIC: how many distinct structural roles a graph has, measured on the tie classes the builder samples inside.

    Reproduces VirtualGraph.build()'s own key math, so this is what the virtual graph actually sees, not a proxy. sampled_frac =
    the share of nodes whose tie class alone can fill K - for them "top-K nearest" is undefined and the K role edges are drawn at
    random inside the class. High sampled_frac = the role signal is arbitrary, so rewiring can only remove real edges.'''
    rows = []
    for ds in datasets:
        G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
        deg = np.array([d for _, d in G.degree()], dtype=float)
        row = {"dataset": ds, "nodes": G.number_of_nodes(), "distinct_degrees": int(len(np.unique(deg))),
               "degree_cv": round(float(deg.std() / deg.mean()), 4) if deg.mean() else float("nan")}
        for sim in sims:
            vals = VirtualGraph(G, seed=cfg.REPRO["seed"]).signatures(sim)[1][:, 0]
            span = float(vals.max() - vals.min())
            keys = np.round(vals / (span * cfg.GRAPH_POLICY["sig_tol"])) if span > 0 else np.zeros(len(vals))
            size = pd.Series(keys).map(pd.Series(keys).value_counts())          # each node -> the size of its tie class
            row[f"distinct_{sim}"] = round(float(len(set(keys)) / len(keys)), 4)
            row[f"sampled_frac_{sim}"] = round(float((size - 1 >= min(k, len(vals) - 1)).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def discovery_evidence():
    '''The frozen Module-2 numbers for the two rules, read straight off results/candidate_rules.csv. Nothing is re-fitted here.'''
    c = pd.read_csv(DISCOVERY_CSV)
    c = c[c["task_family"] == "link prediction"]
    main_t, fixed_t = c[c["target"] == "gap_rel"].set_index("predictor"), c[c["target"] == "gap_fixed_rel"].set_index("predictor")
    rows = []
    for r in fr.FROZEN_RULES:
        m = main_t.loc[r.predictor]
        # Integrity check, not a fit: if the discovery file was ever re-fitted the point would drift away from the frozen rule.
        assert abs(float(m["threshold"]) - r.point) < 1e-9, \
            f"{r.name}: candidate_rules.csv threshold {m['threshold']} != frozen point {r.point} - the discovery evidence has been re-fitted"
        rows.append({"rule": r.name, "predictor": r.predictor, "threshold": r.point,
                     "interval": f"({r.interval[0]}, {r.interval[1]})",
                     "spearman_rho": float(m["spearman_rho"]),                              # target = best-variant gap
                     "fixed_variant_rho": float(fixed_t.loc[r.predictor, "spearman_rho"]),  # selection-bias-free control
                     "lodo_min_abs_rho": float(m["lodo_min_abs_rho"]),
                     "n_datasets": int(m["n_decided"]), "n_exceptions": int(m["n_exceptions"])})
    return pd.DataFrame(rows)


def main(args):
    m = score(args.datasets)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(SCORED_CSV, index=False)
    print(f"{len(m):3d} rows -> results/module3_scored.csv\n")
    print(m.to_string(index=False))
    print()
    for r in fr.FROZEN_RULES:                              # each rule reported separately, so a disagreement is never lost
        col = list(m[f"{r.name}_correct"])
        c = [v for v in col if isinstance(v, bool)]        # bool only: skip pending, no decision (tie/unusable) and did-not-fire
        skipped = [v for v in col if not isinstance(v, bool)]
        print(f"{r.name} ({r.predictor} {r.op} {r.point}): " + (f"{sum(c)}/{len(c)} correct" if c else "nothing scored yet")
              + (f"  [{len(skipped)} not scored: " + ", ".join(sorted({str(v) for v in skipped})) + "]" if skipped else ""))

    # Redundancy diagnostics (review request 2026-08-07): descriptive only, justify keeping R1 and dropping R2 - no refit.
    corr, agree = property_correlation(args.datasets), decision_agreement(args.datasets)
    corr.to_csv(CORR_CSV)
    agree.to_csv(AGREE_CSV, index=False)
    print(f"\nproperty correlation matrix -> results/module3_property_correlation.csv\n{corr.to_string()}")
    dis = agree[agree["rules_agree"] == False]             # == because the column holds True/False/None (R1 n/a)
    r1w, r2w = int((dis["disagreement_result"] == "R1 correct, R2 wrong").sum()), int((dis["disagreement_result"] == "R2 correct, R1 wrong").sum())
    print(f"\ndecision agreement -> results/module3_decision_agreement.csv\n{agree.to_string(index=False)}")
    print(f"\nrules agree on {int((agree['rules_agree'] == True).sum())}/{len(agree)} datasets "
          f"({int(agree['rules_agree'].isna().sum())} R1-n/a); of {len(dis)} disagreements: "
          f"R1 correct {r1w}, R2 correct {r2w}, undecided {len(dis) - r1w - r2w}")


def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Module 3 SCORE: compare the pre-training predictions to what training actually produced.")
    p.add_argument('--datasets', nargs='+', default=None, help='Held-out datasets to score. Default: whatever predict_module3 saved.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
