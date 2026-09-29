'''Stage 1 - PREDICT and SCORE the combined stage-1 call on a genuinely unseen set (Module 5 and after).

Same contract as Module 3, one level up: the pre-registered verdict is written BEFORE any encoder runs, and this script
never fits. It imports virgo.frozen_rules and never calls threshold(), candidate_rules() or gate_rules.screen(), so an
unseen dataset can only be SCORED against the locked rules, never move them.

Since 2026-09-01 the second stage-1 condition is the CLUSTERING EXCEPTION (avg_clustering), measured on the ORIGINAL
graph, so nothing has to be built before predicting. The superseded retention gate is still measured and recorded when
the role graph happens to exist, for continuity with the published Module-4/5 rows - it no longer decides anything.'''

import argparse
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo import frozen_rules as fr
from experiments.gate_rules import CANONICAL_SIM, vg_measure, vg_path
from experiments.predict_module3 import rule_properties
from experiments.score_module3 import actual_lp_verdicts

PRED_CSV = cfg.RESULTS_DIR / "module5_predictions.csv"
SCORED_CSV = cfg.RESULTS_DIR / "module5_scored.csv"

# What gate_properties() can produce. Guards against a future rule needing a property this measurer does not compute.
PRODUCES = {"homophily_adjusted", "largest_component_frac", "avg_clustering", "original_retention"}
assert {r.predictor for r in fr.FROZEN_RULES} | {fr.FROZEN_EXCEPTION.predictor} <= PRODUCES, \
    "a frozen rule or the exception needs a property predict_gate does not measure"


def gate_properties(ds, k):
    '''The stage-1 inputs for one dataset, all from the ORIGINAL graph; the superseded retention gate is recorded only if built.'''
    p = rule_properties(ds)
    p["avg_clustering"] = round(nx.average_clustering(graph_io.load_graph(cfg.dataset(ds)["edgelist"])), 4)
    p["original_retention"] = (vg_measure(ds, [CANONICAL_SIM], k)[0]["original_retention"]
                               if vg_path(ds, CANONICAL_SIM, k).exists() else float("nan"))
    return p


def predictions(datasets, k):
    '''One row per unseen dataset: the properties, each rule's call, the exception's call, and which of them decided.'''
    rows = []
    for ds in datasets:
        p = gate_properties(ds, k)
        calls, exc, combined, decided_by = fr.predict_gated(p)
        rows.append({"dataset": ds, "k": k, "sim": CANONICAL_SIM,
                     "homophily_adjusted": p["homophily_adjusted"], "largest_component_frac": p["largest_component_frac"],
                     "avg_clustering": p["avg_clustering"], "original_retention": p["original_retention"],
                     "rule1_pred": calls["rule1"], "rule2_pred": calls["rule2"], "exception_pred": exc,
                     "exception_interval": f"({fr.FROZEN_EXCEPTION.interval[0]}, {fr.FROZEN_EXCEPTION.interval[1]})",
                     "decided_by": decided_by, "predicted_verdict": combined})
    return pd.DataFrame(rows)


def freeze(datasets, k):
    '''Write-once pre-registration: already-saved rows are kept EXACTLY as frozen, only unseen datasets are appended.'''
    df = predictions(datasets, k)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if PRED_CSV.exists():
        old = pd.read_csv(PRED_CSV)
        new = df[~df["dataset"].isin(old["dataset"])]
        out = pd.concat([old, new], ignore_index=True)
    else:
        new = out = df
    out.to_csv(PRED_CSV, index=False)
    return out, new["dataset"].tolist()


def score(datasets=None):
    '''Join the pre-training predictions to the LP verdict training actually produced. Never fits.'''
    assert PRED_CSV.exists(), f"{PRED_CSV} missing - run --step predict BEFORE training first."
    pred = pd.read_csv(PRED_CSV)
    m = pred.merge(actual_lp_verdicts(datasets or pred["dataset"].tolist()).rename(columns={"verdict": "actual_verdict"}),
                   on="dataset", how="left")
    # A cell scores only when the experiment actually DECIDED: a tie means the 3-seed noise covers the gap, not that the
    # call was wrong. Four states kept apart: "pending" = untrained; "no decision" = tie or unusable; None = did not fire.
    decided = m["actual_verdict"].isin(["augment", "keep original"]) & m["usable"].fillna(False)
    for col, out in [("rule1_pred", "rule1_correct"), ("rule2_pred", "rule2_correct"),
                     ("exception_pred", "exception_correct"), ("predicted_verdict", "combined_correct")]:
        m[out] = ["pending" if not isinstance(av, str) else "no decision" if not ok
                  else (bool(p == av) if p in ("augment", "keep original") else None)
                  for p, av, ok in zip(m[col], m["actual_verdict"], decided)]
    m["actual_verdict"] = m["actual_verdict"].fillna("pending")
    return m[["dataset", "homophily_adjusted", "avg_clustering", "rule1_pred", "exception_pred", "decided_by",
              "predicted_verdict", "actual_verdict", "rule1_correct", "exception_correct", "combined_correct",
              "original", "best_augmented", "best_variant", "gap_sigma"]]


# Guards the test: predicting a dataset the second condition was FITTED on is not a test, it is the training accuracy.
def _assert_unseen(datasets, allow_panel):
    '''Raise when a dataset the second stage-1 condition was fitted on is passed as a test case.'''
    seen = sorted(set(datasets) & set(fr.GATE_PANEL))
    assert allow_panel or not seen, (f"{seen} are in the GATE_PANEL, so the second condition was FITTED on them - predicting "
                                     "them measures fit, not transfer. Pass --allow-panel for a sanity re-check only.")


def main(args):
    datasets = args.datasets or fr.GATE_HELDOUT
    assert datasets, ("no datasets to predict - pass --datasets, or add the third-set datasets to frozen_rules.GATE_HELDOUT. "
                      "Every registered graph is either in GATE_PANEL (fitted) or genuinely unseen.")
    _assert_unseen(datasets, args.allow_panel)
    if args.step in ("predict", "all"):
        out, added = freeze(datasets, args.k)
        print(f"{len(out):3d} rows -> results/module5_predictions.csv  (froze {len(added)} new: {added or 'none'}; existing rows kept untouched)\n")
        print(out.to_string(index=False))
    if args.step in ("score", "all"):
        m = score(datasets)
        m.to_csv(SCORED_CSV, index=False)
        print(f"\n{len(m):3d} rows -> results/module5_scored.csv\n")
        print(m.to_string(index=False))
        for name, col in [("rule 1 alone", "rule1_correct"), (f"the exception ({fr.FROZEN_EXCEPTION.predictor} "
                          f"{fr.FROZEN_EXCEPTION.op} {fr.FROZEN_EXCEPTION.point})", "exception_correct"),
                          ("the full stage-1 call", "combined_correct")]:
            c = [v for v in m[col] if isinstance(v, bool)]     # bool only: skip pending, no decision and did-not-fire
            skipped = [v for v in m[col] if not isinstance(v, bool)]
            print(f"{name}: " + (f"{sum(c)}/{len(c)} correct" if c else "nothing scored yet")
                  + (f"  [{len(skipped)} not scored: " + ", ".join(sorted({str(v) for v in skipped})) + "]" if skipped else ""))
        # The exception only speaks inside the low-homophily zone; outside it the call is rule 1's veto, unchanged.
        n_exc = int((m["decided_by"] == "clustering exception").sum())
        print(f"\n  the clustering exception actually decided {n_exc}/{len(m)} cells; the rest were rule 1's veto or the no-labels fallback.")


def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Stage 1: pre-register the combined prediction on unseen datasets, then score it after training.")
    p.add_argument('--datasets', nargs='+', default=None, help='Unseen datasets (registered in cfg, NOT in GATE_PANEL). Default: frozen_rules.GATE_HELDOUT.')
    p.add_argument('--k', type=int, default=10, help='Top-K the virtual graph was built with. Default 10 (the locked setting).')
    p.add_argument('--step', default='predict', choices=['predict', 'score', 'all'],
                   help="'predict' = freeze the verdict BEFORE training (the default, and the only step to run first); 'score' = after training.")
    p.add_argument('--allow-panel', action='store_true', help='Permit datasets the second condition was fitted on (a sanity re-check only, never a real test).')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
