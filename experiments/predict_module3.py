'''Module 3, step 1 - PREDICT. Measure each held-out graph, apply the FROZEN rules, and save the verdict BEFORE any training.

Read-only w.r.t. the rules: imports virgo.frozen_rules and never calls threshold() or candidate_rules(), so an unseen
dataset can only be SCORED against the locked rule, never re-fit it. Computes only the two properties the rules need,
straight from the edgelist and labels - no feature cache - so it runs before the pipeline has produced anything.'''

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo import frozen_rules as fr
from experiments.characterize import STUDY, homophily, labels_of

PRED_CSV = cfg.RESULTS_DIR / "module3_predictions.csv"

# What rule_properties() can produce without the feature cache. Guards against a future rule needing a property this
# pre-training measurer does not compute - fix that here rather than silently emitting a wrong prediction.
PRODUCES = {"homophily_adjusted", "largest_component_frac"}
assert {r.predictor for r in fr.FROZEN_RULES} <= PRODUCES, "a frozen rule needs a property predict_module3 does not measure"


def rule_properties(ds):
    '''The two frozen-rule inputs for one dataset, measured cache-free from the original graph and its labels.'''
    d = cfg.dataset(ds)
    G = graph_io.load_graph(d["edgelist"])
    lab = labels_of(d["labels"], G)
    adj_h = homophily(G, lab)[2]                                   # (edge, node, ADJUSTED, null) - the class-balance-corrected one
    return {"dataset": ds, "domain": STUDY.get(ds, {}).get("domain", ""), "tasks": STUDY.get(ds, {}).get("tasks", ""),
            "homophily_adjusted": round(adj_h, 4) if adj_h == adj_h else float("nan"),
            "largest_component_frac": graph_io.properties(G)["largest_component_frac"]}


def predictions(datasets):
    '''One row per held-out dataset: the rule properties, each rule's call and interval, and the combined pre-registered verdict.'''
    rows = []
    for ds in datasets:
        p = rule_properties(ds)
        calls, combined = fr.predict(p)
        row = {"dataset": ds, "domain": p["domain"], "tasks": p["tasks"]}
        for r in fr.FROZEN_RULES:                          # per rule, in order: the property value, its frozen interval, the call
            row[r.predictor] = p[r.predictor]
            row[f"{r.name}_interval"] = f"({r.interval[0]}, {r.interval[1]})"
            row[f"{r.name}_pred"] = calls[r.name]
        row["predicted_verdict"] = combined
        rows.append(row)
    return pd.DataFrame(rows)


def freeze_predictions(datasets):
    '''Write-once pre-registration: keep every already-saved row EXACTLY as it was frozen, append only datasets not yet in the
    file. An existing prediction is never rewritten, so re-running this after training cannot overwrite the before-training proof.'''
    df = predictions(datasets)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if PRED_CSV.exists():
        old = pd.read_csv(PRED_CSV)
        new = df[~df["dataset"].isin(old["dataset"])]
        out = pd.concat([old, new], ignore_index=True)
    else:
        new = out = df
    out.to_csv(PRED_CSV, index=False)
    return out, new["dataset"].tolist()


def main(args):
    datasets = args.datasets or fr.HELDOUT
    assert datasets, "no datasets to predict - pass --datasets, or add the held-out datasets to frozen_rules.HELDOUT."
    if not args.allow_panel:
        seen = sorted(set(datasets) & set(fr.DISCOVERY_PANEL))
        assert not seen, (f"{seen} are in the frozen DISCOVERY panel, not held out - predicting them is not a real Module-3 test. "
                          "Pass --allow-panel for a sanity re-check only.")
    out, added = freeze_predictions(datasets)
    print(f"{len(out):3d} rows in results/module3_predictions.csv  (froze {len(added)} new: {added or 'none'}; existing rows kept untouched)\n")
    print(out.to_string(index=False))


def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Module 3 PREDICT: apply the frozen rules to unseen datasets, before training.")
    p.add_argument('--datasets', nargs='+', default=None, help='Held-out datasets to predict (registered in cfg/STUDY, NOT in the discovery panel). Default: frozen_rules.HELDOUT.')
    p.add_argument('--allow-panel', action='store_true', help='Permit datasets from the fitting panel (a sanity re-check only, never a real test).')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
