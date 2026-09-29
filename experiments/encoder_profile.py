'''Module 16: WHICH graphs still need augmentation once the encoder is strong, and which do not - a property profile.'''
# The supervisor's question after Module 15 (2026-09-17): "when the graph gives the useful information by its
# construction, there is no need for augmentation... try to give deeper insight on what properties the graphs have in
# both cases". Module 15 measured THAT the two groups exist; this asks WHAT separates them.
# Two labels, kept apart because they are different questions:
#   helps   - does augmentation beat GATv2's OWN original graph (13 of 30)? the direct "is augmentation useful here".
#   needs   - the Q2 extremes only: GATv2 alone LOSES to GraphSAGE+aug (augmentation carries the graph) vs GATv2 alone
#             BEATS it (the graph carries itself). The 7 no-difference cells are dropped, so this is 23 cells, and it is
#             the cleaner contrast because both sides are decided by the paired band rather than by a single arm.
# THIS IS DESCRIPTION, NOT A RULE SEARCH. The question is what the two cases have in common, on the graphs already
# measured - nothing is frozen, nothing is proposed as a predictor, and no validation set is owed. The threshold column
# is reported only to say how cleanly each property separates the two groups, never as a cut to apply to a new graph.
# Properties are REUSED from the tables the earlier modules measured, never re-measured, so they cannot drift.

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from experiments import encoder_transfer as et
from experiments.characterize import GATES, PREDICTORS, PREDICTORS_PRIMARY, loo_threshold, rho, threshold
from experiments.stage1_pairs import chance

PROFILE_CSV = cfg.RESULTS_DIR / "encoder_profile_groups.csv"
RULES_CSV = cfg.RESULTS_DIR / "encoder_profile_rules.csv"
CELLS_CSV = cfg.RESULTS_DIR / "encoder_profile_cells.csv"


# Properties of the ORIGINAL graph, taken from the tables earlier modules already measured - same numbers, no drift.
def properties(datasets):
    '''One row per dataset with the 15 screened properties, gathered from results/*.csv and measured only if absent.'''
    out = {}
    for f in sorted(cfg.RESULTS_DIR.glob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "dataset" not in d or not set(PREDICTORS) <= set(d.columns):
            continue
        for ds, row in d.drop_duplicates("dataset").set_index("dataset")[PREDICTORS].iterrows():
            out.setdefault(ds, row)
    missing = [d for d in datasets if d not in out]
    if missing:                                                    # measured the same way characterize.py measures them
        from experiments.characterize import graph_properties
        for ds in missing:
            out[ds] = pd.Series(graph_properties(ds))[PREDICTORS]
    return pd.DataFrame({d: out[d] for d in datasets}).T.astype(float).rename_axis("dataset").reset_index()


# The two labels, both read off the Module-15 tables - no retraining, no rescoring.
def cells(datasets):
    '''Per dataset: the two augmentation labels plus the 15 original-graph properties.'''
    fc = pd.read_csv(cfg.RESULTS_DIR / "encoder_transfer_fourcell.csv")
    ps = pd.read_csv(cfg.RESULTS_DIR / "encoder_transfer_perseed.csv")
    gv = fc[fc.encoder == "gatv2_edge"].set_index("dataset")
    cr = et.cross(ps, "gatv2_edge", "graphsage_edge").set_index("dataset")
    band = cr.paired_ratio.apply(lambda r: "beats" if r > 1 else ("no difference" if abs(r) <= 1 else "loses"))
    c = properties(datasets).set_index("dataset")
    c["gatv2_alone"] = band
    c["helps"] = gv.aug_wins.astype(bool)                          # augmentation beats GATv2's own original graph
    c["needs"] = band.map({"loses": True, "beats": False})         # NaN on the 7 no-difference cells -> dropped by screen
    c["gatv2_original"] = gv.original
    c["gatv2_best_aug"] = gv.best_aug
    return c.reset_index()


# What the supervisor asked for: the two groups side by side, property by property.
def profile(c, label):
    '''Per property: the median in each group, the gap, and how strongly it ranks against the label.'''
    d = c[c[label].notna()].copy()
    y = d[label].astype(bool)
    rows = []
    for p in PREDICTORS:
        a, b = d.loc[y, p], d.loc[~y, p]
        n, r, pv = rho(d[p], y.astype(float))
        rows.append({"property": p, "tier": "primary" if p in PREDICTORS_PRIMARY else "exploratory",
                     "augmentation_helps": round(float(a.median()), 4), "does_not": round(float(b.median()), 4),
                     "ratio": round(float(a.median() / b.median()), 2) if b.median() else float("nan"),
                     "rho_vs_label": r, "p": pv, "n": n})
    return pd.DataFrame(rows).reindex(pd.DataFrame(rows).rho_vs_label.abs().sort_values(ascending=False).index)


# Same screen every earlier module ran: split, interval, LOO refit, majority baseline, chance pricing.
def screen(c, label):
    '''Per property: the best single-threshold rule for this label and whether it clears the Module-2 gates.'''
    d = c[c[label].notna()].reset_index(drop=True)
    y = d[label].astype(bool).to_numpy()
    base = max(y.mean(), 1 - y.mean())
    pc, exp = chance(len(y), int(y.sum()), len(PREDICTORS))
    rows = []
    for p in PREDICTORS:
        s = d[d[p].notna()].reset_index(drop=True)
        yy = s[label].astype(bool).to_numpy()
        n, r, pv = rho(s[p], pd.Series(yy, dtype=float))
        t, side, err, acc, lo, hi = threshold(s[p].to_numpy(), yy)
        la, ok, folds, cl, ch = loo_threshold(s[p].to_numpy(), yy)
        rows.append({"property": p, "tier": "primary" if p in PREDICTORS_PRIMARY else "exploratory",
                     "rho": r, "p": pv, "cut": t, "side_that_needs_aug": side, "exceptions": err,
                     "interval_lo": lo, "interval_hi": hi, "loo": la, "loo_folds": folds,
                     "majority_baseline": round(base, 4),
                     "credible": bool(abs(r) >= GATES["min_abs_rho"] if r == r else False)
                                  and err <= GATES["max_exceptions"] and len(s) >= GATES["min_cells"]
                                  and (la > base if la == la else False)})
    out = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False).reset_index(drop=True)
    out.attrs["chance"] = (len(y), int(y.sum()), pc, exp, base)
    return out


# The strongest hits are all degree-spread measures, so report whether they are one finding or several.
def collinearity(c, label, props):
    '''Spearman between the top-ranked properties, so a single underlying property cannot be counted three times.'''
    d = c[c[label].notna()]
    return d[props].corr(method="spearman").round(3)


def report(c, label, prof, rules):
    '''Print the group profile, then the rule screen and what chance alone would have produced.'''
    n, a, pc, exp, base = rules.attrs["chance"]
    print(f"\n{'='*100}\nLABEL '{label}': {a} of {n} graphs need augmentation (majority baseline {base:.3f})")
    print(f"\nGROUP PROFILE - median per property")
    print(prof.to_string(index=False))
    print(f"\nHOW CLEANLY EACH PROPERTY SEPARATES THE TWO GROUPS (descriptive; no rule is proposed)")
    print(rules.to_string(index=False))
    top = list(rules.head(4).property)
    print(f"\nCOLLINEARITY among the top 4 - are these one finding or four?")
    print(collinearity(c, label, top).to_string())
    clean = rules[rules.exceptions <= 1]
    print(f"\nSeparating with at most one exception: {len(clean)} of {len(rules)}  {list(clean.property) if len(clean) else '(none)'}")
    pc_exact = 2 / math.comb(n, a)
    print(f"CHANCE: one candidate separates {n} cells with {a} positives by luck at P = {pc_exact:.2e}; "
          f"over {len(rules)} properties that is {pc_exact * len(rules):.2e} free winners - negligible at this n, so "
          f"the pattern is real, but it is a difference of degree rather than a clean dividing line.")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    c = cells(args.datasets)
    c.to_csv(CELLS_CSV, index=False)
    print(f"{len(c)} graphs -> results/{CELLS_CSV.name}")
    profs, rules = [], []
    for label in ["helps", "needs"]:
        pr, rl = profile(c, label), screen(c, label)
        report(c, label, pr, rl)
        profs.append(pr.assign(label=label)); rules.append(rl.assign(label=label))
    pd.concat(profs).to_csv(PROFILE_CSV, index=False)
    pd.concat(rules).to_csv(RULES_CSV, index=False)
    print(f"\n-> results/{PROFILE_CSV.name}, results/{RULES_CSV.name}")
    print("DESCRIPTIVE ONLY: this characterizes the graphs already measured; no rule is proposed and none is owed a test set.")


# Defines command-line options (mirrors characterize.py / stage1_pairs.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Profile the graphs where augmentation is still needed under a strong encoder.")
    p.add_argument('--datasets', nargs='+', default=cfg.ENCODER_PANEL, help='Default: the Module-15 panel.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
