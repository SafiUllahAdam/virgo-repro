'''Stage 2, the DEGREE branch: label each graph's winning signal, screen original-graph properties for a degree rule, then test it on the pre-registered held-out graphs.'''
# Module 6 froze one strategy rule (adjusted neighbour predictability high -> centrality) and left the rest undetermined.
# Module 9 (2026-09-01, user) drops psi as a target and asks the narrower question: WHEN DOES A DEGREE ROLE GRAPH WIN?
# The blocker was never method - it was that only two graphs in the 22-graph corpus name degree as their sole winner - so
# six candidate graphs were ingested to add evidence, with the fit/validation split fixed on graph PROPERTIES before any
# of them was trained (docs/paper_log.md 10). This script does the fitting and the one test; it never re-runs the encoder.
#   step 1  label every panel graph with the signal its seeds actually name (strategy_select.winners, sem band)
#   step 2  screen each ORIGINAL-graph property as "degree wins" vs "some other signal wins", under the Module-2 gates
#   step 3  apply the fitted cut, unchanged, to the held-out graphs
# Fitting stays on FIT for the same reason characterize.py does: a rule fitted on a held-out graph cannot be tested on it.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import frozen_rules as fr
from experiments.characterize import GATES, PREDICTORS_EXPLORATORY, PREDICTORS_PRIMARY, loo_threshold, rho, threshold
from experiments.stage1_pairs import chance
from experiments.degree_features import DEGREE_TIER, degree_features
from experiments.strategy_select import MIN_SOLE, SIGNAL, TIERS, loo_majority, properties, winners

# The DEGREE-specific tier is kept OUT of 'primary'/'all' on purpose (user, 2026-09-04): a screen must ask for it by
# name, so no published result can move because degree_features.py exists. 'both' is the only place they ever mix, and
# a rule found there must say which tier it came from.
TIERS_D = dict(TIERS, degree=DEGREE_TIER, both=TIERS["all"] + DEGREE_TIER)

# The pre-registered split (docs/paper_log.md 10), fixed on graph properties BEFORE any of the six was trained: each side
# gets one social and one small web graph, and the largest new graph is deliberately held out so the test is not only tiny
# graphs. texas was built for the stage-1 test and has never been used for stage-2 fitting, so it joins the test side.
NEW_FIT = ["twitch_de", "chameleon", "cornell_webkb"]
NEW_HELDOUT = ["deezer_europe", "wisconsin", "texas"]

# The 2026-09-03 VALIDATION half of the degree pool (cfg.DEGREE_RULE_VALIDATION). Module 10 left one candidate -
# nbr_label_entropy < 0.6724 => degree - fitted on nine cells and tested on exactly one, so what it needs is out-of-sample
# cells on BOTH sides of its cut. These six are ingested for that and may only ever be predicted and scored: --preregister
# writes their predictions before any of them is trained, and --heldout scores the same cuts once the sweep has run.
VALIDATION = cfg.DEGREE_RULE_VALIDATION
CANDIDATE = "nbr_label_entropy"        # Module 10's degree-vs-psi candidate; printed first, still unpromoted
FIT = fr.STRATEGY_PANEL + NEW_FIT
assert not (set(FIT) & set(NEW_HELDOUT)), "a dataset is in BOTH the degree fitting panel and its held-out set"

# chameleon_filtered and texas were scored in a batch that was later withdrawn from the study; the run log kept their
# numbers, so they are read back rather than retrained. Same fallback stage1_pairs.py uses, for the same reason.
WITHDRAWN = cfg.RESULTS_DIR / "module7_withdrawn.csv"


def board(datasets):
    '''The scoreboard rows for these datasets, with the withdrawn run-log rows appended for graphs the board never carried.'''
    b = pd.read_csv(cfg.SCOREBOARD_CSV)
    miss = [d for d in datasets if d not in set(b["dataset"])]
    if miss and WITHDRAWN.exists():
        w = pd.read_csv(WITHDRAWN)
        w = w[w["dataset"].isin(miss)].copy()
        w["encoder"], w["top_K_neighbors"] = "graphsage_edge", 10
        w["task"], w["metric"] = "link prediction (AUC)", "auc"
        w["seeds"] = ["|".join(str(42 + i) for i in range(int(n))) for n in w["seeds"]]
        b = pd.concat([b, w[b.columns.intersection(w.columns)]], ignore_index=True)
    return b


# The paired verdicts tie_break.py writes. Same gap, a tighter estimate of its noise, because every variant is scored on
# the SAME split per seed. Used as an OPTION, never as the default: every published cell uses the frozen band, and on the
# eight augmenting panel graphs the two bands agree exactly - pairing only resolves the corpus's marginal cases.
PAIRED_VERDICTS = cfg.RESULTS_DIR / "degree_rule_corpus_verdicts.csv"


def cells(datasets, band="sem", labels="frozen"):
    '''One row per dataset: the winning signal set, whether DEGREE wins alone, and the original-graph properties.'''
    w = winners(datasets, band, board(datasets)).merge(properties(datasets), on="dataset").merge(degree_features(datasets), on="dataset")
    if labels == "paired":
        assert PAIRED_VERDICTS.exists(), f"{PAIRED_VERDICTS} missing - run experiments/tie_break.py first"
        v = pd.read_csv(PAIRED_VERDICTS).query("band == 'paired'").set_index("dataset")
        miss = sorted(set(w["dataset"]) - set(v.index))
        assert not miss, f"no paired verdict for {miss} - re-score them with tie_break.py, or use --labels frozen"
        w["winner_signals"] = [v.loc[d, "winner_signals"] for d in w["dataset"]]
        w["beats_original"] = [bool(v.loc[d, "beats_original"]) for d in w["dataset"]]
    w["degree_sole"] = [s == "degree" for s in w["winner_signals"]]
    w["names_degree"] = ["degree" in s.split("|") for s in w["winner_signals"]]
    return w.sort_values("dataset").reset_index(drop=True)


# The question is conditional, exactly as FROZEN_STRATEGY is: given that augmentation helps, is the signal degree? Asked
# over every graph instead, "not degree" would silently include "do not augment at all", which is stage 1's question.
def screen(d, tier="all"):
    '''One row per property: does it separate the graphs where degree wins alone from the other augmenting graphs?'''
    z = d[d["beats_original"]].reset_index(drop=True)
    rows = []
    for p in TIERS_D[tier]:
        ok = z[p].notna().to_numpy()
        x, y = z.loc[ok, p].to_numpy(dtype=float), z.loc[ok, "degree_sole"].to_numpy()
        fit = len(x) >= GATES["min_cells"] and y.any() and not y.all() and len(np.unique(x)) > 1
        t, side, err, acc, lo, hi = threshold(x, y) if fit else (np.nan, "", -1, np.nan, np.nan, np.nan)
        l_acc, l_ok, l_n, _, _ = loo_threshold(x, y) if fit else (np.nan, 0, 0, np.nan, np.nan)
        l_maj, _ = loo_majority(x, y) if fit else (np.nan, 0)
        major = round(float(max(y.mean(), 1 - y.mean())), 4) if len(y) else np.nan
        r = rho(pd.Series(x), pd.Series(y.astype(float)))[1]
        rows.append({
            "predictor": p, "tier": ("degree" if p in DEGREE_TIER else
                                     "primary" if p in PREDICTORS_PRIMARY else "exploratory"),
            # Every degree-tier feature needs the role graph BUILT, and the mechanism study found the divergence ones
            # sit at |rho| ~0.9 against density - a property already screened and failed. Report it in the row.
            "rho_vs_density": rho(pd.Series(x), z.loc[ok, "density"].reset_index(drop=True))[1],
            "n_augmenting": int(len(x)), "n_degree": int(y.sum()), "n_missing": int((~ok).sum()),
            "degree_datasets": "|".join(sorted(z.loc[ok, "dataset"][y])),
            "degree_range": f"{x[y].min():.4f}-{x[y].max():.4f}" if y.any() else "",
            "rest_range": f"{x[~y].min():.4f}-{x[~y].max():.4f}" if (~y).any() else "",
            "spearman_rho": r, "threshold": t, "degree_side": side, "n_exceptions": err, "accuracy": acc,
            "interval_lo": lo, "interval_hi": hi,                       # the panel pins an interval, never a 4-decimal point
            "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
            "majority_baseline": major, "loo_majority": l_maj,
            "rule": f"use degree when {p} is {side} ({'>' if side == 'high' else '<'} {t})" if side else "",
            # Same gates the augment rules cleared, plus MIN_SOLE: a group of ties is not evidence about degree.
            "blocked_by": ("too few augmenting cells" if len(x) < GATES["min_cells"] else
                           f"degree wins alone on < {MIN_SOLE} graphs" if int(y.sum()) < MIN_SOLE else ""),
            "credible": bool(p in PREDICTORS_PRIMARY + DEGREE_TIER and r == r and abs(r) >= GATES["min_abs_rho"]
                             and 0 <= err <= GATES["max_exceptions"] and len(x) >= GATES["min_cells"]
                             and int(y.sum()) >= MIN_SOLE
                             and (l_acc == l_acc and l_acc > major and l_acc > l_maj if GATES["loo_above_majority"] else True)),
        })
    return pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False).reset_index(drop=True)


# The narrower contrast, which only became fittable when twitch_de landed on psi: degree vs psi among the graphs that
# name ONE of those two, ignoring the centrality cut. n is 4, exactly the Module-2 minimum, and with two against two a
# perfect split is cheap - chance() prices that, and the price is the reason nothing here is promotable.
def pairwise(d, tier="all"):
    '''Degree vs psi over the graphs naming one of them alone: split, interval, LOO, and how many separations chance alone buys.'''
    z = d[[s in ("degree", "psi") for s in d["winner_signals"]]].reset_index(drop=True)
    tried = [p for p in TIERS_D[tier] if z[p].notna().all() and len(np.unique(z[p])) > 1]
    rows = []
    for p in tried:
        x, y = z[p].to_numpy(dtype=float), np.array([s == "degree" for s in z["winner_signals"]])
        t, side, err, acc, lo, hi = threshold(x, y)
        l_acc, l_ok, l_n, _, _ = loo_threshold(x, y)
        pc, free = chance(len(x), int(y.sum()), len(tried))
        rows.append({"predictor": p, "tier": ("degree" if p in DEGREE_TIER else
                                              "primary" if p in PREDICTORS_PRIMARY else "exploratory"),
                     "rho_vs_density": rho(pd.Series(x), z["density"])[1],
                     "n_cells": len(x), "n_degree": int(y.sum()), "n_psi": int((~y).sum()),
                     "degree_values": "|".join(f"{v:.4f}" for v in x[y]), "psi_values": "|".join(f"{v:.4f}" for v in x[~y]),
                     "spearman_rho": rho(pd.Series(x), pd.Series(y.astype(float)))[1],
                     "threshold": t, "degree_side": side, "n_exceptions": err, "accuracy": acc,
                     "interval_lo": lo, "interval_hi": hi, "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
                     "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 4),
                     # At n=4 a clean split is worth 1/3 per property; with this many properties several are FREE.
                     "separation_by_chance": pc, "expected_free_winners": free,
                     "rule": f"use degree when {p} is {side} ({'>' if side == 'high' else '<'} {t})" if side else ""})
    return pd.DataFrame(rows).sort_values(["loo_accuracy", "spearman_rho"], key=abs, ascending=False).reset_index(drop=True)


# Pre-registration, not a test. Every cut here is a function of ORIGINAL-graph properties, which exist before any
# encoder runs, so the prediction can be on disk before the graph is trained - the same order stage 1 uses. heldout()
# scores these graphs later; it needs scoreboard rows, which is exactly what must not exist yet.
def prereg(datasets, scr, pair):
    '''Every fitted cut applied to graphs that are not yet trained: one predicted signal per rule, with no outcome column.'''
    props = properties(datasets)
    rows = []
    for contrast, df in [("degree_vs_rest", scr), ("degree_vs_psi", pair)]:
        for _, r in df[df["degree_side"] != ""].iterrows():
            for _, c in props.iterrows():
                v = c[r["predictor"]]
                rows.append({"contrast": contrast, "predictor": r["predictor"], "tier": r["tier"], "rule": r["rule"],
                             "threshold": r["threshold"], "degree_side": r["degree_side"], "dataset": c["dataset"],
                             "value": v, "predicted": ("n/a" if v != v else "degree" if
                                                       (v > r["threshold"] if r["degree_side"] == "high" else v < r["threshold"])
                                                       else ("psi" if contrast == "degree_vs_psi" else "not degree"))})
    return pd.DataFrame(rows)


# Scores BOTH contrasts, because they are scorable on different cells: degree-vs-rest counts any augmenting graph that
# names one signal, while degree-vs-psi is only defined where the sole winner IS degree or psi - a centrality graph says
# nothing about it. Scoring the candidate against the wrong denominator is how a rule gets credit it never earned.
def heldout(d, scr, pair):
    '''Apply every fitted cut, unchanged, to the held-out graphs: predicted signal against the one their seeds name.'''
    rows = []
    for contrast, df in [("degree_vs_rest", scr), ("degree_vs_psi", pair)]:
        for _, r in df[df["degree_side"] != ""].iterrows():
            for _, c in d.iterrows():
                v, sig = c[r["predictor"]], c["winner_signals"]
                other = "psi" if contrast == "degree_vs_psi" else "not degree"
                pred = ("n/a" if v != v else
                        "degree" if (v > r["threshold"] if r["degree_side"] == "high" else v < r["threshold"]) else other)
                if contrast == "degree_vs_psi":
                    actual = sig if (c["beats_original"] and sig in ("degree", "psi")) else "not applicable"
                    scored = actual in ("degree", "psi")
                else:
                    actual = ("no augmentation" if not c["beats_original"] else
                              "degree" if c["degree_sole"] else "tie" if len(sig.split("|")) > 1 else "not degree")
                    scored = actual in ("degree", "not degree")
                rows.append({"contrast": contrast, "predictor": r["predictor"], "tier": r["tier"], "rule": r["rule"],
                             "dataset": c["dataset"], "value": v, "predicted": pred, "winner_signals": sig,
                             "actual": actual, "seeds": c["seeds"],
                             # Only a cell the contrast is DEFINED on can score it; the rest are reported, not counted.
                             "scored": bool(scored),
                             "correct": bool(pred == actual) if scored and pred != "n/a" else None})
    return pd.DataFrame(rows)


def report(fit, scr, pair, test, pre, out):
    '''Print the winner labels, the screen, and the held-out test - naming the blocking gate when nothing can be fitted.'''
    print("\nSTEP 1 - WINNING SIGNAL PER FITTING DATASET (sem band; seed count differs by batch and is printed)")
    print(fit[["dataset", "seeds", "original", "best_variant", "best_score", "winner_signals", "beats_original",
               "degree_sole"]].to_string(index=False))
    aug, deg = int(fit["beats_original"].sum()), int(fit["degree_sole"].sum())
    print(f"\nCEILING  {len(fit)} datasets  |  {aug} augment  |  {deg} name DEGREE alone "
          f"({', '.join(fit['dataset'][fit['degree_sole']]) or 'none'})  |  gate needs {GATES['min_cells']} augmenting cells and {MIN_SOLE} degree wins")
    print("\nSTEP 2 - PROPERTY -> DEGREE SCREEN (augmenting graphs only, Module-2 gates)")
    print(scr[["predictor", "tier", "n_augmenting", "n_degree", "degree_range", "rest_range", "spearman_rho",
               "rho_vs_density", "threshold", "degree_side", "n_exceptions", "loo_accuracy", "loo_folds",
               "majority_baseline", "loo_majority", "blocked_by", "credible"]].to_string(index=False))
    good = scr[scr["credible"]]
    print("\nCREDIBLE DEGREE RULES: " + ("none" + (f" - {scr['blocked_by'].iloc[0]}" if scr["blocked_by"].iloc[0] else
                                                   " - no property clears the gates") if good.empty else ""))
    for _, r in good.iterrows():
        print(f"  {r['rule']}   [interval {r['interval_lo']}-{r['interval_hi']}, rho {r['spearman_rho']}, "
              f"LOO {r['loo_correct']}/{r['loo_folds']} vs {r['majority_baseline']} majority]")
    print(f"\nSTEP 2b - DEGREE vs PSI among the {len(pair) and int(pair['n_cells'].iloc[0])} graphs naming one of them alone "
          f"(chance of a clean split per property: {pair['separation_by_chance'].iloc[0] if len(pair) else float('nan')}, "
          f"so ~{pair['expected_free_winners'].iloc[0] if len(pair) else float('nan')} of the {len(pair)} tried separate for free)")
    if not pair.empty:
        print(pair[["predictor", "tier", "spearman_rho", "rho_vs_density", "threshold", "degree_side",
                    "n_exceptions", "interval_lo", "interval_hi", "loo_accuracy", "loo_folds", "majority_baseline"]].to_string(index=False))
    if not pre.empty:
        print(f"\nSTEP 3a - PRE-REGISTERED PREDICTIONS for {', '.join(sorted(set(pre['dataset'])))} - written BEFORE these graphs are trained")
        head = pre[pre["predictor"] == CANDIDATE]
        print((head if not head.empty else pre)[["contrast", "predictor", "rule", "dataset", "value", "predicted"]].to_string(index=False))
    if not test.empty:
        print(f"\nSTEP 3 - HELD-OUT TEST ({', '.join(sorted(set(test['dataset'])))}) - fitted cuts applied unchanged")
        head = test[test["predictor"] == CANDIDATE]
        print((head if not head.empty else test)[["contrast", "dataset", "value", "predicted", "winner_signals",
                                                  "actual", "seeds", "scored", "correct"]].to_string(index=False))
        for (contrast, p), g in test.groupby(["contrast", "predictor"]):
            c = [v for v in g["correct"] if isinstance(v, bool)]
            print(f"  {contrast:15s} {p}: " + (f"{sum(c)}/{len(c)} correct" if c else "nothing scorable on this contrast"))
    print(f"\n{len(out):3d} rows written to results/degree_*.csv")


# Guards the test: fitting on a held-out graph would let the test set move the rule that is meant to test it.
def _assert_fit(datasets, allow_refit):
    '''Raise when a held-out or validation dataset is passed as a fitting dataset.'''
    extra = sorted(set(datasets) & (set(NEW_HELDOUT) | set(VALIDATION)))
    assert allow_refit or not extra, (f"{extra} are held out for the degree test, so they may not be fitted on. "
                                      "Pass --allow-refit to break the split deliberately.")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _assert_fit(args.datasets, args.allow_refit)
    fit = cells(args.datasets, args.band, args.labels)
    scr, pair = screen(fit, args.tier), pairwise(fit[fit["beats_original"]], args.tier)
    test = heldout(cells(args.heldout, args.band, args.labels), scr, pair) if args.heldout else pd.DataFrame()
    pre = prereg(args.validation, scr, pair) if args.preregister else pd.DataFrame()
    prefix = ("exploratory_" if args.allow_refit else "") + ("paired_" if args.labels == "paired" else "")
    out = []
    for name, df in [("degree_cells", fit), ("degree_rules", scr), ("degree_psi_contrast", pair), ("degree_heldout", test),
                     ("degree_validation_prereg", pre)]:
        if not df.empty:
            df.to_csv(cfg.RESULTS_DIR / f"{prefix}{name}.csv", index=False)
            out.append(name)
    report(fit, scr, pair, test, pre, out)


# Defines command-line options (mirrors strategy_select.py / stage1_pairs.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Stage 2, the degree branch: fit a degree-augmentation rule and test it on the pre-registered held-out graphs.")
    p.add_argument('--datasets', nargs='+', default=FIT, help='Fitting datasets. Default: the strategy panel plus the new degree-batch fitting graphs.')
    p.add_argument('--heldout', nargs='+', default=NEW_HELDOUT, help='Graphs the rule is TESTED on, never fitted. Default: the pre-registered three.')
    p.add_argument('--tier', default='all', choices=list(TIERS_D),
                   help="'primary' = the general properties that may become a rule; 'all' adds the exploratory tier (default); "
                        "'degree' = ONLY the degree-specific tier (tie structure, degree resolution, role-graph stability, "
                        "divergence from psi); 'both' = all of them together.")
    p.add_argument('--band', default='sem', choices=['sem', 'sigma'], help="How close counts as tied. Default sem, the band Module 6 fitted under.")
    p.add_argument('--labels', default='frozen', choices=['frozen', 'paired'],
                   help="Which band names the winning signal. 'frozen' = the independent-means band every published cell uses "
                        "(default); 'paired' = tie_break.py's per-seed comparison, which resolves marginal cells and agrees "
                        "with the frozen band on all eight augmenting panel graphs.")
    p.add_argument('--validation', nargs='+', default=VALIDATION, help='The degree pool\'s validation half. Predicted by --preregister, scored by passing the same names to --heldout once they are trained.')
    p.add_argument('--preregister', action='store_true', help='Write every fitted cut\'s prediction for --validation to results/degree_validation_prereg.csv. Run this BEFORE training them.')
    p.add_argument('--allow-refit', action='store_true', help='Deliberately fit on a held-out graph. Writes exploratory_*.csv so the panel tables are never overwritten.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
