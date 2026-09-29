'''Stage 2, the PSI branch: when does a psi role graph win, and does the centrality -> psi -> degree cascade hold?'''
# Module 6 froze one strategy rule (adjusted neighbour predictability high -> centrality). Modules 9-12 then asked the
# degree question six ways and returned six negatives, so 2026-09-07 (user) closes the direct degree search and moves the
# target to PSI: find a clean ORIGINAL-graph psi characteristic the same way the centrality rule was found, and test the
# cascade it implies - centrality condition -> centrality, else psi condition -> psi, with DEGREE as the residual region.
#   step 1  label every graph with the signal its seeds name (strategy_select.winners, sem band)
#   step 2  screen each original-graph property as "psi wins" vs "some other signal wins", over the AUGMENTING graphs only
#   step 3  partition by FROZEN_STRATEGY, which is SETTLED and out of scope (user, 2026-09-07): it is read only to mark
#           off the residual region the psi question lives in, never scored, re-fitted or re-tested here
#   step 4  apply every fitted cut to the held-out graphs
# Two label modes are screened, because the centrality rule was fitted on MEMBERSHIP (psi is in the winning band) while
# the degree work labelled SOLE winners. Reporting one and not the other would let the choice of label pick the answer.

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
from experiments.degree_rule import TIERS_D, cells
from experiments.degree_features import DEGREE_TIER
from experiments.strategy_select import MIN_SOLE, loo_majority, properties

# The pool is the 48 graphs tie_break.py has paired verdicts for, so BOTH label bands describe the same cells and a
# result cannot move by silently changing which graphs are in scope. DISCLOSURE: seven strategy-panel graphs (cora,
# enzymes, minesweeper, amazon_photo, amazon_ratings, citeseer_linqs, proteins) and the four LINKX networks are outside
# it. The seven cost nothing - all seven KEEP the original graph, so a conditional screen would drop them anyway - and
# the LINKX four have only five of the seven variants scored, so their band is not comparable to the rest.
POOL = sorted(set(pd.read_csv(cfg.RESULTS_DIR / "degree_rule_corpus_verdicts.csv")["dataset"]))

# The fit/test split is by PROVENANCE, never by outcome: everything ingested for Module 6 or the Module-10 corpus fits,
# everything ingested later only tests. It is RETROSPECTIVE - all 48 were trained before this screen existed - so it is
# transfer evidence, exactly like the Module-8 clustering exception, and must never be called a pre-registration.
PSI_HELDOUT = [d for d in cfg.DEGREE_RULE_VALIDATION + cfg.DEGREE_TEST + cfg.DEGREE_BATCH2 if d in POOL]
PSI_FIT = [d for d in POOL if d not in PSI_HELDOUT]

# Which winning-band label counts as a psi win. "sole" is the strict one the degree work used; "member" is what the
# centrality rule was actually fitted on, so it is the like-for-like target here.
TARGETS = {"sole": "psi_sole", "member": "psi_member"}

# A candidate is CONFOUNDED when it ranks the graphs almost exactly as a property Module 2 already screened and dropped.
# Set by the 2026-09-07 tail study, where rewrite_share and k_over_degree sat at -0.95 and -0.92 against avg_degree and
# the screen was abandoned before it ran: below this, a "new" property is a renamed old one.
CONFOUND = {"avg_degree": 0.9, "density": 0.9}

# residual() can legitimately return nothing - below the centrality cut there are often fewer than min_cells decided
# graphs - so its schema is declared here rather than inferred from a row that may never exist.
RESIDUAL_COLS = ["zone", "predictor", "tier", "n_cells", "n_psi", "n_degree", "psi_values", "degree_values",
                 "spearman_rho", "threshold", "psi_side", "n_exceptions", "accuracy", "interval_lo", "interval_hi",
                 "loo_accuracy", "loo_correct", "loo_folds", "majority_baseline", "separation_by_chance",
                 "expected_free_winners"] + [f"rho_vs_{c}" for c in CONFOUND] + ["rule"]


def label(d):
    '''Adds the two psi targets to a cells() frame: psi wins alone, and psi is named by the winning band.'''
    d["psi_sole"] = [s == "psi" for s in d["winner_signals"]]
    d["psi_member"] = [("psi" in s.split("|")) for s in d["winner_signals"]]
    return d


# The question is conditional, exactly as FROZEN_STRATEGY is: given that augmentation helps, is the signal psi? Asked
# over every graph instead, "not psi" would silently include "do not augment at all", which is stage 1's question.
def screen(d, tier="all"):
    '''One row per target x property: does it separate the graphs psi wins on from the other augmenting graphs?'''
    z = d[d["beats_original"]].reset_index(drop=True)
    rows = []
    for target, col in TARGETS.items():
        tried = [p for p in TIERS_D[tier] if z[p].notna().sum() >= GATES["min_cells"] and z[p].nunique() > 1]
        for p in tried:
            ok = z[p].notna().to_numpy()
            x, y = z.loc[ok, p].to_numpy(dtype=float), z.loc[ok, col].to_numpy()
            sole = z.loc[ok, "psi_sole"].to_numpy()
            fit = len(x) >= GATES["min_cells"] and y.any() and not y.all()
            t, side, err, acc, lo, hi = threshold(x, y) if fit else (np.nan, "", -1, np.nan, np.nan, np.nan)
            l_acc, l_ok, l_n, _, _ = loo_threshold(x, y) if fit else (np.nan, 0, 0, np.nan, np.nan)
            l_maj, _ = loo_majority(x, y) if fit else (np.nan, 0)
            major = round(float(max(y.mean(), 1 - y.mean())), 4) if len(y) else np.nan
            r = rho(pd.Series(x), pd.Series(y.astype(float)))[1]
            conf = {c: (np.nan if c == p else rho(pd.Series(x), z.loc[ok, c].reset_index(drop=True))[1]) for c in CONFOUND}
            pc, free = chance(len(x), int(y.sum()), len(tried))
            rows.append({
                "target": target, "predictor": p,
                "tier": ("degree" if p in DEGREE_TIER else "primary" if p in PREDICTORS_PRIMARY else "exploratory"),
                "n_augmenting": int(len(x)), "n_psi": int(y.sum()), "n_psi_sole": int(sole.sum()),
                "psi_datasets": "|".join(sorted(z.loc[ok, "dataset"][y])),
                "psi_range": f"{x[y].min():.4f}-{x[y].max():.4f}" if y.any() else "",
                "rest_range": f"{x[~y].min():.4f}-{x[~y].max():.4f}" if (~y).any() else "",
                "spearman_rho": r, "threshold": t, "psi_side": side, "n_exceptions": err, "accuracy": acc,
                "interval_lo": lo, "interval_hi": hi,                # the panel pins an interval, never a 4-decimal point
                "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
                "majority_baseline": major, "loo_majority": l_maj,
                # How many of the properties tried separate this many positives for free - the price the degree branch never paid.
                "separation_by_chance": pc, "expected_free_winners": free,
                **{f"rho_vs_{c}": conf[c] for c in CONFOUND},
                # A property that ranks the graphs like one Module 2 already dropped is that property under a new name.
                "confounded": bool(any(conf[c] == conf[c] and abs(conf[c]) >= lim for c, lim in CONFOUND.items())),
                "rule": f"use psi when {p} is {side} ({'>' if side == 'high' else '<'} {t})" if side else "",
                "blocked_by": ("too few augmenting cells" if len(x) < GATES["min_cells"] else
                               f"psi wins alone on < {MIN_SOLE} graphs" if int(sole.sum()) < MIN_SOLE else ""),
                # Same gates the augment rules cleared, plus MIN_SOLE: a group of ties is not evidence about psi.
                "credible": bool(p in PREDICTORS_PRIMARY + DEGREE_TIER and r == r and abs(r) >= GATES["min_abs_rho"]
                                 and 0 <= err <= GATES["max_exceptions"] and len(x) >= GATES["min_cells"]
                                 and int(sole.sum()) >= MIN_SOLE
                                 and (l_acc == l_acc and l_acc > major and l_acc > l_maj if GATES["loo_above_majority"] else True)),
            })
    return pd.DataFrame(rows).sort_values(["target", "spearman_rho"], key=lambda s: abs(s) if s.name == "spearman_rho" else s,
                                          ascending=[True, False]).reset_index(drop=True)


# STEP 3: the cascade the user proposed is centrality first, psi second, degree as whatever is left. This function only
# draws the LINE - which graphs FROZEN_STRATEGY routes to centrality and which fall through to the psi question. The
# centrality rule itself is settled and out of scope (user, 2026-09-07): it is applied, never scored against a baseline.
def cascade(d):
    '''Per augmenting graph: the branch FROZEN_STRATEGY routes it to, and the signal its seeds name.'''
    z = d[d["beats_original"]].reset_index(drop=True)
    v = z[fr.FROZEN_STRATEGY.predictor]
    rows = pd.DataFrame({
        "dataset": z["dataset"], "anp": v, "winner_signals": z["winner_signals"],
        "branch": np.where(v.isna(), "n/a", np.where(v > fr.FROZEN_STRATEGY.point, "centrality", "residual")),
        "decided": [s in ("psi", "degree", "centrality") for s in z["winner_signals"]],
    })
    rows["actual"] = np.where(rows["decided"], rows["winner_signals"], "tie")
    return rows


# The residual is the whole point of the cascade: below the centrality cut, is it psi or degree? n is small there by
# construction - the cut is low enough that most labelled graphs clear it - so chance() prices every split, and the
# residual is ALSO reported unconditionally (psi vs degree over every graph naming one of them) so the two are comparable.
def residual(d, tier="all", zone="cut"):
    '''Psi vs degree, either inside the below-cut residual or over every graph naming one of them: split, LOO, chance price.'''
    z = d[d["beats_original"] & d["winner_signals"].isin(["psi", "degree"])].reset_index(drop=True)
    if zone == "cut":
        z = z[z[fr.FROZEN_STRATEGY.predictor] <= fr.FROZEN_STRATEGY.point].reset_index(drop=True)
    tried = [p for p in TIERS_D[tier] if z[p].notna().all() and z[p].nunique() > 1]
    rows = []
    for p in tried:
        x, y = z[p].to_numpy(dtype=float), (z["winner_signals"] == "psi").to_numpy()
        if len(x) < GATES["min_cells"] or not y.any() or y.all():
            continue
        t, side, err, acc, lo, hi = threshold(x, y)
        l_acc, l_ok, l_n, _, _ = loo_threshold(x, y)
        pc, free = chance(len(x), int(y.sum()), len(tried))
        rows.append({"zone": zone, "predictor": p,
                     "tier": ("degree" if p in DEGREE_TIER else "primary" if p in PREDICTORS_PRIMARY else "exploratory"),
                     "n_cells": len(x), "n_psi": int(y.sum()), "n_degree": int((~y).sum()),
                     "psi_values": "|".join(f"{v:.4f}" for v in x[y]),
                     "degree_values": "|".join(f"{v:.4f}" for v in x[~y]),
                     "spearman_rho": rho(pd.Series(x), pd.Series(y.astype(float)))[1],
                     "threshold": t, "psi_side": side, "n_exceptions": err, "accuracy": acc,
                     "interval_lo": lo, "interval_hi": hi, "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
                     "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 4),
                     "separation_by_chance": pc, "expected_free_winners": free,
                     **{f"rho_vs_{c}": (np.nan if c == p else rho(pd.Series(x), z[c])[1]) for c in CONFOUND},
                     "rule": f"use psi when {p} is {side} ({'>' if side == 'high' else '<'} {t})" if side else ""})
    out = pd.DataFrame(rows, columns=RESIDUAL_COLS)
    return out.sort_values("spearman_rho", key=abs, ascending=False).reset_index(drop=True)


# Applies every fitted cut, unchanged, to graphs no cut has seen. Scored only where the contrast is DEFINED: a
# centrality graph says nothing about psi versus degree, and a tie says nothing at all.
def heldout(d, scr, res):
    '''Each fitted cut applied to the held-out graphs: predicted signal against the one their seeds name.'''
    rows = []
    for contrast, df in [("psi_vs_rest", scr), ("psi_vs_degree", res)]:
        for _, r in df[df["psi_side"] != ""].iterrows():
            for _, c in d.iterrows():
                v, sig = c[r["predictor"]], c["winner_signals"]
                other = "degree" if contrast == "psi_vs_degree" else "not psi"
                pred = ("n/a" if v != v else
                        "psi" if (v > r["threshold"] if r["psi_side"] == "high" else v < r["threshold"]) else other)
                if contrast == "psi_vs_degree":
                    actual = sig if (c["beats_original"] and sig in ("psi", "degree")) else "not applicable"
                    scored = actual in ("psi", "degree")
                else:
                    actual = ("no augmentation" if not c["beats_original"] else
                              "psi" if c["psi_sole"] else "tie" if len(sig.split("|")) > 1 else "not psi")
                    scored = actual in ("psi", "not psi")
                rows.append({"contrast": contrast, "target": r.get("target", ""), "predictor": r["predictor"],
                             "tier": r["tier"], "rule": r["rule"], "dataset": c["dataset"], "value": v,
                             "predicted": pred, "winner_signals": sig, "actual": actual, "seeds": c["seeds"],
                             "scored": bool(scored),
                             "correct": bool(pred == actual) if scored and pred != "n/a" else None})
    return pd.DataFrame(rows)


def report(fit, scr, cas, res, test, out):
    '''Print the labels, the two screens, the cascade against its constant baselines, and the held-out test.'''
    aug = fit[fit["beats_original"]]
    print(f"\nSTEP 1 - WINNING SIGNAL PER FITTING GRAPH  ({len(fit)} graphs, {len(aug)} augment)")
    print(fit[["dataset", "seeds", "original", "best_variant", "best_score", "winner_signals", "beats_original",
               "psi_sole", "psi_member"]].to_string(index=False))
    print(f"\nCEILING  {int(aug['psi_sole'].sum())} augmenting graphs name PSI alone "
          f"({', '.join(aug['dataset'][aug['psi_sole']]) or 'none'}); {int(aug['psi_member'].sum())} name it at all; "
          f"gate needs {GATES['min_cells']} augmenting cells and {MIN_SOLE} psi wins")
    for target in TARGETS:
        s = scr[scr["target"] == target]
        print(f"\nSTEP 2 - PROPERTY -> PSI SCREEN, target '{target}' (augmenting graphs only, Module-2 gates)")
        print(s[["predictor", "tier", "n_augmenting", "n_psi", "psi_range", "rest_range", "spearman_rho",
                 "rho_vs_avg_degree", "rho_vs_density", "confounded", "threshold", "psi_side", "n_exceptions",
                 "loo_accuracy", "majority_baseline", "loo_majority", "credible"]].to_string(index=False))
        good = s[s["credible"] & ~s["confounded"]]
        print(f"CREDIBLE PSI RULES ({target}): " + ("none - no property clears the gates" if good.empty else ""))
        for _, r in good.iterrows():
            print(f"  {r['rule']}   [interval {r['interval_lo']}-{r['interval_hi']}, rho {r['spearman_rho']}, "
                  f"LOO {r['loo_correct']}/{r['loo_folds']} vs {r['majority_baseline']} majority]")
    dec = cas[cas["decided"]]
    print(f"\nSTEP 3 - THE RESIDUAL: FROZEN_STRATEGY (settled, applied not scored) routes {len(cas)} augmenting graphs; {len(dec)} name one signal")
    print(cas[["dataset", "anp", "branch", "winner_signals", "actual"]].to_string(index=False))
    below = dec[dec["branch"] == "residual"]
    print(f"  residual region (below the cut): {len(below)} decided cells"
          + (f", signals {dict(below['actual'].value_counts())}" if not below.empty else "")
          + f"   |   gate needs {GATES['min_cells']}")
    for zone in ("cut", "all"):
        g = res[res["zone"] == zone]
        if g.empty:
            print(f"\nSTEP 3b - PSI vs DEGREE, zone '{zone}': nothing fittable - fewer than {GATES['min_cells']} decided psi/degree cells there")
            continue
        n = int(g["n_cells"].iloc[0])
        print(f"\nSTEP 3b - PSI vs DEGREE, zone '{zone}' ({n} cells; chance of a clean split per property "
              f"{g['separation_by_chance'].iloc[0]}, so ~{g['expected_free_winners'].iloc[0]} of {len(g)} separate for free)")
        print(g[["predictor", "tier", "n_psi", "n_degree", "spearman_rho", "rho_vs_avg_degree", "threshold",
                 "psi_side", "n_exceptions", "loo_accuracy", "loo_folds", "majority_baseline"]].to_string(index=False))
    if not test.empty:
        print(f"\nSTEP 4 - HELD-OUT TEST ({len(set(test['dataset']))} graphs) - fitted cuts applied unchanged")
        # The constant predictors, printed FIRST: Module 11's degree-vs-psi candidate lost to "always psi" on exactly
        # this comparison, so a cut that does not beat the constant on its own scorable cells has not been tested, it has failed.
        for contrast, g in test.groupby("contrast"):
            act = g[g["scored"]].drop_duplicates("dataset")["actual"].value_counts()
            n = int(act.sum())
            print(f"  {contrast:14s} CONSTANTS ({n} scorable cells): "
                  + ", ".join(f"always {k} {v}/{n} = {v / n:.3f}" for k, v in act.items()))
        for (contrast, target, p), g in test.groupby(["contrast", "target", "predictor"]):
            c = [v for v in g["correct"] if isinstance(v, bool)]
            print(f"  {contrast:14s} {target:7s} {p}: " + (f"{sum(c)}/{len(c)} correct" if c else "nothing scorable"))
    print(f"\n{len(out):3d} tables written to results/psi_*.csv")


# Guards the test: fitting on a held-out graph would let the test set move the rule that is meant to test it.
def _assert_fit(datasets, allow_refit):
    '''Raise when a held-out graph is passed as a fitting dataset.'''
    extra = sorted(set(datasets) & set(PSI_HELDOUT))
    assert allow_refit or not extra, (f"{extra} are held out for the psi test, so they may not be fitted on. "
                                      "Pass --allow-refit to break the split deliberately.")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _assert_fit(args.datasets, args.allow_refit)
    fit = label(cells(args.datasets, args.band, args.labels))
    scr = screen(fit, args.tier)
    res = pd.concat([residual(fit, args.tier, z) for z in ("cut", "all")], ignore_index=True)
    cas = cascade(fit)
    test = heldout(label(cells(args.heldout, args.band, args.labels)), scr, res) if args.heldout else pd.DataFrame()
    prefix = ("exploratory_" if args.allow_refit else "") + ("paired_" if args.labels == "paired" else "")
    out = []
    for name, df in [("psi_cells", fit), ("psi_rules", scr), ("psi_cascade", cas), ("psi_residual", res),
                     ("psi_heldout", test)]:
        if not df.empty:
            df.to_csv(cfg.RESULTS_DIR / f"{prefix}{name}.csv", index=False)
            out.append(name)
    report(fit, scr, cas, res, test, out)


# Defines command-line options (mirrors degree_rule.py / strategy_select.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Stage 2, the psi branch: screen for a psi rule and test the centrality -> psi -> degree cascade.")
    p.add_argument('--datasets', nargs='+', default=PSI_FIT, help='Fitting graphs. Default: the strategy panel and the Module-10 corpus half of the paired-verdict pool.')
    p.add_argument('--heldout', nargs='*', default=PSI_HELDOUT, help='Graphs the cuts are TESTED on, never fitted. Default: everything ingested after the corpus.')
    p.add_argument('--tier', default='all', choices=list(TIERS_D),
                   help="'primary' = the general properties that may become a rule; 'all' adds the exploratory tier "
                        "(default); 'degree' = only the degree-specific tier; 'both' = all of them together.")
    p.add_argument('--band', default='sem', choices=['sem', 'sigma'], help="How close counts as tied. Default sem, the band Module 6 fitted under.")
    p.add_argument('--labels', default='frozen', choices=['frozen', 'paired'],
                   help="Which band names the winning signal. 'frozen' = the independent-means band every published cell "
                        "uses (default); 'paired' = tie_break.py's per-seed comparison, which resolves marginal cells.")
    p.add_argument('--allow-refit', action='store_true', help='Deliberately fit on a held-out graph. Writes exploratory_*.csv so the panel tables are never overwritten.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
