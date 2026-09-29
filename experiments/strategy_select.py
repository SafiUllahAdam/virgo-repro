'''Strategy selection: which of the seven graph variants wins link prediction, and whether graph properties predict WHICH one.'''
# Module 2 asked one BINARY question - augment, or keep the original graph. This asks the next one: given that some
# augmentation helps, WHICH structural signal (Psi, degree, eigenvector centrality) suits this graph.
#   Step 4 records the winning variant per dataset, as a SET - two variants a fraction of a seed's noise apart are not
#          ranked by three seeds, so calling one of them "the winner" would be reading noise.
#   Step 5 joins the same seven primary graph properties Module 2 and Module 3 were read against (reused, not re-derived).
#   Step 6 screens each property as a one-vs-rest rule for a signal, under the SAME gates Module 2 used, so a strategy
#          rule cannot clear a lower bar than the augment rules did.
# Fitting stays on PANEL for the reason characterize.py does: a rule fitted on a held-out graph cannot be tested on it.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import frozen_rules as fr
from experiments.characterize import (GATES, PREDICTORS_EXPLORATORY, PREDICTORS_PRIMARY, PRIMARY, graph_properties,
                                      loo_threshold, rho, threshold)

# Two tiers, as characterize.py uses them: only PRIMARY properties may become a rule. The exploratory tier is screened and
# printed - degree spread and assortativity are the properties most plausibly tied to a DEGREE or CENTRALITY graph
# winning, so not looking would be a gap - but a hit there is a lead to test, never a promotable finding.
TIERS = {"primary": PREDICTORS_PRIMARY, "all": PREDICTORS_PRIMARY + PREDICTORS_EXPLORATORY}

# The LP half of the Module-2 discovery panel: ogbn_arxiv is node classification only, and ogbl_ddi is unlabelled AND
# costs ~14h to re-run at seven seeds, so neither can carry a strategy label. Five datasets is too few to fit a
# multi-class rule, so four Module-3 datasets were PROMOTED into the fitting panel on 2026-08-13 (user decision).
CORE_FIVE = [d for d in fr.DISCOVERY_PANEL if d not in ("ogbn_arxiv", "ogbl_ddi")]

# The four promotions, chosen on GRAPH PROPERTIES ONLY - measured before any of them was trained at seven seeds, so the
# choice cannot be selection on outcome. Each fills a hole the core five leave in the property space a rule must span:
#   squirrel_filtered  avg_degree 42.3, between the core five's 6.3 and 88.3 - the only mid-density graph available
#   amazon_ratings     homophily_adjusted 0.140, inside the core five's empty 0.093-0.361 band
#   amazon_photo       the only fragmented candidate (136 components) - fragmentation had just two points, both in the core
#   lastfm_asia        homophily_adjusted 0.856 and 18 classes, extending the top of both ranges
# DISCLOSURE: an exploratory 3-seed screen over all 15 datasets had already been run, so `actor` was known to be the one
# held-out dataset that decided a single signal. It was deliberately NOT promoted - the informative dataset stays in the
# test set. Winners under seven seeds and the sem band were not computed for any candidate before this choice.
PROMOTED = ["squirrel_filtered", "amazon_ratings", "amazon_photo", "lastfm_asia"]

# 2026-08-13, second decision: the remaining three Module-3 datasets joined the panel as well. The binding constraint was
# never measurement precision - it is that the properties do not SEPARATE the winners - so development diversity is worth
# more than a reserved test set. Consequence, stated plainly: NOTHING is held out any more, so any rule this module
# produces is FITTED, not validated, exactly like the Module-4 gate. The test is a set of genuinely new datasets, later.
LATE_ADDED = ["pubmed", "actor", "minesweeper"]

# The 2026-07-29 decision that dropped these two from forward work was REVERSED for this module only (user, 2026-08-13).
# Both are heavily fragmented, which is where the panel is thinnest - enzymes was carrying that region almost alone - and
# for a strategy screen the cost of the original exclusion (they are not part of the headline study panel) does not apply.
REINSTATED = ["citeseer_linqs", "proteins"]
PANEL = CORE_FIVE + PROMOTED + LATE_ADDED + REINSTATED

# Empty by construction now: every labelled dataset is in the panel. ogbl_ddi has no labels, so it can carry no signal label.
HELDOUT = [d for d in fr.HELDOUT if d not in PANEL and d not in ("citeseer_linqs", "proteins")]

# Each variant's ROLE SIGNAL. The union hybrids collapse onto the signal they ADD, so a label answers "which structural
# signal suits this graph", not "which of seven files won"; original is the do-not-augment label.
SIGNAL = {"original": "original", "psi": "psi", "hybrid": "psi", "degree": "degree", "hybrid_degree": "degree",
          "centrality": "centrality", "hybrid_centrality": "centrality"}
SIGNALS = ["psi", "degree", "centrality"]

# A signal rule needs at least this many datasets where that signal wins ALONE. One point is an anecdote, and a group
# built only from three-way ties would let a split be fitted on datasets that never singled the signal out at all.
MIN_SOLE = 2

# ogbl_ddi is scored under the official OGB protocol, so its metric name differs by design. The winner is decided INSIDE
# one dataset, so the scorer cancels out (the 2026-07-23 within-dataset decision) - the mixed metric is flagged in the table.
# VALIDATION, not characterize.PRIMARY's test_hits@20: picking a winner among seven variants IS model selection, and OGB's
# test split is reserved for the one locked configuration (run_ogb.select). Reading test seven times would spend that budget.
LP_METRIC = {"link prediction (AUC)": "auc", "link prediction (OGB official)": "valid_hits@20"}
LP_TASKS = list(LP_METRIC)


# Same guard as characterize.py: fitting on a held-out graph would let the test set move the rule that is meant to test it.
def _assert_panel(datasets, allow_refit):
    '''Raise unless every dataset is inside PANEL (or a deliberate re-fit was requested).'''
    extra = sorted(set(datasets) - set(PANEL))
    assert allow_refit or not extra, (f"strategy fitting must stay on {PANEL}; {extra} are outside it. "
                                      "Pass --allow-refit to re-fit deliberately.")


# STEP 4: the winner is a SET - every variant the seeds cannot separate from the leader. Two bands, because they answer
# different questions and only one of them can be improved by running more seeds:
#   "sigma" = one pooled standard deviation, the convention characterize.compare() uses. Spread of the score itself, so it
#             does NOT shrink as seeds are added - more seeds only estimate the same width more precisely.
#   "sem"   = the standard error of the DIFFERENCE of the two means, sqrt(s1^2/n1 + s2^2/n2). This is the question
#             "are these two means distinguishable", and it narrows as 1/sqrt(seeds) - the only band more seeds can resolve.
BANDS = ["sem", "sigma"]


def winners(datasets, band="sem", board=None):
    '''Per dataset: best LP variant, every variant the seeds cannot separate from it, and the role signals that band names.'''
    board = pd.read_csv(cfg.SCOREBOARD_CSV) if board is None else board   # a caller may pass a board that also carries withdrawn rows
    b = board[(board["encoder"] == "graphsage_edge") & (board["top_K_neighbors"] == 10)
              & board["graph_variant"].isin(cfg.VG_SIMS) & board["dataset"].isin(datasets)
              & board["task"].isin(LP_TASKS)]
    b = b[[LP_METRIC.get(t) == m for t, m in zip(b["task"], b["metric"])]]
    assert band in BANDS, f"band must be one of {BANDS}"
    rows = []
    for (ds, task), g in b.groupby(["dataset", "task"]):
        s = g.set_index("graph_variant")
        assert len(s) == len(cfg.VG_SIMS), f"{ds}: only {len(s)} of {len(cfg.VG_SIMS)} variants scored - run the sweep first"
        n = s["seeds"].str.count(r"\|").astype(float) + 1                      # seeds are recorded as "42|43|44"
        top = s["mean"].idxmax()
        width = (np.sqrt(s.loc[top, "std"] ** 2 / n[top] + s["std"] ** 2 / n) if band == "sem"
                 else np.hypot(s.loc[top, "std"], s["std"]) / np.sqrt(2))
        keep = sorted(s.index[(s.loc[top, "mean"] - s["mean"]) <= width])
        sig = sorted({SIGNAL[v] for v in keep})
        rows.append({"dataset": ds, "task": task, "metric": s.loc[top, "metric"],
                     "seeds": int(n[top]), "band": band,
                     "best_variant": top, "best_score": round(float(s.loc[top, "mean"]), 4),
                     "original": round(float(s.loc["original", "mean"]), 4),
                     "winner_set": "|".join(keep), "winner_signals": "|".join(sig), "n_tied": len(keep),
                     # Does any augmented graph clear the original by more than noise, and does one signal stand alone?
                     "beats_original": bool("original" not in keep),
                     "signal_decided": bool(len(sig) == 1 and sig != ["original"])})
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


# STEP 5: reuse the measured table wherever it exists - these are properties of the GRAPH, so they must be the same
# numbers Module 2 and Module 3 were read against, never a second measurement that could drift from them.
def properties(datasets):
    '''The seven primary graph properties per dataset, read from results/dataset_characterization.csv, measuring only what is missing.'''
    path = cfg.RESULTS_DIR / "dataset_characterization.csv"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["dataset"])
    miss = [d for d in datasets if d not in set(have["dataset"])]
    all_rows = pd.concat([have, pd.DataFrame([graph_properties(d) for d in miss])], ignore_index=True) if miss else have
    out = all_rows[all_rows["dataset"].isin(datasets)][["dataset"] + TIERS["all"]].copy()
    # graph_properties writes n_classes=0 for an unlabelled graph (ogbl_ddi). Ranked as a number it is the smallest value
    # in the panel, so a "low n_classes" split would really be fitting "has no labels" - the same hole the label properties
    # already report as missing. Mark it missing here too, so no rule can be built on the absence of labels.
    out.loc[out["n_classes"] == 0, "n_classes"] = np.nan
    return out.sort_values("dataset").reset_index(drop=True)


# loo_threshold SKIPS any fold it cannot fit - and holding out the only dataset of its class produces exactly such a fold.
# Its accuracy can therefore be scored entirely on majority-class folds, i.e. on the easy cases, while the majority baseline
# is computed over every point. That mismatch credits a rule for never having been tested on the case that decides it, so
# the baseline is recomputed over EXACTLY the folds loo_threshold scored.
def loo_majority(x, y):
    '''Accuracy of always guessing the training majority, over the same folds loo_threshold scores: (accuracy, folds).'''
    ok, n = 0, 0
    for i in range(len(x)):
        m = np.ones(len(x), dtype=bool)
        m[i] = False
        if len(np.unique(x[m])) < 2 or len(np.unique(y[m])) < 2:
            continue
        ok += int((y[m].mean() > 0.5) == y[i])
        n += 1
    return (round(ok / n, 4), n) if n else (float("nan"), 0)


# STEP 6: one-vs-rest screen - does any property separate the graphs whose winning band contains a signal from the rest?
# Reported per (signal, property) with the Module-2 gates; a property that clears them is a CANDIDATE for Module 6, never a rule.
def patterns(win, prop, allow_refit=False, scope="all", tier="primary"):
    '''One row per signal x property: group ranges, best split, exceptions, out-of-sample LOO accuracy and the credible flag.'''
    _assert_panel(win["dataset"], allow_refit)
    d = win.merge(prop, on="dataset")
    # scope="augmented" asks the CONDITIONAL question - given that augmentation helps, which signal - by dropping the
    # keep-original datasets. Without it, "not centrality" silently includes "do not augment at all", which is Module 2's
    # question, not this one. The cost is a smaller n, so both scopes are reported rather than one replacing the other.
    if scope == "augmented":
        d = d[d["beats_original"]].reset_index(drop=True)
    rows = []
    for sig in SIGNALS + ["original"]:
        member = np.array([sig in w.split("|") for w in d["winner_signals"]])
        # A dataset whose band names three signals says nothing about WHICH one is right - it is a member of all three
        # groups at once. Count the datasets that name this signal ALONE: those are the only ones that carry evidence.
        sole = np.array([w == sig for w in d["winner_signals"]])
        for p in TIERS[tier]:
            ok = d[p].notna().to_numpy()                       # ogbl_ddi has no labels -> no homophily, no class count
            x, y = d.loc[ok, p].to_numpy(dtype=float), member[ok]
            fit = len(x) >= GATES["min_cells"] and y.any() and not y.all() and len(np.unique(x)) > 1
            t, side, err, acc, lo, hi = threshold(x, y) if fit else (np.nan, "", -1, np.nan, np.nan, np.nan)
            l_acc, l_ok, l_n, _, _ = loo_threshold(x, y) if fit else (np.nan, 0, 0, np.nan, np.nan)
            l_maj, _ = loo_majority(x, y) if fit else (np.nan, 0)
            major = round(float(max(y.mean(), 1 - y.mean())), 4) if len(y) else np.nan
            r = rho(pd.Series(x), pd.Series(y.astype(float)))[1]
            rows.append({
                "scope": scope, "tier": "primary" if p in PREDICTORS_PRIMARY else "exploratory",
                "signal": sig, "predictor": p, "n_scored": int(len(x)), "n_missing": int((~ok).sum()),
                "n_wins": int(y.sum()), "n_sole_wins": int(sole[ok].sum()), "wins": "|".join(sorted(d.loc[ok, "dataset"][y])),
                # The two groups' value ranges: a rule is only readable if the ranges do not interleave.
                "win_range": f"{x[y].min():.4f}-{x[y].max():.4f}" if y.any() else "",
                "rest_range": f"{x[~y].min():.4f}-{x[~y].max():.4f}" if (~y).any() else "",
                "spearman_rho": r, "threshold": t, "win_side": side, "n_exceptions": err, "accuracy": acc,
                # The panel pins an INTERVAL, never a 4-decimal point: every cut between these two values fits it equally well.
                "interval_lo": lo, "interval_hi": hi,
                "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n, "majority_baseline": major,
                # The baseline on the folds actually scored. l_n < n_scored means the deciding fold was skipped entirely.
                "loo_majority": l_maj, "folds_skipped": int(len(x) - l_n) if fit else -1,
                "rule": f"use {sig} when {p} is {side} ({'>' if side == 'high' else '<'} {t})" if side else "",
                # A group whose members are all ties is not evidence about this signal, however cleanly it splits.
                "tie_dominated": bool(int(sole[ok].sum()) < MIN_SOLE),
                # Module-2 gates, plus MIN_SOLE: in-sample exceptions cannot fail on separable data, so LOO must beat majority too.
                "credible": bool(p in PREDICTORS_PRIMARY and r == r and abs(r) >= GATES["min_abs_rho"] and 0 <= err <= GATES["max_exceptions"]
                                 and len(x) >= GATES["min_cells"] and int(sole[ok].sum()) >= MIN_SOLE
                                 and (l_acc == l_acc and l_acc > major and l_acc > l_maj
                                      if GATES["loo_above_majority"] else True)),
            })
    return pd.DataFrame(rows).sort_values(["signal", "predictor"]).reset_index(drop=True)


# STEP 7: the zone FROZEN_STRATEGY leaves undetermined - psi versus degree, below the centrality cut. n is tiny there by
# construction, and with one psi against two degree datasets EVERY property whose values differ separates them perfectly,
# so "0 exceptions" carries no information at all. The MARGIN is reported instead: the gap between the psi dataset and its
# nearest degree neighbour, as a fraction of that property's spread over the whole panel. Wide margin = a lead worth
# testing on a new dataset; interleaved = dead. Nothing here can become a rule - the gates below say why for each row.
def contrast(win, prop, tier="all", cut=None):
    '''Psi vs degree inside the undetermined zone: per property, each group's values, the separation margin, and the blocking gate.'''
    cut = fr.FROZEN_STRATEGY.point if cut is None else cut
    d = win.merge(prop, on="dataset")
    zone = d[d["beats_original"] & (d[fr.FROZEN_STRATEGY.predictor] <= cut)].reset_index(drop=True)
    rows = []
    for p in TIERS[tier]:
        z = zone[zone[p].notna()]
        psi = z[[s == "psi" for s in z["winner_signals"]]]
        deg = z[[s == "degree" for s in z["winner_signals"]]]
        span = float(d[p].max() - d[p].min())                       # spread over the FULL panel, so margins are comparable
        gap = (min(abs(float(a) - float(b)) for a in psi[p] for b in deg[p]) if len(psi) and len(deg) else float("nan"))
        lo, hi = (float(psi[p].min()), float(psi[p].max())) if len(psi) else (float("nan"),) * 2
        rows.append({
            "predictor": p, "tier": "primary" if p in PREDICTORS_PRIMARY else "exploratory",
            "n_zone": len(z), "n_psi": len(psi), "n_degree": len(deg),
            "psi_values": "|".join(f"{v:.4f}" for v in psi[p]), "degree_values": "|".join(f"{v:.4f}" for v in deg[p]),
            # Separable = psi sits entirely outside the degree range. Guaranteed here whenever the values differ at all.
            "separable": bool(len(psi) and len(deg) and (lo > deg[p].max() or hi < deg[p].min())),
            "margin": round(gap, 4) if gap == gap else float("nan"),
            "margin_frac_of_panel": round(gap / span, 4) if gap == gap and span else float("nan"),
            # Why this can never be promoted, named explicitly rather than left as a silent False.
            "blocked_by": ("n_zone < min_cells" if len(z) < GATES["min_cells"] else
                           "psi has < MIN_SOLE sole wins" if len(psi) < MIN_SOLE else
                           "degree has < MIN_SOLE sole wins" if len(deg) < MIN_SOLE else ""),
        })
    return (pd.DataFrame(rows).sort_values("margin_frac_of_panel", ascending=False)
            .reset_index(drop=True))


# --- the two thesis-facing views -------------------------------------------------------------------------------------
# What the rule was derived FROM: per dataset, the original graph against the best AUGMENTED variant, and which signal won.
# Reads the same scoreboard rows winners() does, so it cannot disagree with the analysis below it.
def panel_table(datasets, band="sem"):
    '''One row per panel dataset: original score, best augmented variant and score, and the winning signal.'''
    win = winners(datasets, band).set_index("dataset")
    board = pd.read_csv(cfg.SCOREBOARD_CSV)
    board = board[(board["encoder"] == "graphsage_edge") & (board["top_K_neighbors"] == 10)
                  & board["graph_variant"].isin(cfg.VG_SIMS) & board["dataset"].isin(datasets)
                  & board["task"].isin(LP_TASKS)]
    rows = []
    for ds in win.index:
        aug = board[(board["dataset"] == ds) & (board["graph_variant"] != "original")]
        best = aug.loc[aug["mean"].idxmax()]
        w = win.loc[ds]
        rows.append({"Dataset": ds, "Original score": round(float(w["original"]), 4),
                     "Best augmented variant": best["graph_variant"], "Best score": round(float(best["mean"]), 4),
                     # The signal of the OVERALL winner: "original" whenever the unchanged graph is still on top.
                     "Winning signal": SIGNAL[w["best_variant"]]})
    return pd.DataFrame(rows).sort_values("Dataset").reset_index(drop=True)


# The one rule that survived the screen, with its evidence - the rejected candidates stay in strategy_patterns.csv.
def rule_table():
    '''The accepted strategy rule as one row: condition, prediction, what it was fitted on, and the evidence behind it.'''
    s = fr.FROZEN_STRATEGY
    e = pd.read_csv(cfg.RESULTS_DIR / "strategy_patterns.csv")
    e = e[(e["scope"] == "augmented") & (e["signal"] == s.signal) & (e["predictor"] == s.predictor)].iloc[0]
    return pd.DataFrame([{
        "Rule": s.name,
        "Condition": f"{s.predictor} {s.op} {s.point}  (interval {s.interval[0]}-{s.interval[1]})",
        "Prediction": f"{s.signal}, else psi or degree (undetermined) - only when {s.applies_when}",
        "Fitted datasets": f"{int(e['n_scored'])} augmenting, of {len(STRATEGY_PANEL := fr.STRATEGY_PANEL)} in the panel",
        "Evidence": (f"{int(e['n_exceptions'])} exceptions, rho {e['spearman_rho']}, "
                     f"LOO {int(e['loo_correct'])}/{int(e['loo_folds'])} vs {e['majority_baseline']} majority; sem band only"),
    }])


# The reading of the two tables: how much the panel can decide at all, then anything that clears the gates.
def report(win, pat):
    '''Print the winners, how many datasets name a single signal, and every credible signal rule.'''
    print(f"\nSTEP 4 - WINNING STRATEGY PER DATASET (band = {win['band'].iloc[0]}; metric differs on ogbl_ddi by protocol)")
    print(win[["dataset", "metric", "seeds", "original", "best_variant", "best_score", "winner_set", "winner_signals",
               "n_tied", "beats_original", "signal_decided"]].to_string(index=False))
    dec = win[win["signal_decided"]]
    print(f"\nCEILING  {len(win)} datasets  |  {int(win['beats_original'].sum())} where augmentation beats the original graph  |  "
          f"{len(dec)} that name ONE signal ({', '.join(dec['dataset'] + '=' + dec['winner_signals']) or 'none'})")
    print("A property can only be asked to predict a signal on the datasets that name one; the rest are ties, not evidence.")
    print("\nSTEP 6 - PROPERTY -> SIGNAL SCREEN (one-vs-rest, Module-2 gates + at least "
          f"{MIN_SOLE} datasets where the signal wins alone)")
    print(pat[["scope", "tier", "signal", "predictor", "n_scored", "n_wins", "n_sole_wins", "win_range", "rest_range",
               "spearman_rho", "threshold", "win_side", "n_exceptions", "loo_accuracy", "loo_folds", "loo_majority",
               "majority_baseline", "tie_dominated", "credible"]].to_string(index=False))
    good = pat[pat["credible"]]
    print("\nCREDIBLE RULES: " + ("none - no property separates the winning signals on this panel" if good.empty else ""))
    for _, r in good.iterrows():
        print(f"  {r['rule']}   [interval {r['interval_lo']}-{r['interval_hi']}, rho {r['spearman_rho']}, "
              f"LOO {r['loo_correct']}/{r['loo_folds']} vs {r['majority_baseline']} majority]")
    # An "original" rule is the Module-2 question restated (augment or not), so it is not new evidence - say so, do not re-freeze it.
    if not good.empty and (good["signal"] == "original").any():
        print("  NOTE: signal='original' rules restate Module 2 (augment vs keep) on the same panel - already frozen, not a new finding.")
    print(f"\nHeld out, never fitted on: {', '.join(HELDOUT)}")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    win, prop = winners(args.datasets, args.band), properties(args.datasets)
    # Both scopes: "all" answers "which signal, over every panel graph", "augmented" the conditional question this module set out to ask.
    pat = pd.concat([patterns(win, prop, args.allow_refit, s, args.tier) for s in ("all", "augmented")], ignore_index=True)
    prefix = "exploratory_" if args.allow_refit else ""
    for name, df in [("strategy_winners", win.merge(prop, on="dataset")), ("strategy_patterns", pat)]:
        df.to_csv(cfg.RESULTS_DIR / f"{prefix}{name}.csv", index=False)
        print(f"{len(df):3d} rows -> results/{prefix}{name}.csv", flush=True)
    report(win, pat)


# Defines command-line options (mirrors characterize.py / gate_rules.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Which graph variant wins link prediction, and can graph properties predict which.")
    p.add_argument('--datasets', nargs='+', default=PANEL,
                   help='Datasets to screen. Default: the link-prediction half of the discovery panel.')
    p.add_argument('--tier', default='all', choices=list(TIERS),
                   help="Which properties to screen. 'primary' = the seven that may become a rule; 'all' also screens the "
                        "exploratory tier, printed as leads and never promotable (default).")
    p.add_argument('--band', default='sem', choices=BANDS,
                   help="How close counts as tied. 'sem' = standard error of the difference of means, the only band more seeds "
                        "can narrow (default). 'sigma' = one pooled std, the Module-2 convention, which seeds do not shrink.")
    p.add_argument('--allow-refit', action='store_true',
                   help='Deliberately fit outside PANEL. OFF by default so held-out graphs cannot move the rule that tests them. '
                        'Writes exploratory_*.csv so the panel tables are never overwritten.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
