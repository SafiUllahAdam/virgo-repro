'''Stage-1 stabilization - screen ORIGINAL-graph properties as the second condition that gates rule 1 (adjusted homophily).

Same question Module 4 asked, with the constraint that made its answer unusable removed. The gate it found,
`original_retention`, is a property of the BUILT role graph: it needs the rewiring constructed before the decision can be
taken, and on the Module-5 held-out set its only differentiating call was wrong. This screen therefore admits only
properties measurable on the original graph, so the resulting rule reads

    adjusted homophily low  AND  X on its augment side  =>  augment,   otherwise keep the original.

Rule 1's reliable side is left untouched: high adjusted homophily keeps calling "keep original", which Module 3 got right
every time. A candidate is only interesting if it repairs the calls rule 1 gets WRONG inside the low-homophily zone
(minesweeper, amazon_ratings) without breaking the ones it already gets right, which is what compound() measures.

It FITS, so it declares its own panel and is guarded. Read-only w.r.t. everything frozen: it reads the scoreboard and the
edgelists, and never writes candidate_rules.csv, gate_candidates.csv or any module*.csv.'''

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import frozen_rules as fr
from experiments.characterize import (GATES, PREDICTORS_EXPLORATORY, PREDICTORS_PRIMARY, STUDY, TARGETS, frozen_inputs,
                                      gaps, graph_properties, loo_threshold, rho, threshold)
from experiments.gate_rules import decisive_pair

# DELIBERATELY the Module-4 panel: this screen exists to replace the gate that was fitted there, so it must be fitted on
# the same evidence or the two cannot be compared. Consequence carried forward unchanged: the panel INCLUDES the Module-3
# datasets, so anything found here is FITTED, not validated, and needs a pre-registered test on a genuinely unseen graph.
PAIR_PANEL = fr.GATE_PANEL

assert set(PAIR_PANEL) <= set(STUDY), f"pair panel names an unregistered dataset: {sorted(set(PAIR_PANEL) - set(STUDY))}"

# The candidates: every property Module 2 measures on the ORIGINAL graph, minus rule 1 itself (it is condition one, not
# condition two) and minus the two label properties that are undefined without labels... which are kept, because inside the
# zone every graph is labelled by construction - rule 1 could not have fired otherwise. Tiering is Module 2's: only primary
# candidates may be promoted, and only they carry the multiple-comparison correction.
CANDIDATES_PRIMARY = [p for p in PREDICTORS_PRIMARY if p != fr.LEAD.predictor]
CANDIDATES_EXPLORATORY = [p for p in PREDICTORS_EXPLORATORY if p != fr.LEAD.predictor]
CANDIDATES = CANDIDATES_PRIMARY + CANDIDATES_EXPLORATORY

COLLINEAR = 0.99           # |rho| at or above which two candidates are reported as one finding, not two
CANONICAL_SIM = "psi"      # the variant the gate was measured on, when the incumbent is quoted for comparison


# The graph description the user asked for: one row per dataset = its properties, its LP verdict, and whether rule 1 fires.
def panel_cells(datasets):
    '''Per dataset: every original-graph property, the frozen LP gap and verdict, and whether the cell is inside the zone.'''
    g = gaps(frozen_inputs(datasets))
    g = g[(g["task_family"] == "link prediction") & g["usable"]].copy()
    props = pd.DataFrame([graph_properties(ds) for ds in g["dataset"]])
    g = g.merge(props.drop(columns=["domain", "tasks", "graph_scope", "directed_source"]), on="dataset")
    # The zone is where rule 1 says "augment": the only place a second condition may speak. No labels -> no homophily ->
    # rule 1 cannot fire, and the cell sits outside the zone by absence, which is the coverage hole rule 2 exists for.
    g["rule1_pred"] = [fr.predict_one(fr.LEAD, v) for v in g["homophily_adjusted"]]
    g["in_zone"] = g["rule1_pred"] == "augment"
    # The incumbent, carried along only so compound() can score it with the same code: original_retention is a property of
    # the BUILT role graph, measured by Module 4 and read off its table here. Never screened as a candidate - it is the
    # thing being replaced - and absent without that table, in which case the comparison row is simply not emitted.
    vg = cfg.RESULTS_DIR / "vg_characterization.csv"
    if vg.exists():
        r = pd.read_csv(vg)
        r = r[r["sim"] == "psi"][["dataset", fr.FROZEN_GATE.predictor]]
        g = g.merge(r, on="dataset", how="left")
    return g


# How often a threshold separates a zone this size PURELY by luck: with n decided cells of which a are augment, only 2 of
# the C(n, a) labellings are separable by any single cut. Quoted next to every "0 exceptions", because on 7 cells with 2
# augment the chance is 2/21 per candidate and 15 candidates are tried - the expected number of free winners is above one.
def chance(n, a, n_tried):
    '''(probability one candidate separates by chance, expected number of candidates that do) for a zone of n cells, a augmenting.'''
    if not 0 < a < n:
        return float("nan"), float("nan")
    p = 2 / math.comb(n, a)
    return round(p, 4), round(p * n_tried, 2)


# Screen each candidate inside the zone, exactly as Module 2 and 4 screen theirs: split, interval, out-of-sample LOO,
# majority baseline, and whether the split calls the decisive pair - the cell a second condition was proposed for - right.
def screen(cells, targets=TARGETS):
    '''One row per target x candidate: rho, split, interval, leave-one-out accuracy, decisive-pair separation, credibility.'''
    rows = []
    for target, vcol in targets.items():
        s = cells[cells["in_zone"] & (cells[vcol] != "tie")].reset_index(drop=True)
        pair = decisive_pair(s, vcol)
        for prop in CANDIDATES:
            d = s[s[prop].notna()].reset_index(drop=True)
            n, r, pv = rho(d[prop], d[target])
            lodo = [v for v in (rho(d[prop].drop(i), d[target].drop(i))[1] for i in d.index) if v == v]
            stable = bool(lodo) and r == r and all(np.sign(v) == np.sign(r) for v in lodo)
            y = (d[vcol] == "augment").to_numpy()
            fit = len(d) >= GATES["min_cells"] and y.any() and not y.all()
            t, side, err, acc, lo, hi = (threshold(d[prop].to_numpy(), y) if fit
                                         else (float("nan"), "", -1, float("nan"), float("nan"), float("nan")))
            l_acc, l_ok, l_n, l_lo, l_hi = (loo_threshold(d[prop].to_numpy(), y) if fit
                                            else (float("nan"), 0, 0, float("nan"), float("nan")))
            major = round(float(max(y.mean(), 1 - y.mean())), 4) if len(y) else float("nan")
            sep = float("nan")
            if pair and side:
                mem = d[d["dataset"].isin(pair[1:])]
                if len(mem) == 2:
                    sep = bool(all(((float(v) > t) if side == "high" else (float(v) < t)) == (w == "augment")
                                   for v, w in zip(mem[prop], mem[vcol])))
            p_chance, exp_hits = chance(len(d), int(y.sum()), len(CANDIDATES_PRIMARY))
            rows.append({
                "target": target, "property": prop,
                "tier": "primary" if prop in CANDIDATES_PRIMARY else "exploratory",
                "n_zone": len(d), "n_augment": int(y.sum()), "distinct_values": int(d[prop].nunique()),
                "distinct_augment": int(d[prop][y].nunique()),
                "spearman_rho": r, "p_value": pv,
                "p_bonferroni": round(min(1.0, pv * len(CANDIDATES_PRIMARY)), 4)
                                if prop in CANDIDATES_PRIMARY and pv == pv else float("nan"),
                "lodo_min_abs_rho": round(min(abs(v) for v in lodo), 4) if lodo else float("nan"),
                "lodo_sign_stable": stable,
                "threshold": t, "augment_side": side, "n_exceptions": err, "accuracy": acc,
                # The zone pins an INTERVAL, not a point: every cut between these two values fits it equally well.
                "interval_lo": lo, "interval_hi": hi,
                "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
                "loo_threshold_lo": l_lo, "loo_threshold_hi": l_hi,
                "majority_baseline": major, "loo_beats_majority": bool(l_acc == l_acc and l_acc > major),
                # In-sample separation is nearly free on a zone this small - these two say how free.
                "separation_by_chance": p_chance, "expected_free_winners": exp_hits,
                "decisive_pair": " vs ".join(pair[1:]) if pair else "", "separates_decisive_pair": sep,
                "rule": (f"augment when {fr.LEAD.predictor} < {fr.LEAD.point} AND {prop} is {side} "
                         f"({'>' if side == 'high' else '<'} {t})") if side else "",
                "credible": bool(r == r and abs(r) >= GATES["min_abs_rho"] and stable
                                 and 0 <= err <= GATES["max_exceptions"] and n >= GATES["min_cells"]
                                 and (l_acc == l_acc and l_acc > major if GATES["loo_above_majority"] else True)),
            })
    return pd.DataFrame(rows)


# The question the zone screen cannot answer: does the compound rule beat rule 1 over the WHOLE panel? A condition that
# repairs the zone by rejecting everything in it would score well above and be useless - so both sides are counted here,
# and the datasets each candidate fixes and breaks are named rather than summarized.
def evaluate(d, vcol, base, conds, name, tier, fit_loo=True):
    '''Score "rule 1 AND every condition in conds" over the decided cells: correct count, delta on rule 1, cells fixed and broken.'''
    hit = [all(bool(v == v and ((v > c) if side == "high" else (v < c))) for v, (_, c, side) in zip(row, conds))
           for row in zip(*[d[p] for p, _, _ in conds])]
    pred = ["augment" if (z and h) else "keep original" for z, h in zip(d["in_zone"], hit)]
    ok = [p == a for p, a in zip(pred, d[vcol])]
    idx = lambda f: ", ".join(d.loc[[i for i in range(len(d)) if f(i)], "dataset"])
    l_acc, l_ok, l_n = loo_panel(d, vcol, [p for p, _, _ in conds]) if fit_loo else (float("nan"), 0, 0)
    return {"property": name, "tier": tier,
            # Out-of-sample: the condition's cut refitted with each dataset hidden. pair_correct above is in-sample.
            "loo_panel_accuracy": l_acc, "loo_panel_correct": l_ok, "loo_panel_folds": l_n,
            "conditions": " AND ".join(f"{p} {'>' if side == 'high' else '<'} {c}" for p, c, side in conds),
            "n_decided": len(d), "rule1_correct": int(sum(base)), "pair_correct": int(sum(ok)),
            "delta": int(sum(ok) - sum(base)),
            # A cell rule 1 missed and the compound rule now calls right, and the reverse - the whole point of the screen.
            "fixes": idx(lambda i: ok[i] and not base[i]), "breaks": idx(lambda i: base[i] and not ok[i]),
            "still_wrong": idx(lambda i: not ok[i]), "n_augment_calls": int(sum(p == "augment" for p in pred)),
            # A condition whose entire effect is one dataset rests on that dataset alone, whatever its accuracy says.
            "cells_moved": int(sum(p != r for p, r in zip(pred, d["rule1_pred"]) if r != "n/a"))}


# evaluate() fits the cut on the same cells it scores, so its count cannot fall below rule 1 by construction. This hides
# one dataset, REFITS every named condition on the remaining zone, and predicts the hidden one - the only out-of-sample
# number for a compound rule. Rule 1 itself is never refitted: its cut is frozen, and only the second condition is fitted.
def loo_panel(d, vcol, props):
    '''Leave-one-dataset-out accuracy of "rule 1 AND the refitted conditions": (accuracy, correct, folds scored).'''
    ok, folds = 0, 0
    for i in range(len(d)):
        tr = d.drop(i)
        z = tr[tr["in_zone"]]
        y = (z[vcol] == "augment").to_numpy()
        if len(z) < GATES["min_cells"] or not y.any() or y.all():
            continue
        conds = []
        for prop in props:
            t, side, _, _, _, _ = threshold(z[prop].to_numpy(), y)
            conds.append((prop, t, side))
        v = [d.loc[i, prop] for prop, _, _ in conds]
        hit = all(bool(x == x and ((x > c) if side == "high" else (x < c))) for x, (_, c, side) in zip(v, conds))
        pred = "augment" if (bool(d.loc[i, "in_zone"]) and hit) else "keep original"
        ok, folds = ok + int(pred == d.loc[i, vcol]), folds + 1
    return (round(ok / folds, 4) if folds else float("nan")), ok, folds


def compound(cells, scr, target="gap_rel"):
    '''Per candidate: rule 1 alone vs "rule 1 AND X" over every decided cell, with the frozen gate scored the same way for comparison.'''
    vcol = TARGETS[target]
    d = cells[cells[vcol] != "tie"].reset_index(drop=True)
    base = [p == a for p, a in zip(d["rule1_pred"], d[vcol])]
    rows = [evaluate(d, vcol, base, [(r.property, r.threshold, r.augment_side)], r.property, r.tier)
            for r in scr[scr["target"] == target].itertuples() if r.augment_side]
    # The incumbent at its FROZEN cut, not refitted: the number any replacement has to beat, on the same cells.
    if fr.FROZEN_GATE.predictor in d:
        rows.append(evaluate(d, vcol, base, [(fr.FROZEN_GATE.predictor, fr.FROZEN_GATE.point, "low")],
                             fr.FROZEN_GATE.predictor, "frozen gate (needs the role graph BUILT)", fit_loo=False))
    return pd.DataFrame(rows).sort_values(["delta", "pair_correct"], ascending=False).reset_index(drop=True)


# No single property repaired both of rule 1's failures, and they have different causes, so the obvious next question is
# whether TWO conditions do. Reported separately and never promoted: with this many combinations over ten decided cells
# the best pair is found by search, so it is a hypothesis to pre-register, not a result.
def pairs(cells, scr, target="gap_rel"):
    '''Every two-condition compound rule at the zone-fitted cuts, scored over the decided cells like the single ones.'''
    vcol = TARGETS[target]
    d = cells[cells[vcol] != "tie"].reset_index(drop=True)
    base = [p == a for p, a in zip(d["rule1_pred"], d[vcol])]
    fitted = [(r.property, r.threshold, r.augment_side, r.tier) for r in scr[scr["target"] == target].itertuples() if r.augment_side]
    rows = []
    for i, a in enumerate(fitted):
        for b in fitted[i + 1:]:
            tier = "primary" if a[3] == b[3] == "primary" else "mixed/exploratory"
            rows.append(evaluate(d, vcol, base, [a[:3], b[:3]], f"{a[0]} + {b[0]}", tier))
    out = pd.DataFrame(rows)
    out["n_combinations"] = len(rows)            # the search size, carried on every row so the number cannot be quoted without it
    return out.sort_values(["delta", "n_augment_calls"], ascending=[False, False]).reset_index(drop=True)


# Which candidates are the SAME measurement: three degree-spread properties splitting the zone identically is ONE finding.
def collinearity(cells):
    '''Spearman between every pair of candidates inside the zone, flagging the pairs that carry identical information.'''
    z = cells[cells["in_zone"]]
    rows = []
    for i, a in enumerate(CANDIDATES):
        for b in CANDIDATES[i + 1:]:
            n, r, _ = rho(z[a], z[b])
            rows.append({"property_a": a, "property_b": b, "n": n, "spearman_rho": r,
                         "same_information": bool(r == r and abs(r) >= COLLINEAR)})
    return pd.DataFrame(rows)


# The reading: what the second condition would be, what it costs, and every reason it is still only a candidate.
def report(cells, scr, comp, pr, coll):
    '''Print the stage-1 pair conclusion: the zone, the graph descriptions, the candidates, and the panel-level effect.'''
    print(f"\nLINK-PREDICTION CELLS (frozen scoreboard; the zone is where rule 1 fires, i.e. {fr.LEAD.predictor} < {fr.LEAD.point})")
    print(cells[["dataset", "homophily_adjusted", "in_zone", "rule1_pred", "original", "best_augmented", "best_variant",
                 "gap_sigma", "verdict", "verdict_fixed"]].to_string(index=False))
    z = cells[cells["in_zone"]]
    print(f"\n  zone holds {len(z)} cells: " + " | ".join(f"{v}: {n}" for v, n in z["verdict"].value_counts().items()))
    print("\nGRAPH DESCRIPTIONS INSIDE THE ZONE (the candidate second conditions, one row per graph)")
    print(z[["dataset", "verdict"] + CANDIDATES_PRIMARY].to_string(index=False))
    print(f"\nCANDIDATE SECOND CONDITIONS (gates: |rho|>={GATES['min_abs_rho']}, LODO sign-stable, "
          f"<={GATES['max_exceptions']} exception, leave-one-out above the majority baseline)")
    print(scr[["target", "property", "tier", "n_zone", "n_augment", "distinct_values", "spearman_rho", "p_bonferroni",
               "threshold", "augment_side", "n_exceptions", "loo_accuracy", "majority_baseline",
               "separates_decisive_pair", "credible"]].to_string(index=False))
    print("\n  collinear candidates inside the zone (|rho| >= %.2f -> ONE finding, not several): " % COLLINEAR +
          (" | ".join(f"{r.property_a} ~ {r.property_b} ({r.spearman_rho:+.2f})"
                      for r in coll[coll["same_information"]].itertuples()) or "none"))
    keep = scr[scr["credible"] & (scr["tier"] == "primary")]
    print("\n  CREDIBLE CANDIDATES: " + (", ".join(dict.fromkeys(keep["property"])) if len(keep) else
                                         "none - no original-graph property clears the gates inside the zone"))
    for r in keep.drop_duplicates("rule").itertuples():
        print(f"    [{r.target}] {r.rule}"
              f"\n      any cut in ({r.interval_lo}, {r.interval_hi}) fits the zone equally well -> quote the interval"
              f"\n      leave-one-out {r.loo_correct}/{r.loo_folds} = {r.loo_accuracy} vs {r.majority_baseline} for always"
              f" guessing the majority verdict; fold cutoffs ranged {r.loo_threshold_lo} to {r.loo_threshold_hi}"
              f"\n      separates the decisive pair ({r.decisive_pair}): {r.separates_decisive_pair}")
    row = scr[scr["target"] == "gap_rel"].head(1)
    if len(row):
        print(f"\n  HOW FREE A CLEAN SPLIT IS HERE - a zone of {int(row['n_zone'].iloc[0])} cells with "
              f"{int(row['n_augment'].iloc[0])} augment is separated by chance with probability "
              f"{row['separation_by_chance'].iloc[0]} per candidate; over the {len(CANDIDATES_PRIMARY)} primary candidates "
              f"tried, {row['expected_free_winners'].iloc[0]} are expected to separate perfectly for no reason at all.")
    print("\nPANEL-LEVEL EFFECT - rule 1 alone vs 'rule 1 AND X' over every decided cell (the zone screen cannot see this)")
    print(comp[["property", "tier", "conditions", "n_decided", "rule1_correct", "pair_correct", "delta",
                "loo_panel_accuracy", "fixes", "breaks", "cells_moved", "n_augment_calls"]].to_string(index=False))
    blind = cells[(cells[TARGETS["gap_rel"]] != "tie") & (cells["rule1_pred"] == "n/a")]["dataset"].tolist()
    if blind:
        print("\n  NO LABELS, so rule 1 cannot fire and the zone never opens: " + ", ".join(blind) +
              " - counted as a miss for EVERY row below, exactly as Module 2 scores an unevaluable rule; rule 2 is what"
              " covers these graphs, and no second condition changes that.")
    n, c1 = int(comp["n_decided"].iloc[0]), int(comp["rule1_correct"].iloc[0])
    print(f"\n  THE BAR: rule 1 alone is {c1}/{n} = {c1 / n:.2f}, and because its cut is FROZEN that is also its"
          " out-of-sample number - a second condition has to beat it in the loo_panel_accuracy column, not in pair_correct,"
          " which is fitted on the same cells it is scored on.")
    one = comp[(comp["cells_moved"] == 1) & (comp["delta"] > 0)]["property"].tolist()
    if one:
        print("  EVIDENCE FROM ONE GRAPH - these move exactly one cell, so their whole case rests on that graph: "
              + ", ".join(one))
    inc = comp[comp["tier"].str.startswith("frozen gate")]
    if len(inc):
        r = inc.iloc[0]
        print(f"\n  THE INCUMBENT, scored on the same cells at its frozen cut: {r['property']} {r['pair_correct']}/{r['n_decided']}"
              f" (delta {r['delta']:+d} on rule 1), fixes [{r['fixes'] or 'nothing'}], breaks [{r['breaks'] or 'nothing'}]."
              " Any original-graph replacement has to match that WITHOUT needing the role graph built first.")
    dead = comp[comp["n_augment_calls"] == 0]["property"].tolist()
    if dead:
        print("  NEVER AUGMENTS - the condition closes the zone entirely, so the 'rule' is just 'keep original always': "
              + ", ".join(dead))
    print("\nTWO CONDITIONS - exploratory only: the best pair below was FOUND BY SEARCH over "
          f"{int(pr['n_combinations'].iloc[0]) if len(pr) else 0} combinations on {int(pr['n_decided'].iloc[0]) if len(pr) else 0}"
          " decided cells, so it is a hypothesis to pre-register, never a fitted result to report.")
    print(pr[pr["delta"] > 0][["property", "tier", "conditions", "rule1_correct", "pair_correct", "delta",
                               "loo_panel_accuracy", "fixes", "breaks", "n_augment_calls"]].head(8).to_string(index=False))
    print("\n  STATUS: fitted on PAIR_PANEL, which INCLUDES the Module-3 datasets - nothing above is validated. A candidate")
    print("  that survives here is a prediction to be pre-registered on a genuinely unseen graph, exactly as Module 3 was,")
    print("  and only then may it be written into virgo/frozen_rules.py.")


# The retrospective check. Applies the PANEL-FITTED cuts, unchanged, to graphs the screen never saw. It is not a
# pre-registration: these graphs were trained and scored months ago, so their outcomes already existed. It is the
# cheapest available evidence that a candidate is not pure panel artefact - and the honest label for it is "retrospective".
def heldout(datasets, scr, comp, target="gap_rel"):
    '''Score rule 1 and each candidate second condition, at the cuts fitted on the panel, against the verdicts of unseen graphs.'''
    vcol = TARGETS[target]
    board = frozen_inputs(datasets)
    # Two of the test graphs were withdrawn from the strategy batch on 2026-08-14, so their ten-seed link-prediction
    # scores live in results/module7_withdrawn.csv rather than the scoreboard. Same runs, same protocol, same five locked
    # variants - read them back rather than retrain, and never write them into the scoreboard.
    miss = [d for d in datasets if d not in set(board["dataset"])]
    wd = cfg.RESULTS_DIR / "module7_withdrawn.csv"
    if miss and wd.exists():
        w = pd.read_csv(wd)
        w = w[w["dataset"].isin(miss) & w["graph_variant"].isin(cfg.VG_SIMS_LOCKED)].copy()
        w["task"], w["metric"] = "link prediction (AUC)", "auc"
        board = pd.concat([board, w[["dataset", "task", "graph_variant", "metric", "mean", "std", "seeds"]]])
    cells = gaps(board)
    cells = cells[(cells["task_family"] == "link prediction") & cells["usable"]].copy()
    props = pd.DataFrame([graph_properties(ds) for ds in cells["dataset"]])
    cells = cells.merge(props.drop(columns=["domain", "tasks", "graph_scope", "directed_source"]), on="dataset")
    cells["rule1_pred"] = [fr.predict_one(fr.LEAD, v) for v in cells["homophily_adjusted"]]
    cells["in_zone"] = cells["rule1_pred"] == "augment"
    d = cells[cells[vcol] != "tie"].reset_index(drop=True)
    base = [p == a for p, a in zip(d["rule1_pred"], d[vcol])]
    fit = {r.property: (r.threshold, r.augment_side) for r in scr[scr["target"] == target].itertuples() if r.augment_side}
    # The incumbent on the same graphs, from the Module-5 pre-registration table: the number the candidates have to match.
    m5 = cfg.RESULTS_DIR / "module5_predictions.csv"
    if m5.exists():
        r = pd.read_csv(m5)
        r = r[r["sim"] == CANONICAL_SIM][["dataset", fr.FROZEN_GATE.predictor]].drop_duplicates("dataset")
        cells = cells.merge(r, on="dataset", how="left")
        d = cells[cells[vcol] != "tie"].reset_index(drop=True)
    rows = [{"property": "rule 1 alone", "tier": "frozen", "conditions": f"{fr.LEAD.predictor} < {fr.LEAD.point}",
             "n_decided": len(d), "rule1_correct": int(sum(base)), "pair_correct": int(sum(base)), "delta": 0,
             "loo_panel_accuracy": float("nan"), "loo_panel_correct": 0, "loo_panel_folds": 0,
             "fixes": "", "breaks": "", "still_wrong": ", ".join(d.loc[[i for i in range(len(d)) if not base[i]], "dataset"]),
             "n_augment_calls": int(sum(p == "augment" for p in d["rule1_pred"])), "cells_moved": 0}]
    # Every candidate the panel screen put ahead of rule 1, single and paired, at its FITTED cut - never refitted here.
    names = [r.property for r in comp.itertuples() if r.delta > 0 and not r.tier.startswith("frozen gate")]
    for name in names:
        conds = [(q, fit[q][0], fit[q][1]) for q in name.split(" + ") if q in fit]
        if conds:
            rows.append(evaluate(d, vcol, base, conds, name, "candidate (panel-fitted cut)", fit_loo=False))
    if fr.FROZEN_GATE.predictor in d:
        rows.append(evaluate(d, vcol, base, [(fr.FROZEN_GATE.predictor, fr.FROZEN_GATE.point, "low")],
                             fr.FROZEN_GATE.predictor, "frozen gate (needs the role graph BUILT)", fit_loo=False))
    return cells, pd.DataFrame(rows)


# Guards the fitting: a dataset outside the declared panel must not silently move a rule that will later be tested on it.
def _assert_panel(datasets, allow_refit):
    '''Raise unless every dataset is inside PAIR_PANEL (or a deliberate re-fit was requested).'''
    extra = sorted(set(datasets) - set(PAIR_PANEL))
    assert allow_refit or not extra, (f"pair fitting must stay on PAIR_PANEL; {extra} are outside it. Add them to the panel "
                                      "deliberately, or pass --allow-refit to write exploratory_ copies instead.")


# Measures the graphs, then screens the second condition; measure always runs, screen reads its output.
def main(args):
    _assert_panel(args.datasets, args.allow_refit)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    cells = out["stage1_pair_cells"] = panel_cells(args.datasets)
    if args.step in ("screen", "all"):
        scr = out["stage1_pair_rules"] = screen(cells)
        comp = out["stage1_pair_compound"] = compound(cells, scr)
        pr = out["stage1_pair_two_condition"] = pairs(cells, scr)
        coll = out["stage1_pair_collinearity"] = collinearity(cells)
    if args.step in ("heldout", "all") and args.heldout:
        hcells, hout = heldout(args.heldout, scr, pd.concat([comp, pr]))
        out["stage1_pair_heldout_cells"], out["stage1_pair_heldout"] = hcells, hout
    prefix = "exploratory_" if args.allow_refit else ""
    for name, df in out.items():
        df.to_csv(cfg.RESULTS_DIR / f"{prefix}{name}.csv", index=False)
        print(f"{len(df):3d} rows -> results/{prefix}{name}.csv", flush=True)
    if args.step in ("screen", "all"):
        report(cells, scr, comp, pr, coll)
    if args.step in ("heldout", "all") and args.heldout:
        print("\nRETROSPECTIVE CHECK on graphs the screen never saw (cuts fitted on the panel, NOT refitted here).")
        print("  Not a pre-registration: these graphs were trained and scored before this screen existed.")
        print(hcells[["dataset", "homophily_adjusted", "degree_skew", "avg_clustering", "degree_gini", "in_zone",
                      "rule1_pred", "verdict", "gap_sigma"]].to_string(index=False))
        print()
        print(hout[["property", "conditions", "n_decided", "pair_correct", "delta", "fixes", "breaks", "still_wrong",
                    "n_augment_calls"]].to_string(index=False))


# Defines command-line options (mirrors experiments/gate_rules.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Stage-1 stabilization: screen original-graph properties as the second condition gating the homophily rule.")
    p.add_argument('--datasets', nargs='+', default=PAIR_PANEL, choices=list(STUDY),
                   help='Datasets to measure and screen. Default: the declared fitting panel (the Module-4 panel, for comparability).')
    p.add_argument('--step', default='all', choices=['measure', 'screen', 'heldout', 'all'],
                   help="'measure' = the graph descriptions only; 'screen' also fits the second condition; 'heldout' / 'all' also apply the fitted cuts to --heldout graphs. Default: all.")
    p.add_argument('--heldout', nargs='+', default=fr.GATE_HELDOUT, choices=list(STUDY),
                   help='Graphs the screen never saw, scored against the panel-fitted cuts. Retrospective, not pre-registered - their outcomes already exist.')
    p.add_argument('--allow-refit', action='store_true',
                   help='Deliberately fit outside PAIR_PANEL. Writes exploratory_*.csv so the panel results are never overwritten.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
