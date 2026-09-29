'''Module 4 - screen VIRTUAL-GRAPH properties as the SECOND variable that gates rule 1 (adjusted homophily).

Module 3 showed rule 1 fails asymmetrically: high adjusted homophily called "keep original" correctly every time, low
adjusted homophily called "augment" and was wrong half the time. So homophily is a reliable VETO, not a predictor, and a
second variable is only needed inside the low-homophily zone - which is where the two closest cells in the whole study
(0.0008 apart in homophily, opposite verdicts) sit, so no single-variable split can ever separate them.

The four candidates are properties of the VIRTUAL GRAPH, not of the original graph: how many distinct structural roles the
graph resolves, how much real adjacency the role edges recover, how much of the original graph survives the rewiring, and
how the edge volume changes. They need no labels, so they also cover ogbl_ddi, where rule 1 cannot be evaluated at all.

Read-only w.r.t. everything already frozen: measures built virtual graphs off disk, reads verdicts through the frozen
scoreboard, and never touches candidate_rules.csv, nested_loo.csv or module3_*.csv. It FITS, so it declares its own panel.'''

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo import frozen_rules as fr
from experiments.characterize import GATES, STUDY, TARGETS, frozen_inputs, gaps, loo_threshold, rho, threshold
from experiments.predict_module3 import rule_properties
from experiments.score_module3 import uniformity

# The Module-4 FITTING panel lives in virgo/frozen_rules.py beside the gate it produced, so the rule and the data it was
# fitted on can never drift apart. citeseer_linqs and proteins stay out (dropped from forward work 2026-07-29); ogbn_arxiv
# is node-classification only and the gate is a LINK-PREDICTION rule, so it carries no cell here.
GATE_PANEL = fr.GATE_PANEL

assert set(GATE_PANEL) <= set(STUDY), f"gate panel names an unregistered dataset: {sorted(set(GATE_PANEL) - set(STUDY))}"
assert {d for d in fr.DISCOVERY_PANEL if "LP" in STUDY[d]["tasks"]} <= set(GATE_PANEL), \
    "every link-prediction dataset rule 1 was fitted on must stay in the gate panel, or the gate is fitted on different evidence"

# The four candidate second variables, one per concept, plus two facets kept as exploratory. Only PRIMARY ones may be
# promoted and only they carry the multiple-comparison correction - same tiering rule as characterize.PREDICTORS_PRIMARY.
PROPERTIES_PRIMARY = ["role_diversity", "role_edge_enrichment", "original_retention", "vg_edge_ratio"]
PROPERTIES_EXPLORATORY = ["role_sampled_frac", "role_edge_overlap"]
PROPERTIES = PROPERTIES_PRIMARY + PROPERTIES_EXPLORATORY

CANONICAL_SIM = "psi"      # the variant under study; the other role graphs are measured as a robustness check, never as the rule
COLLINEAR = 0.99           # |rho| at or above which two properties are reported as one finding, not two


# Where the pipeline wrote each virtual graph (canonical layout, virtual_graph.py's own default output path).
def vg_path(ds, sim, k):
    '''Path of the built virtual edgelist for one dataset x K x variant.'''
    return cfg.NB2_DIR / "virtual_graphs" / ds / f"k{k}" / sim / "virtual_graph.edgelist"


# FAMILY 3: what the rewiring actually did to one graph - measured on the edgelist the encoder was really trained on.
def vg_measure(ds, sims, k):
    '''One row per (dataset, variant): role resolution, how much real adjacency the role edges recover, retention, volume.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    n, m = G.number_of_nodes(), G.number_of_edges()
    density = 2 * m / (n * (n - 1))
    A = {(min(u, v), max(u, v)) for u, v in G.edges}
    # a hybrid is original union a role top-K, so its ROLE signal is that role's: diversity is measured on the underlying signature.
    base = {s: {"hybrid": "psi", "hybrid_degree": "degree", "hybrid_centrality": "centrality"}.get(s, s) for s in sims}
    div = uniformity([ds], k=k, sims=tuple(sorted(set(base.values())))).iloc[0]
    rows = []
    for sim in sims:
        p = vg_path(ds, sim, k)
        assert p.exists(), f"{p} missing -> build the virtual graph for {ds} / {sim} first (virgo/virtual_graph.py writes it)"
        V = nx.read_weighted_edgelist(p, nodetype=int)
        B = {(min(u, v), max(u, v)) for u, v in V.edges}
        shared, nv = len(A & B), len(B)
        overlap = shared / nv if nv else float("nan")
        rows.append({
            "dataset": ds, "sim": sim, "k": k, "nodes": n, "edges": m, "virtual_edges": nv,
            # How finely the signature resolves roles: distinct tie classes per node, and the share of nodes whose class
            # alone can fill K, whose role edges are therefore drawn at RANDOM inside the class rather than ranked.
            "role_diversity": float(div[f"distinct_{base[sim]}"]),
            "role_sampled_frac": float(div[f"sampled_frac_{base[sim]}"]),
            # Do role edges land on real ones more often than chance? overlap is the raw share, enrichment is that share
            # over the graph's density - the lift, which is what compares across graphs of very different sparsity.
            "role_edge_overlap": round(overlap, 6),
            "role_edge_enrichment": round(overlap / density, 2) if density and overlap == overlap else float("nan"),
            # What the rewiring DESTROYS (share of original edges the encoder still sees) and what it costs in volume.
            "original_retention": round(shared / m, 6) if m else float("nan"),
            "vg_edge_ratio": round(nv / m, 4) if m else float("nan"),
        })
    return rows


# The quantity to be explained, exactly as Module 2 and 3 read it: the link-prediction keep/augment verdict per dataset,
# plus the rule-1 property that decides whether the cell is inside the ambiguous zone at all.
def gate_cells(datasets):
    '''Per dataset: the LP gap and verdict from the frozen scoreboard, its adjusted homophily, and whether rule 1 fires.'''
    g = gaps(frozen_inputs(datasets))
    g = g[(g["task_family"] == "link prediction") & g["usable"]].copy()
    props = pd.DataFrame([rule_properties(ds) for ds in g["dataset"]])[["dataset", "homophily_adjusted", "largest_component_frac"]]
    g = g.merge(props, on="dataset")
    # The zone is where rule 1 says "augment" - the only place a second variable is needed. A graph with no labels has no
    # homophily, so rule 1 cannot fire on it: it is outside the zone by absence, which is itself the coverage hole to report.
    g["rule1_fires"] = [fr.predict_one(fr.LEAD, v) == "augment" for v in g["homophily_adjusted"]]
    g["in_zone"] = g["rule1_fires"]
    return g[["dataset", "metric", "original", "best_augmented", "best_variant", "gap_rel", "gap_fixed_rel", "gap_sigma",
              "verdict", "verdict_fixed", "homophily_adjusted", "largest_component_frac", "in_zone"]]


# The pair that makes a second variable necessary: opposite verdicts at the closest adjusted homophily. No split on
# homophily alone can separate it, so a candidate that cannot separate it has not solved the problem it was proposed for.
def decisive_pair(d, vcol):
    '''The two closest-in-homophily cells with opposite verdicts, or None when the scope holds no such pair.'''
    h = d[d["homophily_adjusted"].notna()]
    best = None
    for i, a in h.iterrows():
        for j, b in h.iterrows():
            if a[vcol] == b[vcol] or a["dataset"] >= b["dataset"]:
                continue
            gap = abs(float(a["homophily_adjusted"]) - float(b["homophily_adjusted"]))
            if best is None or gap < best[0]:
                best = (gap, a["dataset"], b["dataset"])
    return best


# Screen every candidate as a binary keep-vs-augment split, in two scopes: inside the low-homophily zone (the GATE it was
# proposed for) and over every decided cell (does it work standalone, without homophily?). Same gates as Module 2.
def screen(cells, props, sims):
    '''One row per variant x scope x gap definition x property: rho, split, interval, out-of-sample accuracy, credibility.'''
    rows = []
    for sim in sims:
        joined = cells.merge(props[props["sim"] == sim].drop(columns=["sim"]), on="dataset")
        for scope, sub in (("low-homophily zone", joined[joined["in_zone"]]), ("all decided cells", joined)):
            for target, vcol in TARGETS.items():
                s = sub[sub[vcol] != "tie"].reset_index(drop=True)
                pair = decisive_pair(s, vcol)
                for prop in PROPERTIES:
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
                    # Does the fitted split call BOTH members of the decisive pair right? A property that only works by
                    # re-separating cells homophily already separates adds nothing to the rule it is meant to gate.
                    sep = float("nan")
                    if pair and side:
                        mem = d[d["dataset"].isin(pair[1:])]
                        if len(mem) == 2:
                            sep = bool(all(((float(v) > t) if side == "high" else (float(v) < t)) == (w == "augment")
                                           for v, w in zip(mem[prop], mem[vcol])))
                    rows.append({
                        "sim": sim, "canonical_sim": bool(sim == CANONICAL_SIM), "scope": scope, "target": target,
                        "property": prop, "tier": "primary" if prop in PROPERTIES_PRIMARY else "exploratory",
                        "n_decided": len(d), "n_augment": int(y.sum()), "distinct_values": int(d[prop].nunique()),
                        "distinct_augment": int(d[prop][y].nunique()),
                        "spearman_rho": r, "p_value": pv,
                        "p_bonferroni": round(min(1.0, pv * len(PROPERTIES_PRIMARY)), 4)
                                        if prop in PROPERTIES_PRIMARY and pv == pv else float("nan"),
                        "lodo_min_abs_rho": round(min(abs(v) for v in lodo), 4) if lodo else float("nan"),
                        "lodo_sign_stable": stable,
                        "threshold": t, "augment_side": side, "n_exceptions": err, "accuracy": acc,
                        # The panel pins an INTERVAL, not a point: every cut between these two values fits it equally well.
                        "interval_lo": lo, "interval_hi": hi,
                        # Out-of-sample: the split refitted with each dataset hidden. `accuracy` is in-sample and cannot fail.
                        "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
                        "loo_threshold_lo": l_lo, "loo_threshold_hi": l_hi,
                        "majority_baseline": major, "loo_beats_majority": bool(l_acc == l_acc and l_acc > major),
                        "decisive_pair": " vs ".join(pair[1:]) if pair else "",
                        "separates_decisive_pair": sep,
                        "rule": f"in the low-homophily zone, augment when {prop} is {side} ({'>' if side == 'high' else '<'} {t})"
                                if side and scope == "low-homophily zone" else
                                (f"augment when {prop} is {side} ({'>' if side == 'high' else '<'} {t})" if side else ""),
                        "credible": bool(r == r and abs(r) >= GATES["min_abs_rho"] and stable
                                         and 0 <= err <= GATES["max_exceptions"] and n >= GATES["min_cells"]
                                         and (l_acc == l_acc and l_acc > major if GATES["loo_above_majority"] else True)),
                    })
    return pd.DataFrame(rows)


# Which candidates are the SAME measurement: collinear properties must collapse to one finding, never be counted twice.
def collinearity(props, sim):
    '''Spearman between every pair of candidates on one variant, flagging the pairs that carry identical information.'''
    p = props[props["sim"] == sim]
    rows = []
    for i, a in enumerate(PROPERTIES):
        for b in PROPERTIES[i + 1:]:
            n, r, _ = rho(p[a], p[b])
            rows.append({"sim": sim, "property_a": a, "property_b": b, "n": n, "spearman_rho": r,
                         "same_information": bool(r == r and abs(r) >= COLLINEAR)})
    return pd.DataFrame(rows)


# The reading of the screen: what the gate would be, and every reason it is still only a candidate.
def report(cells, props, scr, coll):
    '''Print the gate conclusion: the zone, the candidates, the decisive pair, and what remains untested.'''
    print("\nLINK-PREDICTION CELLS (frozen scoreboard; the zone is where rule 1 fires, i.e. adjusted homophily < "
          f"{fr.LEAD.point})")
    print(cells[["dataset", "homophily_adjusted", "in_zone", "original", "best_augmented", "best_variant",
                 "gap_sigma", "verdict", "verdict_fixed"]].to_string(index=False))
    z = cells[cells["in_zone"]]
    print(f"\n  zone holds {len(z)} cells: " + " | ".join(f"{v}: {n}" for v, n in z["verdict"].value_counts().items()))
    blind = cells[cells["homophily_adjusted"].isna()]["dataset"].tolist()
    if blind:
        print("  rule 1 cannot be evaluated on (no labels -> no homophily), so the gate never opens there: " + ", ".join(blind))
    print(f"\nFAMILY 3 - what the rewiring did (variant '{CANONICAL_SIM}', the one under study)")
    print(props[props["sim"] == CANONICAL_SIM][["dataset", "edges", "virtual_edges", "role_diversity", "role_sampled_frac",
                                                "role_edge_overlap", "role_edge_enrichment", "original_retention",
                                                "vg_edge_ratio"]].to_string(index=False))
    print("\n  collinear candidates (|rho| >= %.2f -> ONE finding, not two): " % COLLINEAR +
          (" | ".join(f"{r.property_a} ~ {r.property_b} ({r.spearman_rho:+.2f})"
                      for r in coll[coll["same_information"]].itertuples()) or "none"))
    print(f"\nCANDIDATE GATES (variant '{CANONICAL_SIM}'; gates: |rho|>={GATES['min_abs_rho']}, LODO sign-stable, "
          f"<={GATES['max_exceptions']} exception, leave-one-out above the majority baseline)")
    c = scr[scr["canonical_sim"]]
    print(c[["scope", "target", "property", "tier", "n_decided", "n_augment", "distinct_values", "spearman_rho",
             "lodo_min_abs_rho", "threshold", "augment_side", "n_exceptions", "loo_accuracy", "majority_baseline",
             "separates_decisive_pair", "credible"]].to_string(index=False))
    # Integrity check, not a fit: the frozen gate must still BE the row this screen produces on its own panel.
    z = c[(c["scope"] == "low-homophily zone") & (c["target"] == "gap_rel") & (c["property"] == fr.FROZEN_GATE.predictor)]
    if len(z) == 1 and set(cells["dataset"]) == set(GATE_PANEL):
        assert abs(float(z["threshold"].iloc[0]) - fr.FROZEN_GATE.point) < 1e-9, \
            (f"the screen now fits {float(z['threshold'].iloc[0])} but frozen_rules.FROZEN_GATE.point is "
             f"{fr.FROZEN_GATE.point} - the gate has drifted from the evidence it was locked on")
    keep = c[c["credible"] & (c["tier"] == "primary")]
    # One property clearing several scopes is ONE candidate, not several: the scopes differ only in which cells were fitted.
    print("\n  CREDIBLE CANDIDATES: " + (", ".join(dict.fromkeys(keep["property"])) if len(keep) else
                                         "none - no virtual-graph property clears the gates"))
    for r in keep.itertuples():
        print(f"    [{r.scope} / {r.target}] {r.rule}"
              f"\n      any cut in ({r.interval_lo}, {r.interval_hi}) fits the panel equally well -> quote the interval"
              f"\n      leave-one-out {r.loo_correct}/{r.loo_folds} = {r.loo_accuracy} vs {r.majority_baseline} for always guessing"
              f" the majority verdict; fold cutoffs ranged {r.loo_threshold_lo} to {r.loo_threshold_hi}"
              f"\n      separates the decisive pair ({r.decisive_pair}): {r.separates_decisive_pair}")
    # The point of a second variable was the pair homophily cannot split. A candidate that still splits it wrong has moved
    # the error count without solving the case it was proposed for - the single most important thing to say about it.
    fail = keep[keep["separates_decisive_pair"].apply(lambda v: v is False)]["property"].unique()
    if len(fail):
        print("  DOES NOT SOLVE THE MOTIVATING CASE - still calls the decisive pair wrong, so it improves the count without "
              "fixing the cell a second variable was proposed for: " + ", ".join(fail))
    thin = keep[keep["distinct_augment"] <= 1]["property"].unique()
    if len(thin):
        print("  WEAK EVIDENCE - every augmenting graph shares one value, so this is a two-group split, not a graded trend: "
              + ", ".join(thin))
    robust = scr[~scr["canonical_sim"] & scr["credible"] & (scr["tier"] == "primary")]
    print("\n  ROBUSTNESS - the same property on the other role graphs: " +
          (" | ".join(f"{r.sim}/{r.scope}: {r.property}" for r in robust.drop_duplicates(["sim", "property"]).itertuples())
           or "no other variant reproduces a credible split"))
    print("\n  STATUS: fitted on GATE_PANEL, which INCLUDES the Module-3 datasets - they are no longer unseen, so nothing")
    print("  above is validated. A credible candidate here is a Module-5 prediction to be pre-registered on a third,")
    print("  genuinely unseen set; the untested quadrant is a heterophilous graph that is also FRAGMENTED.")


# Guards the fitting: a dataset outside the declared panel must not silently move a rule that will later be tested on it.
def _assert_panel(datasets, allow_refit):
    '''Raise unless every dataset is inside GATE_PANEL (or a deliberate re-fit was requested).'''
    extra = sorted(set(datasets) - set(GATE_PANEL))
    assert allow_refit or not extra, (f"gate fitting must stay on GATE_PANEL; {extra} are outside it. Add them to the panel "
                                      "deliberately, or pass --allow-refit to write exploratory_ copies instead.")


# Measures the virtual graphs, then screens them as the gate; measure always runs, screen reads its output.
def main(args):
    _assert_panel(args.datasets, args.allow_refit)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    props = out["vg_characterization"] = pd.DataFrame([r for ds in args.datasets for r in vg_measure(ds, args.sims, args.k)])
    if args.step in ("screen", "all"):
        cells = gate_cells(args.datasets)
        scr = out["gate_candidates"] = screen(cells, props, args.sims)
        coll = out["gate_collinearity"] = collinearity(props, CANONICAL_SIM)
    prefix = "exploratory_" if args.allow_refit else ""
    for name, df in out.items():
        df.to_csv(cfg.RESULTS_DIR / f"{prefix}{name}.csv", index=False)
        print(f"{len(df):3d} rows -> results/{prefix}{name}.csv", flush=True)
    if args.step in ("screen", "all"):
        report(cells, props, scr, coll)


# Defines command-line options (mirrors experiments/characterize.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Module 4: screen virtual-graph properties as the second variable gating the homophily rule.")
    p.add_argument('--datasets', nargs='+', default=GATE_PANEL, choices=list(STUDY),
                   help='Datasets to measure and screen. Default: the declared Module-4 fitting panel.')
    p.add_argument('--sims', nargs='+', default=[s for s in cfg.VG_SIMS_LOCKED if s != "original"],
                   help="Virtual-graph variants to measure. 'psi' is canonical; the rest are a robustness check. "
                        "Default = the LOCKED five the frozen gate was fitted on (name a promoted variant to measure it too).")
    p.add_argument('--k', type=int, default=10, help='Top-K the virtual graphs were built with. Default 10 (the locked setting).')
    p.add_argument('--step', default='all', choices=['measure', 'screen', 'all'],
                   help="'measure' = Family 3 only; 'screen' / 'all' also fit the gate. Default: all.")
    p.add_argument('--allow-refit', action='store_true',
                   help='Deliberately fit outside GATE_PANEL. Writes exploratory_*.csv so the panel results are never overwritten.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
