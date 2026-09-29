'''Module 14: is the degree candidate valid on the FAMILY of graphs it was discovered on? Describe, select, pre-register, then score.'''
# Module 10 fitted `nbr_label_entropy < 0.6724 => degree` on nine cells; Module 11 falsified it as a universal rule
# (1/3 on six pre-registered graphs, beaten 3/3 by a constant "always psi"). The user's 2026-09-07 instruction is to ask
# the narrower question instead, and to ask it cleanly:
#     old nine -> describe their structural profile -> pick NEW graphs with similar ORIGINAL-graph properties ->
#     apply the ALREADY-FIXED cut -> write the predictions to disk -> only then train degree and psi -> score.
# Two rules govern everything here. (1) The cut is NEVER refitted - `frozen_rules.FROZEN_ENTROPY` is read, never
# recomputed, and this script contains no call to threshold(). (2) Selection is on ORIGINAL-graph properties only, so
# no outcome can reach the choice of test set; `frozen_rules.FAMILY` excludes the predictor itself for the same reason.
#   step 1  profile   the discovery nine, and the envelope drawn from them
#   step 2  member    which graphs are in the family, and which axis each outsider fails
#   step 3  prereg    the fixed cut applied to family members that are not yet trained - written BEFORE the encoder runs
#   step 4  score     the same predictions against the signal the seeds name, once the sweep has run
# If it survives, the finding is "valid for this family", never "a degree rule".

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import frozen_rules as fr
from virgo import graph_io
from experiments.degree_rule import board
from experiments.strategy_select import SIGNAL, properties, winners

# The paired band names the winning signal, because the discovery nine were labelled under it: testing a cut fitted on
# paired verdicts against frozen-band labels would change the question, not the answer.
PAIRED_VERDICTS = cfg.RESULTS_DIR / "degree_rule_corpus_verdicts.csv"

# Four graphs (actor, pubmed, lastfm_asia, squirrel_filtered) no longer have a labels/ file, so graph_properties() can
# no longer recompute their label-derived properties and returns NaN. Their values were measured when the files existed
# and are carried in the cells table, so that table is the fallback source - the SAME numbers every earlier module was
# read against, which is the repo's rule: reuse the measured table, never take a second measurement that can drift.
MEASURED = cfg.RESULTS_DIR / "exploratory_paired_degree_cells.csv"


def props(datasets):
    '''The FAMILY axes plus the predictor per dataset, with gaps filled from the previously measured cells table.'''
    p = properties(datasets).set_index("dataset")
    cols = [c for c in list(fr.FAMILY) + ["nbr_label_entropy"] if c in p.columns]
    if MEASURED.exists():
        m = pd.read_csv(MEASURED).set_index("dataset")
        for c in cols:
            if c in m.columns:
                p[c] = p[c].fillna(m[c].reindex(p.index))
    return p


# Bipartiteness is not in the characterization table and is the one FAMILY axis that must be measured, so it is measured
# here - once per graph, off the same edgelist every other stage reads.
def bipartite(ds):
    '''Is this dataset's original graph bipartite? None when the edgelist is not on disk (tolokers is served from cache).'''
    path = cfg.dataset(ds)["edgelist"]
    return bool(nx.is_bipartite(graph_io.load_graph(path))) if Path(path).exists() else None


# STEP 1 + 2: the envelope is a DEFINITION taken from the discovery nine, so it contains all nine by construction. The
# table is printed anyway - a reader has to be able to see that it describes the nine rather than separating them.
def member(datasets, check_bipartite=True):
    '''One row per dataset: the FAMILY axes, whether it is in the family, and which axes an outsider fails.'''
    p = props(datasets)
    rows = []
    for ds in datasets:
        v = {k: p.loc[ds, k] for k in fr.FAMILY if k in p.columns}
        v["bipartite"] = bipartite(ds) if check_bipartite else False
        ok, bad = fr.in_family(v)
        rows.append({"dataset": ds, **{k: v[k] for k in fr.FAMILY}, "bipartite": v["bipartite"],
                     "nbr_label_entropy": p.loc[ds, "nbr_label_entropy"],
                     "in_family": ok, "fails": "|".join(bad),
                     # Where a graph stands relative to the discovery nine, stated as its ROLE, never as an outcome.
                     "role": ("discovery" if ds in fr.DISCOVERY_9 else
                              "spent" if ds in fr.ENTROPY_SPENT else "candidate")})
    return pd.DataFrame(rows).sort_values(["in_family", "role", "dataset"], ascending=[False, True, True]).reset_index(drop=True)


# STEP 3: the fixed cut applied to family members, with NO outcome column. Written before the encoder runs, which is what
# makes the later score a test rather than a description - the same order stage 1 uses.
def prereg(m, datasets):
    '''Every in-family candidate's stage-1 call and predicted signal under the frozen cut, with no outcome column.'''
    z = m[m["in_family"] & (m["role"] == "candidate")]
    # Stage 2 is CONDITIONAL on augmentation being indicated, so the stage-1 call is recorded with the prediction. A
    # graph stage 1 sends to "keep original" can carry no stage-2 cell at all, and that has to be on record beforehand.
    full = properties(datasets).set_index("dataset")
    gate = {d: fr.predict_gated(full.loc[d].to_dict())[2] for d in z["dataset"] if d in full.index}
    return pd.DataFrame({"dataset": z["dataset"], "predictor": fr.FROZEN_ENTROPY.predictor,
                         "value": z["nbr_label_entropy"], "cut": fr.FROZEN_ENTROPY.point,
                         "rule": f"{fr.FROZEN_ENTROPY.predictor} < {fr.FROZEN_ENTROPY.point} => degree",
                         "stage1": [gate.get(d, "n/a") for d in z["dataset"]],
                         "predicted": [fr.predict_entropy({fr.FROZEN_ENTROPY.predictor: v}) for v in z["nbr_label_entropy"]],
                         }).reset_index(drop=True)


# STEP 4: score the SAME predictions, under TWO labellings, both recorded so neither can be lost:
#   band    the tie-aware label every published cell uses - a cell whose band names two signals is unscorable, because
#           the seeds cannot tell those two apart.
#   argmax  the raw winner - whichever variant has the highest mean, however small the margin (user, 2026-09-07:
#           "a win is a win, even 0.00001"). This is the labelling the score below is reported on.
# The difference is not cosmetic and is stated wherever the number appears: an argmax label on a sub-sem margin is a
# coin flip re-read as a decision, so an argmax result is evidence about the RULE only as far as the margins allow. The
# margin is carried in the table (`margin`, `margin_sems`) next to every argmax call so a reader can see which is which.
def score(pre, band="sem"):
    '''The pre-registered predictions against the signal each graph's seeds name, plus the constant baselines.'''
    ds = list(pre["dataset"])
    # A graph mid-sweep has SOME variant rows but not all seven, and winners() asserts on that. Scoring in stages - the
    # cheap graphs read while the expensive one still trains - is legitimate because every prediction is already frozen
    # on disk, so only the COMPLETE graphs are passed through, and the rest are reported as "not trained".
    b = pd.read_csv(cfg.SCOREBOARD_CSV)
    b = b[(b["encoder"] == "graphsage_edge") & (b["top_K_neighbors"] == 10)
          & b["graph_variant"].isin(cfg.VG_SIMS) & b["task"].str.startswith("link prediction")]
    n_var = b.groupby("dataset")["graph_variant"].nunique()
    have = [d for d in ds if n_var.get(d, 0) == len(cfg.VG_SIMS)]
    if not have:
        return pd.DataFrame(columns=list(pre.columns) + ["winner_signals", "beats_original", "actual", "scored", "correct"])
    w = winners(have, band, board(have)).set_index("dataset")
    if PAIRED_VERDICTS.exists():                       # the discovery nine were labelled under the paired band
        v = pd.read_csv(PAIRED_VERDICTS).query("band == 'paired'").set_index("dataset")
        for d in have:
            if d in v.index:
                w.loc[d, "winner_signals"] = v.loc[d, "winner_signals"]
                w.loc[d, "beats_original"] = bool(v.loc[d, "beats_original"])
    # The raw argmax winner and how big its lead is, read off the same board winners() used.
    raw = pd.read_csv(cfg.SCOREBOARD_CSV)
    raw = raw[(raw["encoder"] == "graphsage_edge") & (raw["top_K_neighbors"] == 10)
              & raw["graph_variant"].isin(cfg.VG_SIMS) & raw["task"].str.startswith("link prediction")]
    rows = []
    for _, r in pre.iterrows():
        d = r["dataset"]
        sig = w.loc[d, "winner_signals"] if d in w.index else ""
        aug = bool(w.loc[d, "beats_original"]) if d in w.index else False
        # A band that names ONE signal is a decision even when that signal is centrality - it is then a failed degree
        # call, not an unscorable cell. Only a genuine multi-signal band is a tie.
        band = ("not trained" if d not in w.index else "no augmentation" if not aug else
                sig if sig in ("degree", "psi", "centrality") else "tie (" + sig + ")")
        # argmax over the AUGMENTED variants only - the question is which signal, given that augmentation is indicated.
        g = raw[raw["dataset"] == d].set_index("graph_variant")
        arg, margin, sems = "not trained", float("nan"), float("nan")
        if d in w.index and len(g):
            a = g.drop(index="original", errors="ignore")["mean"].sort_values(ascending=False)
            arg = SIGNAL[a.index[0]]
            rival = next((v for v in a.index[1:] if SIGNAL[v] != arg), None)   # nearest variant naming a DIFFERENT signal
            if rival is not None:
                margin = float(a.iloc[0] - a[rival])
                n1 = g.loc[a.index[0], "seeds"].count("|") + 1
                sd = np.hypot(g.loc[a.index[0], "std"], g.loc[rival, "std"]) / np.sqrt(2 * n1)
                sems = float(margin / sd) if sd else float("inf")
            if not aug:
                arg = "no augmentation"
        rows.append({**r.to_dict(), "winner_signals": sig, "beats_original": aug,
                     "actual_band": band, "actual_argmax": arg,
                     "margin": round(margin, 6) if margin == margin else margin,
                     "margin_sems": round(sems, 2) if sems == sems else sems,
                     "scored_band": band in ("degree", "psi", "centrality"),
                     "correct_band": bool(r["predicted"] == band) if band in ("degree", "psi", "centrality") else None,
                     "scored": arg in ("degree", "psi", "centrality"),
                     "correct": bool(r["predicted"] == arg) if arg in ("degree", "psi", "centrality") else None})
    return pd.DataFrame(rows)


def report(prof, m, pre, sc, out):
    '''Print the discovery profile and its envelope, the family membership, the pre-registered calls, and any score.'''
    print(f"\nSTEP 1 - THE DISCOVERY NINE ({', '.join(fr.DISCOVERY_9)})")
    print(prof.to_string(index=False))
    print("\nTHE FAMILY ENVELOPE - a DEFINITION drawn from those nine, so it contains all nine by construction")
    for k, (lo, hi) in fr.FAMILY.items():
        print(f"  {k:24s} [{lo}, {hi}]   (nine span {prof[k].min():.4f} - {prof[k].max():.4f})")
    print(f"  {'bipartite':24s} excluded" if fr.FAMILY_EXCLUDE_BIPARTITE else "")
    print(f"\nSTEP 2 - FAMILY MEMBERSHIP over {len(m)} graphs")
    print(m[["dataset", "role", "in_family", "fails", "nodes", "avg_degree", "density", "degree_assortativity",
             "largest_component_frac", "n_classes", "bipartite", "nbr_label_entropy"]].to_string(index=False))
    inf = m[m["in_family"]]
    print(f"\n  in family: {len(inf)} of {len(m)}  |  discovery {int((inf['role'] == 'discovery').sum())}/9, "
          f"spent {int((inf['role'] == 'spent').sum())}, candidates {int((inf['role'] == 'candidate').sum())}")
    print(f"\nSTEP 3 - PRE-REGISTERED CALLS under the FIXED cut ({fr.FROZEN_ENTROPY.predictor} < {fr.FROZEN_ENTROPY.point} => degree; never refitted)")
    print(pre.to_string(index=False) if not pre.empty else "  no in-family candidates")
    if not sc.empty:
        pend = sorted(sc["dataset"][sc["actual_band"] == "not trained"])
        print("\nSTEP 4 - SCORE" + (f"   [PRELIMINARY - still training: {', '.join(pend)}]" if pend else "   [COMPLETE]"))
        print(sc[["dataset", "value", "predicted", "winner_signals", "actual_band", "actual_argmax",
                  "margin", "margin_sems", "correct"]].to_string(index=False))
        c = [v for v in sc["correct"] if isinstance(v, bool)]
        cb = [v for v in sc["correct_band"] if isinstance(v, bool)]
        a = sc[sc["scored"]]["actual_argmax"].value_counts()
        n = int(a.sum())
        # Three readings, all printed, because the denominator is a CHOICE and hiding it would flatter the rule:
        #   argmax        every cell counts, a win is a win (user, 2026-09-07) - the headline
        #   band          only cells whose band names ONE signal; a centrality winner still counts as a failed degree call
        #   band, deg/psi Module 11's contrast, where a centrality winner is "not applicable" rather than wrong
        band1 = sc[sc["actual_band"].isin(["degree", "psi", "centrality"])]
        band2 = sc[sc["actual_band"].isin(["degree", "psi"])]
        print(f"  rule, ARGMAX labels (a win is a win):        {sum(c)}/{len(c)}" if c else "  nothing scorable yet")
        for name, g in [("BAND labels, centrality counts as wrong", band1),
                        ("BAND labels, degree-vs-psi only (M11)  ", band2)]:
            print(f"  rule, {name}: {int((g['predicted'] == g['actual_band']).sum())}/{len(g)}" if len(g)
                  else f"  rule, {name}: no cell qualifies")
        # The cells the data actually decides. A rule that only scores where the seeds cannot separate the signals has
        # not been tested by those cells, whatever the headline count says.
        dec = sc[sc["scored"] & (sc["margin_sems"] >= 1)]
        if len(dec):
            print(f"  rule on the {len(dec)} cell(s) with a margin >= 1 sem: "
                  f"{int(dec['correct'].sum())}/{len(dec)}  ({', '.join(dec['dataset'])})")
        if n:
            print("  constants (argmax): " + ", ".join(f"always {k} {v}/{n} = {v / n:.3f}" for k, v in a.items()))
            thin = sc[sc["scored"] & (sc["margin_sems"] < 1)]
            if len(thin):
                print(f"  CAUTION: {len(thin)} of {n} argmax calls sit under 1 sem "
                      f"({', '.join(f'{d} {m:.5f}' for d, m in zip(thin['dataset'], thin['margin']))}) - "
                      "at that margin the label is a coin flip re-read as a decision")
    print(f"\n{len(out):3d} tables written to results/degree_family_*.csv")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds = sorted(set(args.datasets) | set(fr.DISCOVERY_9))
    m = member(ds, not args.skip_bipartite)
    prof = m[m["role"] == "discovery"][["dataset"] + list(fr.FAMILY) + ["nbr_label_entropy"]]
    pre = prereg(m, ds)
    sc = score(pre, args.band) if args.score else pd.DataFrame()
    out = []
    for name, df in [("degree_family_profile", prof), ("degree_family_members", m),
                     ("degree_family_prereg", pre), ("degree_family_scored", sc)]:
        if not df.empty:
            df.to_csv(cfg.RESULTS_DIR / f"{name}.csv", index=False)
            out.append(name)
    report(prof, m, pre, sc, out)


# Defines command-line options (mirrors degree_rule.py / psi_rule.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Module 14: test the FIXED degree cut on the family of graphs it was discovered on.")
    p.add_argument('--datasets', nargs='*', default=[], help='Graphs to test for family membership. The discovery nine are always included.')
    p.add_argument('--band', default='sem', choices=['sem', 'sigma'], help="How close counts as tied. Default sem.")
    p.add_argument('--score', action='store_true', help='Score the pre-registered calls. Run this only AFTER the candidates are trained.')
    p.add_argument('--skip-bipartite', action='store_true', help='Skip the bipartite check (it loads every graph). Family membership is then unfiltered on that axis.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
