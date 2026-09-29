'''Module 7 - PREDICT and SCORE the frozen STRATEGY rule on a genuinely unseen set (frozen_rules.STRATEGY_HELDOUT).

Same contract as Modules 3 and 5: the pre-registered call is written BEFORE any encoder runs, and this script never fits.
It imports virgo.frozen_rules for the rule and strategy_select.winners() for the outcome - winners() only READS the
scoreboard - and never calls threshold(), patterns() or contrast(), so an unseen dataset can only be scored against the
locked cut, never move it. The datasets were fixed in frozen_rules.STRATEGY_HELDOUT before any of them was measured.

The rule is CONDITIONAL ("augmentation is already indicated"), so a dataset where the ORIGINAL graph wins does not test
it: that cell is reported as precondition failed and excluded from the count, never silently dropped.'''

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo import frozen_rules as fr
from experiments.characterize import labels_of, neighbour_predictability
from experiments.strategy_select import BANDS, winners

PRED_CSV = cfg.RESULTS_DIR / "module7_predictions.csv"
SCORED_CSV = cfg.RESULTS_DIR / "module7_scored.csv"
PANEL_CSV = cfg.RESULTS_DIR / "strategy_winners.csv"          # Module-6 fitted panel, READ-ONLY here (baseline only)

# What strategy_property() can produce without the feature cache. Guards against a future strategy rule needing a
# property this pre-training measurer does not compute.
PRODUCES = {"nbr_predictability_adjusted"}
assert fr.FROZEN_STRATEGY.predictor in PRODUCES, "the frozen strategy rule needs a property predict_strategy does not measure"


def strategy_property(ds):
    '''The frozen rule's input for one dataset, measured cache-free from the original graph and its labels.'''
    d = cfg.dataset(ds)
    G = graph_io.load_graph(d["edgelist"])
    pred_adj = neighbour_predictability(G, labels_of(d["labels"], G))[0]     # (ADJUSTED, raw, entropy)
    return {"dataset": ds, fr.FROZEN_STRATEGY.predictor: round(pred_adj, 4) if pred_adj == pred_adj else float("nan")}


def predictions(datasets):
    '''One row per unseen dataset: the property, the frozen cut and its interval, and the pre-registered signal call.'''
    # Same guard predict_gate.py uses: a dataset the rule was FITTED on cannot test it, however the call comes out.
    fitted = sorted(set(datasets) & set(fr.STRATEGY_PANEL))
    assert not fitted, f"{fitted} are in STRATEGY_PANEL - the rule was fitted on them, so they cannot test it."
    rows = []
    for ds in datasets:
        p = strategy_property(ds)
        s = fr.FROZEN_STRATEGY
        rows.append({"dataset": ds, s.predictor: p[s.predictor],
                     "cut": s.point, "interval": f"({s.interval[0]}, {s.interval[1]})",
                     "applies_when": s.applies_when, "predicted_signal": fr.predict_strategy(p)})
    return pd.DataFrame(rows)


def freeze(datasets):
    '''Write-once pre-registration: already-saved rows are kept EXACTLY as frozen, only unseen datasets are appended.'''
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


# The rule's output space is BINARY - centrality, or not centrality - so a tie is only undecided when it straddles that
# line. "psi|degree" leaves the rule's own question answered (not centrality) even though Module 6's finer question is not.
def actual_signal(row):
    '''(actual signal, scored?, why): read the seeds' winner set against the rule's binary centrality/not-centrality question.'''
    sig = set(row["winner_signals"].split("|"))
    if not row["beats_original"]:
        return "", False, "precondition failed - the original graph wins, so no augmentation is indicated"
    if sig == {"centrality"}:
        return fr.FROZEN_STRATEGY.signal, True, ""
    if fr.FROZEN_STRATEGY.signal not in sig:
        return "psi or degree (undetermined)", True, ""
    return "|".join(sorted(sig)), False, "undecided - the seed band cannot separate centrality from the other winners"


def score(datasets=None, band="sem"):
    '''Join the pre-training predictions to the winner the seeds actually produced. Never fits.'''
    assert PRED_CSV.exists(), f"{PRED_CSV} missing - run --step predict BEFORE training first."
    pred = pd.read_csv(PRED_CSV)
    dsets = datasets or pred["dataset"].tolist()
    win = winners(dsets, band)
    win = win[win["task"] == "link prediction (AUC)"]
    m = pred[pred["dataset"].isin(dsets)].merge(win, on="dataset", how="left")
    got = [actual_signal(r) for _, r in m.iterrows()]
    m["actual_signal"], m["scored"], m["not_scored_reason"] = [g[0] for g in got], [g[1] for g in got], [g[2] for g in got]
    m["correct"] = [bool(s and p == a) for s, p, a in zip(m["scored"], m["predicted_signal"], m["actual_signal"])]
    return m


# The honest reading: how many cells could test the rule at all, how many it got right, and what guessing would have scored.
def report(m, band):
    '''Print the pred-vs-actual table, the precondition failures, and the 2/3 pre-registered criterion.'''
    print(f"\nMODULE 7 - THE FROZEN STRATEGY RULE ON UNSEEN DATA (band = {band})")
    print(f"RULE  use {fr.FROZEN_STRATEGY.signal} when {fr.FROZEN_STRATEGY.predictor} > {fr.FROZEN_STRATEGY.point} "
          f"[interval {fr.FROZEN_STRATEGY.interval}], only when {fr.FROZEN_STRATEGY.applies_when}")
    print(m[["dataset", fr.FROZEN_STRATEGY.predictor, "predicted_signal", "original", "best_variant", "best_score",
             "winner_set", "winner_signals", "beats_original", "actual_signal", "scored", "correct"]].to_string(index=False))
    for _, r in m[~m["scored"]].iterrows():
        print(f"  NOT SCORED  {r['dataset']}: {r['not_scored_reason']}")
    n, ok = int(m["scored"].sum()), int(m["correct"].sum())
    if PANEL_CSV.exists():                                    # what always guessing the fitted panel's commonest call scores
        p = pd.read_csv(PANEL_CSV)
        p = p[p["beats_original"]] if "beats_original" in p else p
        cent = int((p["winner_signals"] == fr.FROZEN_STRATEGY.signal).sum())
        print(f"\nPANEL BASELINE  centrality won alone in {cent}/{len(p)} of the fitted panel's augmenting datasets - "
              f"always guessing the majority call is the bar {ok}/{n} has to clear.")
    print(f"\nSCORED {ok}/{n} correct  ({len(m) - n} of {len(m)} cells could not test the rule)")
    print("SUPPORTING EVIDENCE" if ok >= 2 else "NOT SUPPORTED", f"- the pre-registered criterion was 2 of 3 correct; {ok} correct.")
    if n < 3:
        print(f"CAVEAT  only {n} of 3 datasets actually tested the rule, so this is weaker than a 3-dataset test either way.")


# --- the held-out story, as two tables --------------------------------------------------------------------------------
# Every dataset put through the two stages, including the withdrawn first batch, so the record shows what was tried and
# not only what survived. WITHDRAWN holds that batch's measured scores; its graphs and scoreboard rows were deleted at the
# user's request on 2026-08-14, so the numbers live here rather than being recomputable.
WITHDRAWN_CSV = cfg.RESULTS_DIR / "module7_withdrawn.csv"
STAGE1_CSV = cfg.RESULTS_DIR / "module5_predictions.csv"


def variant_scores(datasets):
    '''Per (dataset, variant) LP mean/std at 10 seeds: the live scoreboard, plus the withdrawn batch's kept numbers.'''
    b = pd.read_csv(cfg.SCOREBOARD_CSV)
    b = b[(b["encoder"] == "graphsage_edge") & (b["top_K_neighbors"] == 10)
          & (b["task"] == "link prediction (AUC)") & b["dataset"].isin(datasets)]
    b = b[["dataset", "graph_variant", "mean", "std"]].assign(seeds=b["seeds"].str.count(r"\|") + 1)
    w = pd.read_csv(WITHDRAWN_CSV) if WITHDRAWN_CSV.exists() else pd.DataFrame(columns=b.columns)
    w = w[w["dataset"].isin(datasets)][["dataset", "graph_variant", "mean", "std", "seeds", "nbr_predictability_adjusted"]]
    return pd.concat([b, w], ignore_index=True)


# STAGE 1: who even reaches stage 2. A "keep original" call ends the dataset's run - the strategy rule is conditional, so
# testing it on a graph the earlier rules say not to augment would score it on a question it does not claim to answer.
def routing(datasets):
    '''One row per held-out dataset: stage-1 inputs and call, whether stage 2 ran, and the frozen stage-2 prediction.'''
    s1 = pd.read_csv(STAGE1_CSV) if STAGE1_CSV.exists() else pd.DataFrame(columns=["dataset"])
    s2 = pd.read_csv(PRED_CSV) if PRED_CSV.exists() else pd.DataFrame(columns=["dataset"])
    wd = pd.read_csv(WITHDRAWN_CSV) if WITHDRAWN_CSV.exists() else pd.DataFrame(columns=["dataset"])
    rows = []
    for ds in datasets:
            a, b = s1[s1["dataset"] == ds], s2[s2["dataset"] == ds]
            call = a["predicted_verdict"].iloc[0] if len(a) else "not run (batch predates stage 1)"
            # The withdrawn batch kept its measured property but lost its prediction row; the rule is deterministic, so
            # re-applying the FROZEN cut to the SAME stored value reproduces the call it was given - it is not a re-fit.
            v = (b[fr.FROZEN_STRATEGY.predictor].iloc[0] if len(b) else
                 wd.loc[wd["dataset"] == ds, fr.FROZEN_STRATEGY.predictor].iloc[0] if (wd["dataset"] == ds).any() else float("nan"))
            ran = bool(len(b)) or (wd["dataset"] == ds).any()
            rows.append({"dataset": ds,
                         "homophily_adjusted": a["homophily_adjusted"].iloc[0] if len(a) else float("nan"),
                         "original_retention": round(float(a["original_retention"].iloc[0]), 4) if len(a) else float("nan"),
                         "stage1_call": call,
                         "nbr_predictability_adjusted": v, "stage2_ran": ran,
                         "stage2_prediction": (fr.predict_strategy({fr.FROZEN_STRATEGY.predictor: v}) if ran else
                                               "skipped - stage 1 says keep original" if call == "keep original" else "not run")})
    return pd.DataFrame(rows)


# STAGE 2: the winner against the RUNNER-UP, whatever it is - original or another virtual graph. Two verdicts, because
# they answer different questions and the panel rule was frozen under the second one:
#   by_margin  the highest mean wins, however small the gap. A win is a win.
#   by_band    the gap must also clear the standard error of the difference over the seeds, i.e. the seeds can tell the
#              two apart. Anything inside the band is unresolved - NOT a loss for the leader.
def outcome(datasets, band_k=1.0):
    '''Per dataset: best variant vs runner-up, the margin, the seed band, and the frozen prediction judged both ways.'''
    sc, s2 = variant_scores(datasets), pd.read_csv(PRED_CSV)
    rows = []
    for ds in datasets:
        g = sc[sc["dataset"] == ds].sort_values("mean", ascending=False).reset_index(drop=True)
        if g.empty:
            continue
        (v1, m1, s1_, n1), (v2, m2, s2_, n2) = [tuple(g.loc[i, ["graph_variant", "mean", "std", "seeds"]]) for i in (0, 1)]
        band = band_k * float(np.sqrt(s1_ ** 2 / n1 + s2_ ** 2 / n2))       # sem of the difference of the two means
        orig = float(g.loc[g["graph_variant"] == "original", "mean"].iloc[0])
        pred = s2.loc[s2["dataset"] == ds, "predicted_signal"]
        pred = pred.iloc[0] if len(pred) else fr.predict_strategy({fr.FROZEN_STRATEGY.predictor: float(g["nbr_predictability_adjusted"].iloc[0])})
        sig1, sig2 = SIGNAL_OF[v1], SIGNAL_OF[v2]
        rows.append({"dataset": ds, "best_variant": v1, "best_score": round(float(m1), 4),
                     "runner_up": v2, "runner_up_score": round(float(m2), 4),
                     "margin": round(float(m1 - m2), 4), "seed_band": round(band, 4),
                     "margin_clears_band": bool(m1 - m2 > band),
                     "original": round(orig, 4), "original_rank": int(g.index[g["graph_variant"] == "original"][0]) + 1,
                     "vs_original": round(float(m1) - orig, 4),
                     "predicted_signal": pred, "winning_signal": sig1, "runner_up_signal": sig2,
                     # A win is a win: the top mean carries the prediction, whatever the gap.
                     "correct_by_margin": bool(pred.startswith(sig1) if sig1 != "original" else pred == "original"),
                     # The stricter reading: only counted when the seeds can separate the top two AND they differ in signal.
                     "resolved": bool(m1 - m2 > band or sig1 == sig2)})
    d = pd.DataFrame(rows)
    d["correct_by_band"] = [c if r else None for c, r in zip(d["correct_by_margin"], d["resolved"])]
    return d


# The reading table: one row per held-out dataset, in words rather than columns of numbers. Everything in it is derived
# from routing() and outcome(), so it cannot drift from the measured values.
PRETTY = {"original": "Original", "psi": "Ψ", "degree": "Degree", "centrality": "Centrality",
          "hybrid": "Hybrid", "hybrid_degree": "Hybrid-degree", "hybrid_centrality": "Hybrid-centrality"}


# A stage-1 KEEP row is NOT self-justifying: if the graph was trained anyway, its scores can contradict the call, and
# reed98 (2026-08-15) is exactly that case. Read against the same 1-sigma pooled band actual_lp_verdicts() uses, so this
# table can never disagree with results/module5_scored.csv.
def keep_verdict(ds):
    '''Interpretation for a stage-1 KEEP row, checked against the measured scores whenever the graph was trained anyway.'''
    g = variant_scores([ds])
    if g.empty:
        return "Stage 1 said KEEP → not trained, call unscored"
    o = g[g["graph_variant"] == "original"].iloc[0]
    a = g[g["graph_variant"] != "original"].sort_values("mean", ascending=False).iloc[0]
    gap, noise = float(a["mean"] - o["mean"]), float(np.hypot(o["std"], a["std"]) / np.sqrt(2))
    return (f"Stage 1 said KEEP, but {PRETTY[a['graph_variant']]} beat the original by {gap:+.4f} ({gap / noise:.2f}σ) → call WRONG"
            if gap > noise else "Stage 1 said KEEP → augmentation unnecessary")


def summary(route, out):
    '''The held-out story as one readable table: how each dataset was routed, what was predicted, what won, what it means.'''
    o = out.set_index("dataset")
    rows = []
    for _, r in route.iterrows():
        ds, staged = r["dataset"], r["dataset"] in o.index
        gate = r["stage1_call"]
        w = o.loc[ds] if staged else None
        # The earlier batch ran before stage 1 existed, so its stage-2 call is retrospective - marked with * throughout,
        # never counted as a pre-registered test.
        star = "*" if staged and gate != "augment" else ""
        used = ("Yes" if gate == "augment" else
                "No" if not staged or w["winning_signal"] == "original" else "Yes*")
        pred = (PRETTY[fr.FROZEN_STRATEGY.signal] + star) if used.startswith("Yes") else "N/A"
        keep_row = not used.startswith("Yes")                  # KEEP = stage 2 never ran, so no result is reported
        rows.append({
            "Dataset": ds, "Test batch": "Full 2-stage" if gate != "not run (batch predates stage 1)" else "Earlier test",
            # "KEEP" for the earlier batch is INFERRED from its outcome, not a measured call - stage 1 did not exist yet,
            # and those graphs were deleted, so it cannot be computed now. Measured only where Test batch is "Full 2-stage".
            "Stage 1: KEEP or AUGMENT?": {"augment": "AUGMENT"}.get(gate, "KEEP"),
            "Stage 2 used?": used, "Stage 2 prediction": pred,
            # Every KEEP row reads the same, whether or not it was trained: stage 2 never ran, so its scores are not the
            # question. "not applicable" rather than "not trained" - two of these graphs WERE trained (the withdrawn
            # batch, scores in module7_withdrawn.csv), and saying otherwise would put a false claim in the record.
            "Best result": f"{PRETTY[w['best_variant']]} {w['best_score']:.4f}" if keep_row is False else "- (not applicable)",
            "2nd-best result": f"{PRETTY[w['runner_up']]} {w['runner_up_score']:.4f}" if keep_row is False else "- (not applicable)",
            "Final interpretation":
                keep_verdict(ds) if keep_row else
                f"Correct: {w['winning_signal']} best" if w["correct_by_margin"] else
                f"Augmentation helped, but {fr.FROZEN_STRATEGY.signal} prediction failed",
        })
    return pd.DataFrame(rows)


# Variant -> the structural signal it adds; the union hybrids collapse onto the signal they add (same map strategy_select uses).
SIGNAL_OF = {"original": "original", "psi": "psi", "hybrid": "psi", "degree": "degree", "hybrid_degree": "degree",
             "centrality": "centrality", "hybrid_centrality": "centrality"}


def main(args):
    if args.step == "predict":
        out, new = freeze(args.datasets)
        print(out.to_string(index=False))
        print(f"\n{len(new)} new pre-registration(s) {new} -> results/{PRED_CSV.name}  (written BEFORE training)")
        return
    m = score(args.datasets, args.band)
    m.to_csv(SCORED_CSV, index=False)
    print(f"{len(m):3d} rows -> results/{SCORED_CSV.name}")
    report(m, args.band)


# Defines command-line options: predict before training, score after.
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Module 7: pre-register and then score the frozen strategy rule on unseen datasets.")
    p.add_argument('--step', default='predict', choices=['predict', 'score'], help='predict = freeze the call BEFORE training; score = after. Default predict.')
    p.add_argument('--datasets', nargs='+', default=fr.STRATEGY_HELDOUT, help='Unseen datasets. Default frozen_rules.STRATEGY_HELDOUT.')
    p.add_argument('--band', default='sem', choices=BANDS, help='Tie band for the winner set. Default sem - the band the rule was frozen under.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
