'''Resolve degree-corpus ties the cheap way: score the CACHED embeddings per seed, then compare signals PAIRED on the same split.'''
# Wave 1 and 2 of DEGREE_RULE_CORPUS produced eight graphs whose top two signals sit inside the tie band, several under
# 0.1 sem apart. More seeds cannot fix that: closing a 0.08-sem gap to 2 sem needs ~3750 seeds, because the band shrinks
# as 1/sqrt(n). But the band itself is the wrong instrument. scoreboard.csv stores each variant's mean and std as if the
# variants were measured independently, when in fact every variant is scored on the SAME 70:30 split for a given seed -
# so the split-to-split variance, which dominates, is shared and cancels in a per-seed difference.
# This script recovers what the scoreboard throws away: it re-scores the embeddings already on disk (no training), keeps
# the per-seed values, and compares two signals by the standard error of their DIFFERENCE across seeds.
# Status, stated so it cannot be misread: the paired band is a SUPPLEMENTARY instrument for this corpus only. Every
# published cell, including the centrality rule's evidence, uses the frozen independent-means band, so a cell resolved
# here is reported as "resolved paired", never silently merged into the panel's labels.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from experiments.run_core import embed, score, tag
from experiments.strategy_select import SIGNAL

PER_SEED_CSV = cfg.RESULTS_DIR / "degree_rule_corpus_perseed.csv"
PAIRED_CSV = cfg.RESULTS_DIR / "degree_rule_corpus_paired.csv"
VERDICT_CSV = cfg.RESULTS_DIR / "degree_rule_corpus_verdicts.csv"
TASK = "link_prediction"

# Same threshold the frozen band uses, so only the ESTIMATE of the noise changes, never how much of it counts as a win.
BAND = 1.0


def per_seed(datasets, k, seeds, encoder="graphsage_edge"):
    '''One row per dataset x variant x seed, scored from the embedding already on disk; never trains.'''
    rows = []
    for ds in datasets:
        for sim in cfg.VG_SIMS:
            for seed in seeds:
                emb = cfg.NB3_DIR / TASK / ds / f"k{k}" / sim / f"{tag(encoder, 'all')}_s{seed}.emb"
                if not emb.exists():                       # a seed that was never run for this variant: report, do not train
                    continue
                rows.append({"dataset": ds, "graph_variant": sim, "signal": SIGNAL[sim], "seed": seed,
                             "auc": round(float(score(ds, emb, TASK, seed)), 6)})
    return pd.DataFrame(rows)


def paired(df):
    '''Per dataset: the best signal, and every rival compared PAIRED on the shared seeds - mean gap, its sem, and the verdict.'''
    rows = []
    for ds, g in df.groupby("dataset"):
        # A signal's score is its best variant's, exactly as winners() reads it; hybrids collapse onto the signal they add.
        best_var = g.groupby(["signal", "graph_variant"])["auc"].mean().reset_index()
        pick = {s: v.loc[v["auc"].idxmax(), "graph_variant"] for s, v in best_var.groupby("signal")}
        wide = g[[v in pick.values() for v in g["graph_variant"]]].pivot_table(index="seed", columns="signal", values="auc")
        top = wide.mean().idxmax()
        for sig in wide.columns:
            if sig == top:
                continue
            d = (wide[top] - wide[sig]).dropna()
            sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            # The same two numbers under the FROZEN band, so the two instruments can be read side by side.
            a, b = g[g["graph_variant"] == pick[top]]["auc"], g[g["graph_variant"] == pick[sig]]["auc"]
            ind = float(np.sqrt(a.std(ddof=1) ** 2 / len(a) + b.std(ddof=1) ** 2 / len(b)))
            rows.append({"dataset": ds, "top_signal": top, "top_variant": pick[top], "rival": sig,
                         "rival_variant": pick[sig], "n_seeds": len(d),
                         "gap": round(float(d.mean()), 6), "paired_sem": round(sem, 6),
                         "paired_ratio": round(float(d.mean() / sem), 2) if sem and sem == sem else float("nan"),
                         "independent_sem": round(ind, 6),
                         "independent_ratio": round(float(d.mean() / ind), 2) if ind else float("nan"),
                         "separated_paired": bool(sem == sem and d.mean() > BAND * sem),
                         "separated_frozen": bool(ind and d.mean() > BAND * ind)})
    return pd.DataFrame(rows).sort_values(["dataset", "rival"]).reset_index(drop=True)


def verdicts(pr):
    '''Collapse the pairwise rows to one line per dataset: which signals each band leaves standing.'''
    rows = []
    for ds, g in pr.groupby("dataset"):
        top = g["top_signal"].iloc[0]
        for band, col in [("frozen", "separated_frozen"), ("paired", "separated_paired")]:
            tied = sorted(set(g.loc[~g[col], "rival"]) | {top})
            rows.append({"dataset": ds, "band": band, "winner_signals": "|".join(tied),
                         "decided": len(tied) == 1, "signal": top if len(tied) == 1 else "",
                         "beats_original": "original" not in tied})
    return pd.DataFrame(rows)


def report(ps, pr, vd):
    '''Print the paired comparison and what it resolves that the frozen band could not.'''
    print(f"\n{len(ps)} per-seed scores over {ps['dataset'].nunique()} datasets "
          f"(seeds per cell: {sorted(ps.groupby('dataset')['seed'].nunique().unique())})")
    print("\nPAIRED vs FROZEN BAND - same gap, two estimates of its noise")
    print(pr[["dataset", "top_signal", "rival", "n_seeds", "gap", "paired_sem", "paired_ratio",
              "independent_sem", "independent_ratio", "separated_paired", "separated_frozen"]].to_string(index=False))
    w = vd.pivot(index="dataset", columns="band", values="winner_signals")
    w["resolved_by_pairing"] = w["paired"] != w["frozen"]
    print("\nWHAT EACH BAND LEAVES STANDING")
    print(w.to_string())
    n = int(w["resolved_by_pairing"].sum())
    print(f"\npairing narrows {n}/{len(w)} datasets; single-signal calls: frozen "
          f"{int(vd[(vd.band == 'frozen')]['decided'].sum())}, paired {int(vd[(vd.band == 'paired')]['decided'].sum())}")
    print("The paired band is SUPPLEMENTARY: the panel and every frozen rule use the independent-means band.")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ps = per_seed(args.datasets, args.k, args.seeds)
    assert not ps.empty, "no cached embeddings found for these datasets/seeds - run run_core.py first (this script never trains)"
    pr = paired(ps)
    vd = verdicts(pr)
    # Merged, not overwritten: the corpus and the panel are re-scored in separate invocations and both belong in the file.
    for name, df, key in [(PER_SEED_CSV, ps, ["dataset", "graph_variant", "seed"]), (PAIRED_CSV, pr, ["dataset", "rival"]),
                          (VERDICT_CSV, vd, ["dataset", "band"])]:
        if name.exists():
            old = pd.read_csv(name)
            df = pd.concat([old[~old["dataset"].isin(df["dataset"])], df], ignore_index=True).sort_values(key)
        df.to_csv(name, index=False)
        print(f"{len(df):4d} rows -> results/{name.name}")
    report(ps, pr, vd)


# Defines command-line options (mirrors run_core.py / degree_rule.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Score cached embeddings per seed and break degree-corpus ties with a paired comparison.")
    p.add_argument('--datasets', nargs='+', default=cfg.DEGREE_RULE_CORPUS, help='Datasets to re-score. Default: the whole degree corpus.')
    p.add_argument('--k', type=int, default=10, help='Top-K the virtual graphs were built with. Default 10 (locked).')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46, 47], help='Seeds to score; missing embeddings are skipped, never trained.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
