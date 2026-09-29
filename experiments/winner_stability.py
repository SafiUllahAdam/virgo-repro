'''How much of a winning-signal label is seed noise? Resample 3-seed subsets from a deeper seed pool and count the winners.'''
# Every stage-2 cell in this study is labelled from THREE seeds (paper_log 12). Module 12's second batch found two
# non-independent pairs that disagree - homologous ppi_rat/ppi_mouse name psi/degree, same-population
# foursquare_checkin/foursquare_tips name tie/degree (paper_log 22) - which raises a question no property screen can
# answer: how reproducible is a 3-seed label on ONE graph?
# This script answers it directly. It scores a deeper pool of seeds, then enumerates every 3-seed subset of that pool,
# recomputes the paired-band winner for each subset, and reports the distribution. It trains nothing and fits nothing.
#   flip_rate   fraction of 3-seed subsets whose winner differs from the full-pool winner
#   modal_frac  how often the most common 3-seed winner appears - 1.0 means the label is determined, 0.33 means a coin
#   published   the winner from seeds 42/43/44, i.e. the label the scoreboard actually carries

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from experiments.tie_break import BAND, per_seed

PUBLISHED = [42, 43, 44]        # the seeds every published cell was labelled from


# Same rule tie_break.verdicts applies, on one subset of seeds. The best variant per signal is chosen INSIDE the subset,
# not from the full pool: picking it globally would leak the answer the resampling is meant to measure.
def winner(g):
    '''The signal set a paired comparison leaves standing on these seeds, as a "a|b" string.'''
    best = g.groupby(["signal", "graph_variant"])["auc"].mean().reset_index()
    pick = {s: v.loc[v["auc"].idxmax(), "graph_variant"] for s, v in best.groupby("signal")}
    wide = g[[v in pick.values() for v in g["graph_variant"]]].pivot_table(index="seed", columns="signal", values="auc")
    top = wide.mean().idxmax()
    stand = [top]
    for sig in wide.columns:
        if sig == top:
            continue
        d = (wide[top] - wide[sig]).dropna()
        sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        if not (sem == sem and d.mean() > BAND * sem):        # not separated -> it stands with the top
            stand.append(sig)
    return "|".join(sorted(stand))


def stability(df, k):
    '''One row per dataset: the full-pool winner, the published 3-seed winner, and how 3-seed subsets are distributed.'''
    rows = []
    for ds, g in df.groupby("dataset"):
        seeds = sorted(g["seed"].unique())
        if len(seeds) < k:
            continue
        subs = [winner(g[g["seed"].isin(c)]) for c in combinations(seeds, k)]
        counts = Counter(subs)
        full = winner(g)
        pub = winner(g[g["seed"].isin(PUBLISHED)]) if len(set(PUBLISHED) & set(seeds)) == k else ""
        rows.append({"dataset": ds, "n_seeds": len(seeds), "n_subsets": len(subs),
                     "full_pool_winner": full, "published_3seed_winner": pub,
                     "modal_3seed_winner": counts.most_common(1)[0][0],
                     "modal_frac": round(counts.most_common(1)[0][1] / len(subs), 4),
                     "flip_rate": round(sum(w != full for w in subs) / len(subs), 4),
                     "distinct_winners": len(counts),
                     "published_matches_full": bool(pub == full) if pub else None,
                     "winner_distribution": " ".join(f"{w}:{c}" for w, c in counts.most_common(5))})
    return pd.DataFrame(rows).sort_values("flip_rate", ascending=False).reset_index(drop=True)


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ps = per_seed(args.datasets, args.k, args.seeds)
    assert not ps.empty, "no embeddings found on disk for those datasets/seeds - train them first"
    out = stability(ps, args.subset)
    out.to_csv(cfg.RESULTS_DIR / "winner_stability.csv", index=False)
    print(f"\nWINNER STABILITY - every {args.subset}-seed subset of the pool, paired band")
    print(out[["dataset", "n_seeds", "n_subsets", "full_pool_winner", "published_3seed_winner",
               "modal_3seed_winner", "modal_frac", "flip_rate", "distinct_winners"]].to_string(index=False))
    print("\nWINNER DISTRIBUTION per dataset")
    for _, r in out.iterrows():
        print(f"  {r['dataset']:22s} {r['winner_distribution']}")
    ok = out["published_matches_full"].dropna()
    print(f"\nMEAN flip rate {out['flip_rate'].mean():.3f}  |  mean modal fraction {out['modal_frac'].mean():.3f}  |  "
          f"published 3-seed label matches the full pool on {int(ok.sum())}/{len(ok)} datasets")
    print(f"\n{len(out):3d} rows -> results/winner_stability.csv")


# Defines command-line options (mirrors tie_break.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure how reproducible a k-seed winning-signal label is, by resampling seed subsets.")
    p.add_argument('--datasets', nargs='+', required=True, help='Datasets to measure; their embeddings must already be on disk.')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)), help='The seed POOL to resample from. Default 42-51.')
    p.add_argument('--subset', type=int, default=3, help='Subset size to resample. Default 3, the size every published cell used.')
    p.add_argument('--k', type=int, default=10, help='Top-K the virtual graphs were built with. Default 10 (locked).')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
