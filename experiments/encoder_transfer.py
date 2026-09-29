'''Module 15: does the framework's CALL survive a change of encoder? Train a second encoder on the frozen graphs, compare verdicts.'''
# GraphSAGE stays the study's encoder - every frozen rule was fitted under it and nothing here re-fits or re-scores one.
# The question is narrower and is the only one asked: on the graphs where stage 1 and stage 2 DECIDE something, does a
# different aggregator keep the same decision? So three things are held byte-identical to the GraphSAGE arm - the graph,
# the structural features (same content-hashed cache) and the 70:30 split (same per dataset x seed) - and only the
# convolution changes. Nothing GraphSAGE wrote is read for writing: embeddings carry the encoder in their filename and
# scoreboard rows are keyed on the encoder column, so the published arm cannot be touched.
#   --step run      train (or reuse) the new encoder over the seven official variants, score, record
#   --step compare  read both arms per seed, compare them PAIRED on the shared split, and report where the call moves
# The paired band is the right instrument here for the reason tie_break.py gives: both encoders are scored on the SAME
# split per seed, so the split-to-split variance that dominates cancels in a per-seed difference. The baseline arm is
# never trained by this script - a missing GraphSAGE embedding is reported, not silently recreated.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo.eval import results_io
from virgo.encoders import ENCODERS
from experiments.run_core import embed, score, tag
from experiments.strategy_select import SIGNAL
from experiments.tie_break import BAND, paired, verdicts

PER_SEED_CSV = cfg.RESULTS_DIR / "encoder_transfer_perseed.csv"
FOURCELL_CSV = cfg.RESULTS_DIR / "encoder_transfer_fourcell.csv"
VERDICT_CSV = cfg.RESULTS_DIR / "encoder_transfer_verdicts.csv"
AGREEMENT_CSV = cfg.RESULTS_DIR / "encoder_transfer_agreement.csv"
TASK = "link_prediction"
BASELINE = "graphsage_edge"                                       # the frozen arm: read only, never trained here


# One row per dataset x variant x seed. train=False scores what is already on disk, which is how the frozen arm is read.
def per_seed(datasets, encoder, seeds, k, train, sims=None):
    '''Per-seed AUC for one encoder over the seven official variants; trains only when train=True.'''
    rows = []
    for ds in datasets:
        for sim in (sims or cfg.VG_SIMS_OFFICIAL):
            for seed in seeds:
                emb = cfg.NB3_DIR / TASK / ds / f"k{k}" / sim / f"{tag(encoder, 'all')}_s{seed}.emb"
                if not emb.exists() and not train:
                    print(f"MISSING {ds} {sim} {encoder} s{seed} - reported, not trained", flush=True)
                    continue
                emb = embed(ds, k, sim, encoder, seed, TASK) if train else emb
                rows.append({"dataset": ds, "encoder": encoder, "graph_variant": sim, "signal": SIGNAL[sim],
                             "seed": seed, "auc": round(float(score(ds, emb, TASK, seed)), 6)})
    return pd.DataFrame(rows)


# The professor's four cells: <encoder> alone vs <encoder>+augmentation, paired on the shared split so the gap is testable.
def four_cell(ps):
    '''Per dataset x encoder: the original graph, the best augmented graph, and their paired difference.'''
    rows = []
    for (ds, enc), g in ps.groupby(["dataset", "encoder"]):
        wide = g.pivot_table(index="seed", columns="graph_variant", values="auc")
        if "original" not in wide:
            continue
        aug = wide.drop(columns=["original"]).mean().idxmax()
        d = (wide[aug] - wide["original"]).dropna()
        sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        rows.append({"dataset": ds, "encoder": enc, "n_seeds": len(d),
                     "original": round(float(wide["original"].mean()), 4),
                     "best_aug_variant": aug, "best_aug_signal": SIGNAL[aug],
                     "best_aug": round(float(wide[aug].mean()), 4), "gap": round(float(d.mean()), 4),
                     "paired_sem": round(sem, 6) if sem == sem else float("nan"),
                     "paired_ratio": round(float(d.mean() / sem), 2) if sem and sem == sem else float("nan"),
                     "aug_wins": bool(sem == sem and d.mean() > BAND * sem)})
    return pd.DataFrame(rows).sort_values(["dataset", "encoder"]).reset_index(drop=True)


# The cross-encoder cell the objection turns on: is the new encoder ALONE already as good as GraphSAGE WITH augmentation?
def cross(ps, encoder, baseline):
    '''Per dataset: <encoder> on the original graph vs <baseline> on its best augmented graph, paired on the shared split.'''
    rows = []
    for ds, g in ps.groupby("dataset"):
        a = g[(g.encoder == encoder) & (g.graph_variant == "original")].set_index("seed")["auc"]
        b = g[g.encoder == baseline]
        if a.empty or b.empty or "original" not in set(b["graph_variant"]):
            continue
        wide = b.pivot_table(index="seed", columns="graph_variant", values="auc")
        aug = wide.drop(columns=["original"]).mean().idxmax()
        d = (a - wide[aug]).dropna()
        sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        rows.append({"dataset": ds, "n_seeds": len(d), f"{encoder}_original": round(float(a.mean()), 4),
                     f"{baseline}_aug_variant": aug, f"{baseline}_aug": round(float(wide[aug].mean()), 4),
                     "gap": round(float(d.mean()), 4), "paired_sem": round(sem, 6) if sem == sem else float("nan"),
                     "paired_ratio": round(float(d.mean() / sem), 2) if sem and sem == sem else float("nan"),
                     "new_encoder_alone_suffices": bool(sem == sem and d.mean() > BAND * sem)})
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


# Stage 1 = augment or keep; stage 2 = which signal. Both read off the SAME verdict rows, so a call cannot drift between them.
def agreement(vd, encoder, baseline, band="paired"):
    '''Per dataset: each arm's stage-1 and stage-2 call, and whether the two encoders agree.'''
    v = vd[vd.band == band].set_index(["encoder", "dataset"])
    rows = []
    for ds in sorted({d for _, d in v.index}):
        if (encoder, ds) not in v.index or (baseline, ds) not in v.index:
            continue
        a, b = v.loc[(baseline, ds)], v.loc[(encoder, ds)]
        rows.append({"dataset": ds,
                     "stage1_baseline": "augment" if a["beats_original"] else "keep original",
                     "stage1_encoder": "augment" if b["beats_original"] else "keep original",
                     "stage2_baseline": a["winner_signals"], "stage2_encoder": b["winner_signals"],
                     "baseline_decided": bool(a["decided"]), "encoder_decided": bool(b["decided"]),
                     "stage1_agree": bool(a["beats_original"] == b["beats_original"]),
                     # Only a cell BOTH arms resolve to one signal can agree or disagree; the rest are undecided, not errors.
                     "stage2_comparable": bool(a["decided"] and b["decided"]),
                     "stage2_agree": bool(a["decided"] and b["decided"] and a["signal"] == b["signal"])})
    return pd.DataFrame(rows)


# Merged, never overwritten: panels are run in separate invocations and every one of them belongs in the file.
def save(path, df, key):
    '''Upsert df into path on `key`, keeping rows from datasets this run did not touch.'''
    if path.exists():
        old = pd.read_csv(path)
        drop = old[key].apply(tuple, axis=1).isin(df[key].apply(tuple, axis=1)) if not df.empty else False
        df = pd.concat([old[~drop], df], ignore_index=True)
    df.sort_values(key).to_csv(path, index=False)
    print(f"{len(df):4d} rows -> results/{path.name}")
    return df


def report(fc, cr, ag, encoder, baseline):
    '''Print the four-cell table, the cross-encoder cell, and where the framework's call moves.'''
    print("\nFOUR CELLS - each encoder on the original graph vs its best augmented graph (paired on the shared split)")
    print(fc.to_string(index=False))
    print(f"\nTHE OBJECTION - {encoder} ALONE vs {baseline} WITH augmentation")
    print(cr.to_string(index=False) if not cr.empty else "  (no comparable cells)")
    print("\nDOES THE CALL SURVIVE THE ENCODER?")
    print(ag.to_string(index=False) if not ag.empty else "  (no comparable cells)")
    if not ag.empty:
        n2 = int(ag["stage2_comparable"].sum())
        print(f"\nstage 1 agrees {int(ag['stage1_agree'].sum())}/{len(ag)}   "
              f"stage 2 agrees {int(ag['stage2_agree'].sum())}/{n2} of the {n2} cells both arms decide")
    for enc, g in fc.groupby("encoder"):
        print(f"augmentation beats the original under {enc}: {int(g['aug_wins'].sum())}/{len(g)} at {BAND} paired sem")
    print(f"\n{baseline} is the frozen arm: read, never retrained. No published row is touched by this script.")


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ps = pd.DataFrame()
    if args.step in ("run", "all"):
        ps = per_seed(args.datasets, args.encoder, args.seeds, args.k, True, args.sims)
        assert not ps.empty, f"nothing scored for {args.encoder} - check the dataset names"
        for (ds, sim), g in ps.groupby(["dataset", "graph_variant"]):
            results_io.record_score(ds, args.encoder, sim, args.k, "link prediction (AUC)",
                                    sorted(g["seed"]), g["auc"].tolist())
    if args.step in ("compare", "all"):
        base = per_seed(args.datasets, args.baseline, args.seeds, args.k, False)
        ps = pd.concat([ps if not ps.empty else per_seed(args.datasets, args.encoder, args.seeds, args.k, False), base],
                       ignore_index=True)
        assert not ps.empty, "no per-seed scores to compare - run --step run first"
        ps = save(PER_SEED_CSV, ps, ["dataset", "encoder", "graph_variant", "seed"])
        ps = ps[ps["dataset"].isin(args.datasets)]
        vd = pd.concat([verdicts(paired(g)).assign(encoder=enc) for enc, g in ps.groupby("encoder")], ignore_index=True)
        fc, cr = four_cell(ps), cross(ps, args.encoder, args.baseline)
        ag = agreement(vd, args.encoder, args.baseline)
        save(VERDICT_CSV, vd, ["dataset", "encoder", "band"])
        save(FOURCELL_CSV, fc, ["dataset", "encoder"])
        save(AGREEMENT_CSV, ag, ["dataset"])
        report(fc, cr, ag, args.encoder, args.baseline)


# Defines command-line options (mirrors run_core.py / tie_break.py); every default is the locked setting.
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Train a second encoder on the frozen graphs and test whether the framework's calls survive.")
    p.add_argument('--step', default='all', choices=['run', 'compare', 'all'], help='run=train+score+record, compare=read both arms and report. Default all.')
    p.add_argument('--encoder', default='gatv2_edge', choices=[e for e in ENCODERS if e != BASELINE],
                   help='The encoder under test. Default gatv2_edge.')
    p.add_argument('--baseline', default=BASELINE, choices=list(ENCODERS), help='The frozen arm, read only. Default graphsage_edge.')
    p.add_argument('--datasets', nargs='+', default=cfg.ENCODER_SMOKE,
                   help='Default: the smoke four. cfg.ENCODER_PANEL is the full stage-1 + stage-2 panel.')
    p.add_argument('--k', type=int, default=10, help='Top-K of the role graph. Default 10 (locked).')
    p.add_argument('--seeds', type=int, nargs='+', default=cfg.VG_SEEDS, help='Encoder seeds. Default 42 43 44.')
    p.add_argument('--sims', nargs='+', default=None, choices=cfg.VG_SIMS_OFFICIAL,
                   help='Train only these variants (chunks a large graph across invocations). Default: all seven. --step compare ignores it and reads whatever is on disk.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
