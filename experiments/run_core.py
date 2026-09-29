'''Run the non-OGB study end-to-end under the LOCKED config: reuse the Phase-2 virtual graphs, train the encoders, score NC + LP, record.'''
# Headless equivalent of notebook 3 §8 (+ the notebook-2 bridge cells) for the non-OGB datasets: same reuse-or-create
# discipline, same output zones, same scoreboard rows. Exists so the whole study runs from two scripts
# (this + run_ogb.py) under ONE frozen pipeline instead of by hand in a notebook.
# roman_empire + tolokers joined this list on 2026-07-27 with the pipeline unchanged - retuning on new datasets
# would void the held-out prediction they were added to test.
# --train-only writes embeddings and records nothing, so several datasets/variants can train in parallel without
# racing on results/scoreboard.csv; a later plain run reuses them and does all the scoring in one process.

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo.eval import results_io
from virgo.eval import nodeclass as eval_nodeclass
from virgo.eval import linkpred as eval_linkpred
from virgo.data import prepare_linkpred
from virgo.virtual_graph import VirtualGraph
from virgo.encoders import ENCODERS, feature_cache
from virgo.encoders.walk import DeepWalkModel
from experiments import run_ogb                                # for ensure_virtual only: the node-classification virtual graph is built the SAME way in both drivers

CORE = ["cora", "citeseer_linqs", "enzymes", "proteins", "roman_empire", "tolokers", "questions"]   # the non-OGB study datasets, all on ONE protocol (OGB pair lives in run_ogb.py)
# Module-3 held-out (2026-08-02), runnable but NOT in the default sweep: the pipeline is unchanged, exactly as when
# roman_empire/tolokers joined - retuning anything on these would void the prediction frozen before they were trained.
HELDOUT = ["pubmed", "actor", "minesweeper", "amazon_photo", "lastfm_asia", "amazon_ratings", "squirrel_filtered"]
# Module-7 held-out (2026-08-14): same rule again - the pipeline is unchanged, and the strategy prediction is frozen
# before any of these trains, so nothing here may be tuned on them.
STRATEGY_HELDOUT = ["reed98", "amherst41", "johnshopkins55", "cornell5"]
# The 2026-09-01 degree-evidence batch, same discipline again: the pipeline is unchanged, and the fit/validation split
# among them was pre-registered on graph properties BEFORE any of them was trained (docs/paper_log.md).
DEGREE_BATCH = ["twitch_de", "deezer_europe", "wisconsin", "cornell_webkb", "chameleon", "chameleon_filtered", "texas"]
# The 2026-09-03 degree VALIDATION set: same discipline once more - the pipeline is unchanged, and the candidate rule's
# prediction for each graph is written to results/degree_validation_prereg.csv before any of them is trained.
# The 2026-09-07 FAMILY validation set (Module 14): same discipline once more - the pipeline is unchanged, and the
# already-fixed cut's prediction for each graph is on disk in results/degree_family_prereg.csv before any of them trains.
RUNNABLE = (CORE + HELDOUT + STRATEGY_HELDOUT + DEGREE_BATCH + cfg.DEGREE_RULE_CORPUS
            + cfg.DEGREE_RULE_VALIDATION + cfg.DEGREE_TEST + cfg.DEGREE_BATCH2 + cfg.DEGREE_FAMILY_TEST
            + cfg.DEGREE_FAMILY_TEST2)
TASKS = {"node_classification": "node classification (weighted F1)", "link_prediction": "link prediction (AUC)"}


# Ablation-D arms write their own files and their own scoreboard rows; the locked D0 ("all") keeps the plain encoder name.
def tag(encoder, feats):
    '''Encoder id used for both the .emb filename and the scoreboard row (same convention as encoder.main).'''
    return encoder if feats == "all" else f"{encoder}_feat_{feats}"


# The shared 70/30 link-prediction split for one (dataset, seed): one split per seed, reused by every encoder and variant.
def split(ds, seed):
    '''Path to the seed's LP split dir; builds it if missing (deterministic -> rebuilding gives the same split).'''
    d = cfg.LP_SPLITS_VG / ds / f"seed_{seed}"
    if not (d / "test_pos.txt").exists():
        prepare_linkpred.prepare(cfg.dataset(ds)["edgelist"], d, test_frac=cfg.REPRO["linkpred_test_frac"], seed=seed)
    return d


# Trains (or reuses) one embedding: NC embeds the FULL graph, LP embeds the seed's 70% train graph (virtual graph rebuilt from it).
def embed(ds, k, sim, encoder, seed, task, feats="all"):
    '''Train one encoder over the virtual graph for one task -> .emb path; reuse if already on disk.'''
    zone = cfg.NB3_DIR if encoder in ENCODERS else cfg.NB2_DIR
    emb = zone / task / ds / f"k{k}" / sim / f"{tag(encoder, feats)}_s{seed}.emb"
    if emb.exists():
        print(f"reuse  {ds} {task} {sim} {encoder} s{seed}", flush=True)
        return emb
    if task == "node_classification":
        edgelist = Path(cfg.dataset(ds)["edgelist"])
        G = graph_io.load_graph(edgelist)
        V = run_ogb.ensure_virtual(G, ds, k, sim)              # reuse the notebook-2 build when it exists, else build it here + log a health row
    else:
        edgelist = split(ds, seed) / "train.edgelist"
        G = graph_io.load_graph(edgelist)
        V = VirtualGraph(G, seed=cfg.REPRO["seed"]).build(sim, k)   # train edges only; VG seed = project seed, NOT the loop seed
    print(f"TRAIN  {ds} {task} {sim} {encoder} s{seed} | V: {V.number_of_nodes()} nodes / {V.number_of_edges()} edges", flush=True)
    emb.parent.mkdir(parents=True, exist_ok=True)
    if encoder in ENCODERS:                                    # any registered GNN encoder; adding one needs no edit here
        p = cfg.GNN_PARAMS
        enc = ENCODERS[encoder](G, V, seed, hidden=p["hidden"], dimensions=p["dimensions"], layers=p["layers"],
                                agg=p["agg"], feats=feats, cache=feature_cache(edgelist))
        enc.train(p["epochs"], lr=p["lr"], negatives=p["negatives"],
                  pairs_per_epoch=p["pairs_per_epoch"], max_pairs=p["max_pairs"], positives=p["positives"])
        enc.save(emb)
    else:                                                      # deepwalk bridge: plain unweighted copy of V (same as notebook 2)
        flat = tempfile.NamedTemporaryFile(suffix=".edgelist", delete=False).name
        nx.write_edgelist(V, flat, data=False)
        DeepWalkModel().train(flat, str(emb), seed=seed, params=cfg.I2V_PARAMS)
        os.unlink(flat)
    return emb


# Scores one embedding with the shared protocol (identical to notebooks 2-3): weighted F1, or held-out AUC.
def score(ds, emb, task, seed):
    '''Node classification weighted F1, or leakage-free link-prediction AUC, for one embedding.'''
    if task == "node_classification":
        f1s, _, _ = eval_nodeclass.evaluate(str(emb), str(cfg.dataset(ds)["labels"]),
                                            train_frac=cfg.REPRO["nodeclass_train_frac"], seed=seed)
        return f1s["weighted"]
    return eval_linkpred.evaluate(str(emb), str(split(ds, seed)), seed=seed, score=cfg.REPRO["linkpred_score"])


# Runs the sweep: for every dataset x task x variant x encoder, train-or-reuse each seed, then record one scoreboard row.
def main(args):
    tasks = list(TASKS) if args.task == "all" else [args.task]
    encoders = ["graphsage_edge", "deepwalk"] if args.encoder == "all" else [args.encoder]   # "all" = the LOCKED pair; a new encoder is opt-in by name
    if args.features != "all":
        encoders = [e for e in encoders if e in ENCODERS]      # ablation D varies the GNN's inputs; the walk encoder has none
    sims = cfg.VG_SIMS if args.sim == "all" else [args.sim]
    for ds in args.datasets:
        for task in tasks:
            if task == "node_classification" and not cfg.dataset(ds)["labels"]:
                print(f"{ds}: no labels -> node classification skipped", flush=True)
                continue
            for sim in sims:
                for encoder in encoders:
                    vals = [score(ds, embed(ds, args.k, sim, encoder, seed, task, args.features), task, seed) if not args.train_only
                            else embed(ds, args.k, sim, encoder, seed, task, args.features) for seed in args.seeds]
                    if args.train_only:
                        continue
                    name = tag(encoder, args.features)
                    results_io.record_score(ds, name, sim, args.k, TASKS[task], args.seeds, vals)
                    print(f"score  {ds} {task} {sim} {name} | {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}", flush=True)


# Defines command-line options (mirrors run_ogb.py); every default is the locked setting.
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Run the core-4 study under the locked config (notebook-3 §8 headless).")
    p.add_argument('--datasets', nargs='+', default=CORE, choices=RUNNABLE,
                   help='Datasets to run. Default: the seven study datasets; the Module-3 held-out graphs are opt-in by name.')
    p.add_argument('--task', default='all', choices=['all'] + list(TASKS), help='Which task(s) to run. Default all.')
    p.add_argument('--encoder', default='all', choices=['all', 'deepwalk'] + list(ENCODERS),
                   help='Which encoder(s). "all" = the locked graphsage_edge + deepwalk pair; name any registered encoder to opt it in. Default all.')
    p.add_argument('--sim', default='all', choices=['all'] + cfg.VG_SIMS,
                   help='Which virtual-graph variant(s). Default all = the seven official variants in cfg.VG_SIMS.')
    p.add_argument('--features', default='all', choices=[f for f in cfg.D_FEATURES if f != 'none_mp'],
                   help='Ablation D input features (D6 none_mp is layers=0 and has no training -> not run here). Default all = the locked D0.')
    p.add_argument('--k', type=int, default=10, help='Top-K of the virtual graph. Default 10.')
    p.add_argument('--seeds', type=int, nargs='+', default=cfg.VG_SEEDS, help='Encoder seeds. Default 42 43 44.')
    p.add_argument('--train-only', action='store_true',
                   help='Write embeddings and record nothing (lets several runs train in parallel without racing on the scoreboard).')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
