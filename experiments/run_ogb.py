'''Run the OGB study end-to-end under the OFFICIAL protocol: build virtual graphs, train encoders, score VALIDATION; --final scores TEST once.'''
# Step-6 discipline, enforced by workflow: embeddings train on the official TRAIN split only; every sweep scores
# the VALIDATION split and records it; --final is the single test read, run AFTER choices are locked, and it
# refuses to run while the scoreboard holds no validation rows for the dataset. Test never drives selection.
# Reuse-or-create throughout: existing virtual graphs and .emb files are reused, so --final retrains nothing.

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo.eval import results_io
from virgo.eval import ogb as eval_ogb
from virgo.data import make_ogb
from virgo.virtual_graph import VirtualGraph
from virgo.encoders import ENCODERS, feature_cache
from virgo.encoders.walk import DeepWalkModel

# Dataset -> (output task folder, scoreboard task string). Metric names carry the split ("valid_acc"), so rows never collide.
TASKS = {"ogbn_arxiv": ("node_classification", "node classification (OGB official)"),
         "ogbl_ddi":   ("link_prediction",     "link prediction (OGB official)")}

# Display-only names: the scoreboard keeps the raw ids, the printed tables read like the paper.
LABELS = {"psi": "Ψ", "degree": "Degree", "centrality": "Centrality", "original": "Original", "hybrid": "Hybrid",
          "hybrid_degree": "Hybrid+Deg", "hybrid_centrality": "Hybrid+Cent",
          "graphsage_edge": "GraphSAGE", "deepwalk": "DeepWalk"}
UNIT = {"acc": "Accuracy", "hits@20": "Hits@20"}              # primary-metric suffix -> display name
PRIMARY = ["acc", "hits@20"]                                  # secondaries (weighted/macro F1) stay out of the main table


# Builds (once) and loads the virtual graph for one variant; logs a health row on build, restores the full node set on load.
def ensure_virtual(G, ds, k, sim):
    '''Build+save the virtual graph if missing (with a graph_health row), else load it; full node set either way.'''
    vg = cfg.NB2_DIR / "virtual_graphs" / ds / f"k{k}" / sim / "virtual_graph.edgelist"
    if not vg.exists():
        print(f"build  {sim} virtual graph ...", flush=True)
        V = VirtualGraph(G, seed=cfg.REPRO["seed"]).build(sim, k)
        vg.parent.mkdir(parents=True, exist_ok=True)
        nx.write_weighted_edgelist(V, vg)
        row = {"dataset": ds, "sim": sim, "K": k, "nodes": V.number_of_nodes(), "edges": V.number_of_edges(),
               "avg_degree": round(2 * V.number_of_edges() / max(V.number_of_nodes(), 1), 2),
               "max_degree": max(d for _, d in V.degree), "components": nx.number_connected_components(V),
               "isolates": sum(1 for _, d in V.degree if d == 0), "path": str(vg)}
        pd.DataFrame([row]).to_csv(cfg.GRAPH_HEALTH_CSV, mode="a", header=not cfg.GRAPH_HEALTH_CSV.exists(), index=False)
        return V
    V = nx.read_weighted_edgelist(vg, nodetype=int)
    V.add_nodes_from(G.nodes)                                  # edgelists omit isolated nodes -> restore the full node set
    return V


# Ablation-D arms write their own files and their own scoreboard rows; the locked D0 ("all") keeps the plain encoder name.
def tag(encoder, feats):
    '''Encoder id used for both the .emb filename and the scoreboard row (mirrors run_core.tag; kept local - run_core imports this module).'''
    return encoder if feats == "all" else f"{encoder}_feat_{feats}"


# Trains (or reuses) one embedding on the official train graph; graphsage -> notebook-3 zone, deepwalk -> notebook-2 bridge zone.
def embed(G, V, ds, k, sim, encoder, seed, edgelist, feats="all"):
    '''Train one encoder over the virtual graph (official train edges only) -> .emb path; reuse if already on disk.'''
    zone = cfg.NB3_DIR if encoder in ENCODERS else cfg.NB2_DIR
    emb = zone / TASKS[ds][0] / ds / f"k{k}" / sim / f"{tag(encoder, feats)}_s{seed}.emb"
    if emb.exists():
        print(f"reuse  {sim} {tag(encoder, feats)} seed {seed} -> {emb.name}", flush=True)
        return emb
    print(f"TRAIN  {sim} {tag(encoder, feats)} seed {seed} | V: {V.number_of_nodes()} nodes / {V.number_of_edges()} edges", flush=True)
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


# Scores one embedding on ONE official split; returns {metric_name: value} with the split baked into the metric name.
def score(ds, emb, split, seed):
    '''Official OGB metric(s) for one embedding on one split: arxiv Accuracy (+F1 secondaries on test), ddi Hits@20.'''
    d = cfg.DATASETS[ds]
    if ds == "ogbn_arxiv":
        r = eval_ogb.evaluate_nodeclass(emb, str(d["labels"]), str(d["split"]), seed=seed, split=split)
        out = {f"{split}_acc": r["acc"]}
        if split == "test":                                    # secondary metrics reported on the official test nodes only
            out |= {"test_weighted_f1": r["weighted_f1"], "test_macro_f1": r["macro_f1"]}
        return out
    r = eval_ogb.evaluate_linkpred(emb, str(d["pairs"]), split=split, scorer="mlp",   # OGB-style decoder, TRAIN edges only
                                   train_edgelist=str(d["edgelist"]), seed=seed)
    return {f"{split}_hits@20": r["hits@20"]}


# Scoreboard cache: scoring is deterministic (seeded scorer + reused embeddings) -> once a config's rows exist, recomputing repeats identical numbers.
def cached(ds, enc, sim, k, split):
    '''Return {metric: mean} if this config's {split} primary metric is already on the scoreboard, else None (recompute).'''
    p = Path(cfg.SCOREBOARD_CSV)
    if not (p.exists() and p.stat().st_size):
        return None
    board = pd.read_csv(p)
    r = board[(board["dataset"] == ds) & (board["encoder"] == enc) & (board["graph_variant"] == sim)
              & (board["top_K_neighbors"] == k) & board["metric"].isin([f"{split}_{m}" for m in PRIMARY])]
    return dict(zip(r["metric"], r["mean"])) if not r.empty else None


# Pivots the scoreboard into a readable graph x encoder table for one split (the primary metric only).
def table(ds, split, better=False):
    '''Graph x encoder table of the primary OGB metric for one split; better=True adds the winning-encoder column.'''
    board = pd.read_csv(cfg.SCOREBOARD_CSV)
    rows = board[(board["dataset"] == ds) & board["metric"].isin([f"{split}_{m}" for m in PRIMARY])]
    assert not rows.empty, f"no {split} rows for {ds} - run the {split} sweep first"
    unit = UNIT[rows["metric"].iloc[0].split("_", 1)[1]]
    t = rows.pivot(index="graph_variant", columns="encoder", values="mean").reindex(cfg.VG_SIMS).dropna(how="all")
    t = t.reindex(columns=[c for c in ("graphsage_edge", "deepwalk") if c in t.columns]).round(4)
    t.columns = [f"{LABELS.get(c, c)} {'' if split == 'valid' else 'test '}{unit}" for c in t.columns]
    t.index = [LABELS.get(i, i) for i in t.index]      # a variant with no display label prints its raw name, never crashes
    if better and t.shape[1] > 1:
        win = t.idxmax(axis=1).str.split().str[0]              # each column header starts with its encoder label
        win[(t.nunique(axis=1) == 1) & t.notna().all(axis=1)] = "tie"   # identical scores -> no winner to name
        win[t.stack().idxmax()[0]] += " - best overall"        # the single best cell in the table
        t["Better encoder"] = win
    t.index.name = "Graph version" if split == "valid" else "Graph"
    return t


SELECTION_JSON = cfg.RESULTS_DIR / "ogb_selection.json"       # the lock: one entry per dataset, written by select(), read by selection()


# Picks the winner on VALIDATION and locks it to results/ogb_selection.json; frozen once any test row exists.
def select(ds):
    '''Pick + lock the validation winner; idempotent once locked (re-run just re-shows the lock), never re-picks after test.'''
    board = pd.read_csv(cfg.SCOREBOARD_CSV)
    rows = board[board["dataset"] == ds]
    sel = json.loads(SELECTION_JSON.read_text()) if SELECTION_JSON.exists() else {}
    if ds in sel:                                         # already locked -> re-running §6 is a no-op: show it, keep the same winner
        e = sel[ds]
        print(f"{ds} already locked ({e['locked_at']}): {LABELS[e['graph_variant']]} graph + {LABELS[e['encoder']]}, "
              f"{UNIT[e['metric'].split('_', 1)[1]]} = {e['mean']:.4f} ± {e['std']:.4f}   (unchanged -> {SELECTION_JSON})")
        return e
    assert rows[rows["metric"].isin([f"test_{m}" for m in PRIMARY])].empty, \
        f"{ds} has test rows but no saved lock - refusing to create a selection after test was read (would bias the result)"
    valid = rows[rows["metric"].isin(["valid_acc", "valid_hits@20"])]
    assert not valid.empty, f"no validation rows for {ds} - run the validation sweep first"
    unit = UNIT[valid["metric"].iloc[0].split("_", 1)[1]]
    show = valid.sort_values("mean", ascending=False)[["encoder", "graph_variant", "mean", "std"]].round(4).replace(LABELS)
    print(show.rename(columns={"encoder": "Encoder", "graph_variant": "Graph",
                               "mean": f"Validation {unit}", "std": "Std. deviation"}).to_string(index=False))
    best = valid.loc[valid["mean"].idxmax()]
    entry = {"encoder": best["encoder"], "graph_variant": best["graph_variant"],
             "top_K_neighbors": int(best["top_K_neighbors"]), "metric": best["metric"],
             "mean": float(best["mean"]), "std": float(best["std"]), "seeds": best["seeds"],
             "locked_at": datetime.now().isoformat(timespec="seconds")}
    sel[ds] = entry
    SELECTION_JSON.write_text(json.dumps(sel, indent=1) + "\n")
    print(f"\nValidation winner: {LABELS[best['graph_variant']]} graph + {LABELS[best['encoder']]}, "
          f"{unit} = {best['mean']:.4f} ± {best['std']:.4f}   (locked -> {SELECTION_JSON})")
    return entry


# Reads the locked choice for a dataset; the guard every test read must pass first.
def selection(ds):
    '''Return the locked validation choice; refuse if none was saved (select() has not run).'''
    sel = json.loads(SELECTION_JSON.read_text()) if SELECTION_JSON.exists() else {}
    assert ds in sel, f"no locked selection for {ds} in {SELECTION_JSON} - sweep on valid + select() first, then read test once"
    return sel[ds]


# Prints the final report: the LOCKED winner names the headline; the test table is shown, never re-selected from.
def report(ds):
    '''Locked winner + its test score (the headline), the valid/test tables, then the one-line conclusion.'''
    lock = selection(ds)
    unit = UNIT[lock["metric"].split("_", 1)[1]]
    board = pd.read_csv(cfg.SCOREBOARD_CSV)
    rows = board[board["dataset"] == ds]
    print(f"Validation winner (locked {lock['locked_at']}): {LABELS[lock['graph_variant']]} graph + "
          f"{LABELS[lock['encoder']]}, {unit} = {lock['mean']:.4f} ± {lock['std']:.4f}")
    print("\nValidation\n" + table(ds, "valid", better=True).to_string())
    t = table(ds, "test")
    print("\nTest\n" + t.to_string())
    sec = rows[rows["metric"].isin(["test_weighted_f1", "test_macro_f1"])]
    if not sec.empty:                                          # arxiv only: OGB-secondary F1 scores, reported not selected on
        s = (sec.pivot(index="graph_variant", columns=["encoder", "metric"], values="mean")
             .reindex(cfg.VG_SIMS).dropna(how="all").round(4))   # dropna: a variant not yet run on this dataset is absent, not an empty row
        print("\nTest (secondary F1)\n" + s.to_string())
    hl = rows[(rows["encoder"] == lock["encoder"]) & (rows["graph_variant"] == lock["graph_variant"])
              & rows["metric"].isin(["test_acc", "test_hits@20"])]
    if not hl.empty:
        h = hl.iloc[0]
        floor = t.max(numeric_only=True).max() < 0.05          # near-floor scores: report the number, but flag it as uninformative
        print(f"\nCONCLUSION: on {ds}, {LABELS[lock['graph_variant']]} + {LABELS[lock['encoder']]} remained the best "
              f"configuration, with test {unit} = {h['mean']:.4f} ± {h['std']:.4f}."
              + (" All scores are extremely low - structural-only embeddings with cosine scoring perform"
                 " poorly on this dataset." if floor else ""))


# Runs the sweep: ensure OGB files -> build graphs -> train -> score one split -> record -> print the selection table.
def main(args):
    ds = args.dataset
    split = "test" if args.final else "valid"
    if args.final:                                             # no-test-peeking guard: the winner must be LOCKED before the one test read
        selection(ds)
    make_ogb.ensure_ogb(ds)                                    # converter files (edgelist/.nodes/labels/splits) built if absent
    edgelist = str(cfg.DATASETS[ds]["edgelist"])
    G = graph_io.load_graph(edgelist)
    graph_io.check(G, ds)
    encoders = ["graphsage_edge", "deepwalk"] if args.encoder == "all" else [args.encoder]   # "all" = the LOCKED pair; a new encoder is opt-in by name
    if args.features != "all":
        encoders = [e for e in encoders if e in ENCODERS]      # ablation D varies the GNN's inputs; the walk encoder has none
    sims = cfg.VG_SIMS if args.sim == "all" else [args.sim]
    for sim in sims:
        V = ensure_virtual(G, ds, args.k, sim)
        for encoder in encoders:
            per_metric = {}
            for seed in args.seeds:
                emb = embed(G, V, ds, args.k, sim, encoder, seed, edgelist, args.features)
                for m, v in score(ds, str(emb), split, seed).items():
                    per_metric.setdefault(m, []).append(v)
            name = tag(encoder, args.features)
            for m, vals in per_metric.items():
                results_io.record_score(ds, name, sim, args.k, TASKS[ds][1], args.seeds, vals, metric=m)
            print(f"score  {sim} {name} | " + " ".join(f"{m}={np.mean(v):.4f}" for m, v in per_metric.items()), flush=True)
    if args.features != "all":
        return                                                 # ablation arms are diagnostics, not candidates: they must not touch the locked-winner bookkeeping
    if not args.final:
        print("\n" + table(ds, "valid", better=True).to_string())
    report(ds) if args.final else select(ds)                   # valid sweep locks the winner; the test read reports against the lock


# Defines command-line options: dataset, the one-shot --final test read, encoder/sim/K/seed scope.
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Run the OGB study under the official protocol (valid sweep; --final = one test read).")
    p.add_argument('--dataset', required=True, choices=list(TASKS), help='ogbn_arxiv = node classification, ogbl_ddi = link prediction')
    p.add_argument('--final', action='store_true',
                   help='Score the OFFICIAL TEST split, once, AFTER selecting on validation. Refuses if no valid rows exist.')
    p.add_argument('--encoder', default='all', choices=['all', 'deepwalk'] + list(ENCODERS),
                   help='Which encoder(s) to run. "all" = the locked graphsage_edge + deepwalk pair; name any registered encoder to opt it in. Default all.')
    p.add_argument('--sim', default='all', choices=['all'] + cfg.VG_SIMS, help='Which virtual-graph variant(s). Default all.')
    p.add_argument('--features', default='all', choices=[f for f in cfg.D_FEATURES if f != 'none_mp'],
                   help='Ablation D input features (D6 none_mp is layers=0 and has no training -> not run here). Default all = the locked D0.')
    p.add_argument('--k', type=int, default=10, help='Top-K of the virtual graph. Default 10.')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44], help='Encoder seeds. Default 42 43 44.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
