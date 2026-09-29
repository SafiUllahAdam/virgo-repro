"""Orchestrate reproduction tasks by calling the Phase-1 entry points - nothing is moved."""

import subprocess
import sys
from pathlib import Path

from virgo.config import (NB1_DIR, LP_SPLITS_ORIG, PROJECT_ROOT, REPRO, I2V_PARAMS,
                          I2V_BASELINE_POLICY, dataset)
from virgo.utils import set_seed


# Runs Identity2Vec (train.py) as a subprocess to learn an embedding - this is the slow step.
def embed(input_path, output_path, params=None, cached=True, seed=None):
    """Train an I2V embedding on a given edgelist (cached fast path + fixed seed by default)."""
    params = params or I2V_PARAMS
    seed = REPRO["seed"] if seed is None else seed
    cmd = [sys.executable, str(PROJECT_ROOT / "experiments" / "train.py"),
           "--input", str(input_path), "--output", str(output_path),
           "--dimensions", str(params["dimensions"]),
           "--walk-length", str(params["walk_length"]),
           "--num-walks", str(params["num_walks"]),
           "--window-size", str(params["window_size"]),
           "--epochs", str(params["epochs"]), "--sg", str(params["sg"]),
           "--seed", str(seed)]
    if cached:
        cmd.append("--cached")
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return Path(output_path)


# Link prediction: split edges 70/30, optionally retrain on the train graph, report held-out AUC.
def run_linkpred(name, emb=None, retrain=False, params=None, seed=None):
    """Return link-prediction metrics + the settings used."""
    seed = REPRO["seed"] if seed is None else seed
    set_seed(seed)
    from virgo.data.prepare_linkpred import prepare   # lazy: keeps --list working without sklearn
    from virgo.eval.linkpred import evaluate as linkpred_eval

    split_dir = LP_SPLITS_ORIG / name / f"seed_{seed}"          # splits/link_prediction/original_graph/<ds>/seed_<s>/
    counts = prepare(dataset(name)["edgelist"], split_dir, REPRO["linkpred_test_frac"], seed,
                     negatives=I2V_BASELINE_POLICY["lp_negatives"])   # Phase-1 contract: I2V's own uniform negatives, not GRAPH_POLICY's
    if retrain:
        emb = NB1_DIR / name / "link_prediction" / f"i2v_s{seed}.emb"
        embed(split_dir / "train.edgelist", emb, params)
    elif emb is None:
        emb = NB1_DIR / name / "node_classification" / f"i2v_s{seed}.emb"
        print(f"  ! no --retrain: using full-graph {Path(emb).name} (LEAKAGE - plumbing check, not a paper number)")

    auc = linkpred_eval(emb, split_dir, REPRO["linkpred_op"], seed, REPRO["linkpred_score"])
    metrics = {"auc": auc}
    settings = {"seed": seed, "op": REPRO["linkpred_op"], "test_frac": REPRO["linkpred_test_frac"],
                "retrain": retrain, "emb": Path(emb).name, **counts}
    return metrics, settings


# Node classification: logistic regression on embeddings vs labels, report weighted F1.
def run_nodeclass(name, emb=None, seed=None):
    """Return node-classification metrics + the settings used."""
    seed = REPRO["seed"] if seed is None else seed
    set_seed(seed)
    from virgo.eval.nodeclass import evaluate as nodeclass_eval

    labels = dataset(name)["labels"]
    if labels is None or not Path(labels).exists():
        raise FileNotFoundError(
            f"No labels for '{name}' (expected {labels}). See labels/README.md - node classification is blocked.")
    emb = emb or NB1_DIR / name / "node_classification" / f"i2v_s{seed}.emb"
    f1s, n_nodes, n_classes = nodeclass_eval(emb, labels, REPRO["nodeclass_train_frac"], seed)
    metrics = {"micro_f1": f1s["micro"], "macro_f1": f1s["macro"], "weighted_f1": f1s["weighted"],
               "n_nodes": n_nodes, "n_classes": n_classes}
    settings = {"seed": seed, "train_frac": REPRO["nodeclass_train_frac"], "emb": Path(emb).name}
    return metrics, settings


# Repeat node classification over seeds for one model: a fresh full-graph embedding per seed -> per-seed metric rows.
def run_nodeclass_repeated(info, seeds=(42, 43, 44), params=None, model="identity2vec"):
    """Train one embedding per seed with `model`, score node classification each time. Returns per-seed rows."""
    params = params or I2V_PARAMS
    from virgo.eval.nodeclass import evaluate as nodeclass_eval
    from virgo.encoders.walk import get_model
    mdl = get_model(model)
    out_dir = NB1_DIR / info["safe"] / "node_classification"     # output/notebook1_reproduce_i2v/<dataset>/node_classification/
    out_dir.mkdir(parents=True, exist_ok=True)
    short = "i2v" if model == "identity2vec" else model          # folder names the dataset+task; file names the model+seed
    rows = []
    for s in seeds:
        emb = out_dir / f"{short}_s{s}.emb"
        if emb.exists():
            print(f"  [{model} nc s{s}] reuse existing {emb.name}")   # delete file to force rebuild
        else:
            mdl.train(info["edge_path"], emb, s, params)
        f1s, n, c = nodeclass_eval(emb, info["label_path"], REPRO["nodeclass_train_frac"], s)
        rows.append({"model": model, "dataset": info["base"], "version": info["version"], "task": "nodeclass", "seed": s,
                     "micro_f1": f1s["micro"], "macro_f1": f1s["macro"], "weighted_f1": f1s["weighted"]})
        print(f"  [{model} nc s{s}] micro={f1s['micro']:.4f} macro={f1s['macro']:.4f} weighted={f1s['weighted']:.4f}")
    return rows


# Repeat link prediction over seeds for one model: per-seed split + train-only embedding (no leakage) -> per-seed AUC rows.
def run_linkpred_repeated(info, seeds=(42, 43, 44), params=None, model="identity2vec"):
    """Per seed: split edges, train `model` on the 70% graph only, score AUC. Returns per-seed rows."""
    params = params or I2V_PARAMS
    from virgo.data.prepare_linkpred import prepare
    from virgo.eval.linkpred import evaluate as linkpred_eval
    from virgo.encoders.walk import get_model
    mdl = get_model(model)
    out_dir = NB1_DIR / info["safe"] / "link_prediction"        # output/notebook1_reproduce_i2v/<dataset>/link_prediction/
    out_dir.mkdir(parents=True, exist_ok=True)
    short = "i2v" if model == "identity2vec" else model         # split is model-independent; only the emb carries the model
    rows = []
    for s in seeds:
        split_dir = LP_SPLITS_ORIG / info["safe"] / f"seed_{s}"  # one shared split per (dataset, seed) -> fair across models
        prepare(info["edge_path"], split_dir, REPRO["linkpred_test_frac"], s,      # deterministic: recreates same split cheaply
                negatives=I2V_BASELINE_POLICY["lp_negatives"])                     # Phase-1 contract: I2V's own uniform negatives
        train_edges = split_dir / "train.edgelist"
        emb = out_dir / f"{short}_s{s}.emb"
        if emb.exists() and emb.stat().st_mtime >= train_edges.stat().st_mtime:
            print(f"  [{model} lp s{s}] reuse existing {emb.name}")   # delete file to force rebuild
        else:
            if emb.exists():          # the split changed after this embedding was trained; reusing it would score an embedding
                print(f"  [{model} lp s{s}] {emb.name} is older than the split -> retraining")   # against a test set it never saw
            mdl.train(train_edges, emb, s, params)
        auc_logreg = linkpred_eval(emb, split_dir, REPRO["linkpred_op"], s, "logreg")   # main: Hadamard + logreg
        auc_cosine = linkpred_eval(emb, split_dir, REPRO["linkpred_op"], s, "cosine")   # second column: unsupervised similarity
        auc = auc_logreg if REPRO["linkpred_score"] == "logreg" else auc_cosine            # headline AUC = configured scorer
        rows.append({"model": model, "dataset": info["base"], "version": info["version"], "task": "linkpred", "seed": s,
                     "auc": auc, "auc_logreg": auc_logreg, "auc_cosine": auc_cosine})
        print(f"  [{model} lp s{s}] AUC={auc:.4f} (logreg={auc_logreg:.4f} cosine={auc_cosine:.4f})")
    return rows


# Aggregate per-seed rows into a tidy summary: mean, sample-std (ddof=1), and range (delta = max-min) per metric.
def summarize_seed_results(node_rows, lp_rows):
    """Return (per_seed_df, summary_df) from the repeated-run rows."""
    import pandas as pd
    per_seed = pd.DataFrame(node_rows + lp_rows)
    recs = []
    for (ds, ver, task), g in per_seed.groupby(["dataset", "version", "task"], sort=False):
        metrics = ["micro_f1", "macro_f1", "weighted_f1"] if task == "nodeclass" else ["auc"]
        for m in metrics:
            v = g[m].dropna().to_numpy()
            recs.append({"dataset": ds, "version": ver, "task": task, "metric": m,
                         "mean": v.mean(), "std": v.std(ddof=1), "delta": v.max() - v.min(), "n": len(v)})
    return per_seed, pd.DataFrame(recs)


TASKS = {"linkpred": run_linkpred, "nodeclass": run_nodeclass}
