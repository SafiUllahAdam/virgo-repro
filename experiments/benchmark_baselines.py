

"""Cross-model, cross-dataset benchmark: I2V vs deepwalk / node2vec / struc2vec on node-class + link-pred.

Reuses the existing per-seed runners and eval modules UNCHANGED; this file only adds the dataset x model loop
and the two final comparison tables (datasets x methods). Embeddings/splits are cached by filename, so reruns
train only what is missing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

import pandas as pd
from virgo.config import BENCH_DATASETS, BENCH_MODELS, RESULTS_DIR, REPRO
from virgo.data.make_labels import prepare_dataset
from virgo.eval.runner import run_nodeclass_repeated, run_linkpred_repeated


# Runs every (dataset, model) for both tasks over the seeds; node-class is skipped where labels can't be aligned.
def run_benchmark(datasets=None, models=None, seeds=(42, 43, 44)):
    """Loop datasets x models -> per-seed node-class + link-pred rows (reusing the existing runners)."""
    datasets = datasets or BENCH_DATASETS
    models = models or BENCH_MODELS
    rows = []
    for ds in datasets:
        try:
            info = prepare_dataset(ds)                          # resolves to the safe (aligned) version, builds if missing
        except Exception as e:                                  # labels couldn't be built (e.g. politics) -> still do link-pred
            raw = _ROOT / "input" / f"{ds}.edgelist"
            if not raw.exists():
                print(f"!! skip dataset '{ds}': {e}")
                continue
            print(f"!! {ds}: no aligned labels ({e}) -> link prediction only (node classification skipped).")
            info = {"base": ds, "version": "orig", "safe": ds,
                    "edge_path": str(raw), "label_path": str(_ROOT / "labels" / f"{ds}.labels")}
        has_labels = Path(info["label_path"]).exists()
        for model in models:
            print(f"\n=== {model} on {info['safe']} ===")
            if has_labels:
                try:
                    rows += run_nodeclass_repeated(info, seeds=seeds, model=model)
                except Exception as e:                          # one model failing must not abort the sweep or lose other rows
                    print(f"  !! {model} node-class FAILED on {info['safe']}: {e}")
            else:
                print(f"  (no labels for {info['safe']} -> node classification skipped; link prediction still runs)")
            try:
                rows += run_linkpred_repeated(info, seeds=seeds, model=model)
            except Exception as e:
                print(f"  !! {model} link-pred FAILED on {info['safe']}: {e}")
    return rows


# Pivots per-seed rows into a datasets x methods table of "mean ± std" for one (task, metric).
def benchmark_table(rows, task, metric):
    """Rows = datasets, Columns = methods, values = mean ± std across seeds."""
    df = pd.DataFrame(rows)
    df = df[df["task"] == task].dropna(subset=[metric])
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["dataset", "model"])[metric].agg(["mean", "std"]).reset_index()
    g["cell"] = g.apply(lambda r: f"{r['mean']:.4f} ± {(0.0 if pd.isna(r['std']) else r['std']):.4f}", axis=1)
    table = g.pivot(index="dataset", columns="model", values="cell")
    cols = [m for m in BENCH_MODELS if m in table.columns]      # keep the method column order stable
    return table[cols]


# Builds both final tables, saves them + the raw per-seed rows under results/benchmark/.
def save_benchmark(rows, out_dir=None):
    """Write per-seed rows + Table 1 (node-class weighted F1) + Table 2 (link-pred AUC, one file per scorer). Returns the two tables."""
    out_dir = Path(out_dir or RESULTS_DIR / "notebook1_reproduce_i2v" / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "benchmark_per_seed.csv", index=False)
    head = REPRO["linkpred_score"]                                   # headline scorer, named in the filename so the two tables can never be confused
    other = "logreg" if head == "cosine" else "cosine"
    nc = benchmark_table(rows, "nodeclass", "weighted_f1")
    lp = benchmark_table(rows, "linkpred", "auc")                    # headline AUC = whichever scorer REPRO selects
    lp_other = benchmark_table(rows, "linkpred", f"auc_{other}")     # the alternative scorer: never a duplicate of the headline
    nc.to_csv(out_dir / "table1_nodeclass_weighted_f1.csv")
    lp.to_csv(out_dir / f"table2_linkpred_auc_{head}.csv")
    lp_other.to_csv(out_dir / f"table2_linkpred_auc_{other}.csv")
    return nc, lp


if __name__ == "__main__":
    rows = run_benchmark()
    nc, lp = save_benchmark(rows)
    print("\n=== Table 1: Node Classification (weighted F1, mean ± std) ===")
    print(nc.to_string())
    print(f"\n=== Table 2: Link Prediction (AUC via {REPRO['linkpred_score']}, mean ± std) ===")
    print(lp.to_string())
