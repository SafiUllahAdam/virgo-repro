"""Save a benchmark run to results/ as a numbered CSV with a JSON metadata header."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from virgo.utils import next_run_id


# Writes one run: results/NNN.{dataset}.{task}.csv with a #META header line then metric rows.
def save_result(results_dir, dataset, task, metrics, settings):
    """Persist metrics + run settings; returns the file path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rid = next_run_id(results_dir)
    stamp = datetime.now().strftime("%d.%m")
    path = results_dir / f"{rid:03d}.{stamp}.{dataset}.{task}.csv"

    meta = {"run_id": rid, "dataset": dataset, "task": task,
            "timestamp": datetime.now().isoformat(), **settings}
    with open(path, "w", newline="") as f:
        f.write(f"#META:{json.dumps(meta)}\n")
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])

    print(f"✓ saved {path.relative_to(results_dir.parent)}")
    return path


# Legacy task -> metric map: pre-2026-07-22 rows named their metric inside the task string; the metric column is now the authority.
LEGACY_METRIC = {"link prediction (AUC)": "auc", "node classification (weighted F1)": "weighted_f1"}


# Upserts one scoreboard row (key: dataset x encoder x graph_variant x top_K_neighbors x task x metric); accumulates every variant ever scored.
def record_score(dataset, encoder, graph_variant, top_k, task, seeds, scores,
                 metric=None, csv_path="results/scoreboard.csv"):
    """Add/refresh one (dataset, encoder, graph_variant, top_K_neighbors, task, metric) row in the accumulating scoreboard CSV."""
    metric = metric or LEGACY_METRIC.get(task)   # OGB rows MUST pass it (e.g. "valid_hits@20", "test_acc"): mean alone is ambiguous
    assert metric, f"metric is required for task '{task}': acc/F1/AUC/hits@20 numbers are not comparable unnamed"
    # sample std (ddof=1) so this matches runner.summarize_seed_results and pandas .agg("std"): every reported std is comparable.
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    row = {"dataset": dataset, "encoder": encoder, "graph_variant": graph_variant, "top_K_neighbors": top_k,
           "task": task, "metric": metric, "seeds": "|".join(str(s) for s in seeds),
           "mean": round(float(np.mean(scores)), 4), "std": round(std, 4)}
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # size check, not just exists(): an emptied 0-byte file passes exists() and makes read_csv raise EmptyDataError.
    board = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame(columns=list(row))
    if "metric" not in board.columns:            # pre-2026-07-22 file (e.g. restored from git): backfill from the task names
        board["metric"] = board["task"].map(LEGACY_METRIC)
    key = ((board["dataset"] == dataset) & (board["encoder"] == encoder) & (board["graph_variant"] == graph_variant)
           & (board["top_K_neighbors"] == top_k) & (board["task"] == task) & (board["metric"] == metric))
    board = pd.concat([board[~key], pd.DataFrame([row])], ignore_index=True)
    (board[list(row)].sort_values(["dataset", "task", "metric", "encoder", "graph_variant", "top_K_neighbors"])
          .reset_index(drop=True).to_csv(path, index=False))
    return path
