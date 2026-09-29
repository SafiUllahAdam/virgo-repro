"""CLI - run an I2V reproduction task and save the result."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo.config import DATASETS, RESULTS_DIR, REPRO
from virgo.eval.results_io import save_result
from virgo.eval.runner import TASKS


# Defines the command-line options: which task, which dataset, optional embedding / config / flags.
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=list(TASKS), help="Reproduction task to run")
    p.add_argument("--dataset", help="Dataset name (see --list)")
    p.add_argument("--emb", default=None, help="Use an existing embedding instead of retraining")
    p.add_argument("--retrain", action="store_true", help="Link pred: retrain I2V on the 70%% train graph (no leakage)")
    p.add_argument("--config", default=None, help="JSON run-config; its task/dataset override the flags")
    p.add_argument("--list", action="store_true", help="List datasets and exit")
    p.add_argument("--no-save", action="store_true", help="Do not write a results CSV")
    p.add_argument("--seed", type=int, default=REPRO["seed"])
    return p.parse_args()


# Lists datasets and whether their label files are present yet.
def list_datasets():
    """Print the dataset registry."""
    print("Datasets:")
    for name, d in DATASETS.items():
        has = "labels" if d["labels"] and Path(d["labels"]).exists() else "no-labels"
        print(f"  {name:<10} {d['edgelist'].name:<22} {has}")


# Runs one task end to end: dispatch to the runner, print the metrics, save a results CSV.
def main():
    args = parse_args()
    if args.list:
        list_datasets()
        return

    cfg = json.loads(Path(args.config).read_text()) if args.config else {}
    task = cfg.get("task", args.task)
    dataset = cfg.get("dataset", args.dataset)
    if not task or not dataset:
        sys.exit("Need --task and --dataset (or --config). Use --list to see datasets.")

    print(f"\ntask={task} · dataset={dataset} · seed={args.seed}")
    kwargs = {"emb": args.emb, "seed": args.seed}
    if task == "linkpred":
        kwargs["retrain"] = args.retrain
    metrics, settings = TASKS[task](dataset, **kwargs)

    print("  result:", "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in metrics.items()))
    if not args.no_save:
        save_result(RESULTS_DIR / "notebook1_reproduce_i2v", dataset, task, metrics, settings)


if __name__ == "__main__":
    main()
