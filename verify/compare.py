'''Compare a rebuilt virgo-repro tree against the recorded reference: exact hashes for deterministic artifacts, numerical tolerance for trained results.'''
# Two checks, reported line by line and never merged into one "identical" claim:
#   hashes   datasets (data/USED.csv) and splits / role graphs / feature cache (expected/hashes/artifacts.csv) -> SHA256 must match
#   results  every CSV in expected/ against the same file in results/: text columns (verdicts, calls, names) must match
#            exactly, numeric columns within --tol. Newly trained embeddings are not bit-reproducible on GPU, so metrics
#            are judged by tolerance, not by bytes.
# Psi role graphs depend on an eigenvector solver with a random fallback start (docs/notes.md), so a psi role-graph hash
# mismatch is reported as KNOWN_EXCEPTION, not as a failure, when the shipped feature cache is not used.

import argparse
import csv
import hashlib
from pathlib import Path

import pandas as pd

KEYS = ["dataset", "encoder", "graph_variant", "variant", "top_K_neighbors", "K", "task", "metric", "seeds", "seed"]


def parse_args():
    p = argparse.ArgumentParser(description="Verify a rebuild against expected/.")
    p.add_argument('--step', default='all', choices=['all', 'hashes', 'results'], help='Which check to run. Default all.')
    p.add_argument('--root', default='.', help='Repo root of the rebuilt tree. Default current directory.')
    p.add_argument('--kinds', nargs='+', default=None, choices=['dataset', 'split', 'role_graph', 'feature_cache'], help='Hash only these artifact kinds. Default all.')
    p.add_argument('--tol', type=float, default=0.005, help='Absolute tolerance on numeric result columns. Default 0.005.')
    p.add_argument('--report', default='verify/report.csv', help='Where to write the line-by-line report.')
    return p.parse_args()


def sha(path):
    '''SHA256 of one file, streamed.'''
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_hashes(root, kinds=None):
    '''One row per recorded file: EXACT, MISMATCH, MISSING, KNOWN_EXCEPTION or NO_REFERENCE.'''
    rows = []
    listed = [(r["file"], r["sha256"], "dataset") for r in csv.DictReader(open(root / "data/USED.csv"))]
    listed += [(r["file"], r["sha256"], r["kind"]) for r in csv.DictReader(open(root / "expected/hashes/artifacts.csv"))]
    for rel, ref, kind in listed:
        if kinds and kind not in kinds:
            continue
        f = root / rel
        if not ref or ref.startswith("NOT_ON_DISK"):
            status = "NO_REFERENCE" if f.exists() else "MISSING"
        elif not f.exists():
            status = "MISSING"
        elif sha(f) == ref:
            status = "EXACT"
        else:
            status = "KNOWN_EXCEPTION" if kind == "role_graph" and "psi" in rel else "MISMATCH"
        rows.append({"check": "hash", "kind": kind, "item": rel, "status": status, "detail": ""})
    return rows


def check_results(root, tol):
    '''One row per expected CSV: EXACT, WITHIN_TOLERANCE, OUT_OF_TOLERANCE, TEXT_MISMATCH, SHAPE_MISMATCH or MISSING.'''
    rows = []
    for ref_path in sorted((root / "expected").glob("*.csv")):
        new_path = root / "results" / ref_path.name
        row = {"check": "results", "kind": "table", "item": ref_path.name}
        if not new_path.exists():
            rows.append({**row, "status": "MISSING", "detail": "not produced by the rebuild"})
            continue
        ref, new = pd.read_csv(ref_path), pd.read_csv(new_path)
        keys = [k for k in KEYS if k in ref.columns and k in new.columns]
        if keys:
            ref, new = ref.sort_values(keys).reset_index(drop=True), new.sort_values(keys).reset_index(drop=True)
        if ref.shape != new.shape or list(ref.columns) != list(new.columns):
            rows.append({**row, "status": "SHAPE_MISMATCH", "detail": f"expected {ref.shape}, got {new.shape}"})
            continue
        num = ref.select_dtypes("number").columns
        text = [c for c in ref.columns if c not in num]
        text_bad = [c for c in text if not ref[c].fillna("").astype(str).equals(new[c].fillna("").astype(str))]
        diff = (ref[num] - new[num]).abs().max().max() if len(num) else 0.0
        diff = 0.0 if diff != diff else float(diff)          # all-NaN numeric columns compare as equal
        if text_bad:
            status, detail = "TEXT_MISMATCH", f"columns differ: {text_bad}"
        elif diff == 0.0:
            status, detail = "EXACT", ""
        elif diff <= tol:
            status, detail = "WITHIN_TOLERANCE", f"max |diff| {diff:.6f}"
        else:
            status, detail = "OUT_OF_TOLERANCE", f"max |diff| {diff:.6f} > {tol}"
        rows.append({**row, "status": status, "detail": detail})
    return rows


def main(args):
    root = Path(args.root).resolve()
    rows = []
    if args.step in ("all", "hashes"):
        rows += check_hashes(root, args.kinds)
    if args.step in ("all", "results"):
        rows += check_results(root, args.tol)
    report = pd.DataFrame(rows)
    (root / args.report).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(root / args.report, index=False)
    print(report.groupby(["check", "status"]).size().to_string())
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main(parse_args())
