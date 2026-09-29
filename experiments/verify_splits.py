'''Prove the stored link-prediction splits are regenerable: rebuild each one into a temp dir and byte-compare.'''
# splits/ is 1.6G and is NOT tracked in git. That is only defensible if a clone can rebuild the exact same splits,
# so this script is the evidence for that claim rather than an assertion in a commit message.
# What the determinism actually rests on: prepare_linkpred.split_edges() seeds random.Random(seed) and
# sample_non_edges() seeds random.Random(seed + 1), and both sort their output - but the spanning forest comes from
# nx.minimum_spanning_tree() on an UNWEIGHTED graph, where Kruskal breaks ties by edge insertion order, i.e. the line
# order of input/<ds>.edgelist. So the edgelist file, not just the seed, is part of the split's identity. That is why
# input/ stays tracked while splits/ does not, and why a re-download that renumbers nodes would invalidate this check.

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo.data import prepare_linkpred

FILES = ["train.edgelist", "train_neg.txt", "test_pos.txt", "test_neg.txt"]


# Defines command-line options: which datasets to check, and whether to stop at the first mismatch.
def parse_args():
    parser = argparse.ArgumentParser(description="Byte-compare regenerated link-prediction splits against the stored ones.")
    parser.add_argument('--datasets', nargs='*', default=None, help='Datasets to check. Default: every one with a stored split.')
    parser.add_argument('--limit', type=int, default=None, help='Check only the first N seed folders (quick smoke test).')
    return parser.parse_args()


# Regenerates one seed folder into a temp dir and returns the split files whose bytes differ.
def compare(ds, seed, stored):
    '''Rebuild the (dataset, seed) split from the edgelist and return the names of the files that differ.'''
    digest = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        prepare_linkpred.prepare(cfg.dataset(ds)["edgelist"], tmp, test_frac=cfg.REPRO["linkpred_test_frac"], seed=seed)
        return [f for f in FILES if digest(stored / f) != digest(Path(tmp) / f)]


# Walks every stored seed folder, reports per-dataset mismatches, and exits non-zero if any split failed to reproduce.
def main(args):
    dirs = sorted(d for d in cfg.LP_SPLITS_VG.glob("*/seed_*") if (d / "test_pos.txt").exists())
    if args.datasets:
        dirs = [d for d in dirs if d.parent.name in args.datasets]
    dirs = dirs[:args.limit] if args.limit else dirs

    ok = bad = skip = 0
    for d in dirs:
        ds, seed = d.parent.name, int(d.name.split("_")[1])
        if ds not in cfg.DATASETS or not Path(cfg.dataset(ds)["edgelist"]).exists():
            print(f"SKIP {ds:26s} s{seed}  edgelist unavailable -> rebuild it with virgo/data/make_*.py", flush=True)
            skip += 1
            continue
        diff = compare(ds, seed, d)
        bad, ok = (bad + 1, ok) if diff else (bad, ok + 1)
        if diff:
            print(f"FAIL {ds:26s} s{seed}  differs: {diff}", flush=True)

    print(f"\nidentical {ok} | differ {bad} | skipped {skip}  (of {len(dirs)} seed folders)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
