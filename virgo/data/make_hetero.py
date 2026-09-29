'''Convert the heterophilous benchmarks (Platonov et al. 2023) to project edgelist/label/.nodes files: roman_empire, tolokers, questions - structural-only.'''
# Added 2026-07-27 to test the AUGMENT side of the rule. The six-dataset panel held no low-homophily graph, so the
# only "augment" verdict was a single cell (ogbl-ddi LP) and the decision boundary was fitted from one point.
# These two sit on the two axes the rule predicts from - adjusted homophily (node classification) and average degree
# (link prediction) - and the prediction is recorded in docs/paper_log.md BEFORE the run, so the outcome is held out.
# Purely structural: node_features (roman-empire 300-dim fastText, tolokers 10-dim worker profile, amazon-ratings 300-dim
# fastText of the product description, squirrel-filtered 2089-dim) are IGNORED by design.
# squirrel_filtered is the authors' de-duplicated Squirrel: the ORIGINAL WikipediaNetwork("squirrel") has repeated nodes
# that leak between train and test, so it is deliberately not used - only the filtered .npz from the authors' own repo.
# Platonov's own 10 train/val/test masks are also ignored: these run the CORE protocol (stratified 70% node
# classification, 70:30 link prediction) so their cells stay comparable with the four core datasets.
# Adjusted homophily is the metric this panel already reports, and it comes from this same paper.

import argparse
from pathlib import Path

from virgo.data.make_ogb import _write_edges, _write_nodes   # same writers as the OGB conversion; the import also installs the torch.load allowlist

RAW = "output/hetero_raw"                         # PyG download cache: derived, never hand-edited, kept out of input/

# our name -> PyG name, or None for the archives PyG's class does not list (fetched from the SAME repo it downloads
# from, so the data path is identical). roman_empire/tolokers/questions = discovery panel; minesweeper 2026-08-02,
# amazon_ratings 2026-08-05, squirrel_filtered 2026-08-05 = Module-3 held-out.
HETERO = {"roman_empire": "Roman-empire", "tolokers": "Tolokers", "questions": "Questions",
          "minesweeper": "Minesweeper", "amazon_ratings": "Amazon-ratings", "squirrel_filtered": None,
          # chameleon_filtered 2026-08-31: rebuilt to TEST the stage-1 second condition. Same de-duplication story as
          # squirrel_filtered - the original WikipediaNetwork copy leaks train->test - so only the authors' filtered .npz.
          "chameleon_filtered": None}

REPO = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data"   # the URL HeterophilousGraphDataset itself downloads from


# Reads one of the authors' .npz archives directly, for the graphs PyG's class does not expose (squirrel/chameleon filtered).
def _npz(name):
    '''One Platonov .npz -> PyG Data. Same file and format as the five HeterophilousGraphDataset names, different file list.'''
    import numpy as np
    import torch
    from torch_geometric.data import Data, download_url
    f = Path(download_url(f"{REPO}/{name}.npz", str(Path(RAW) / name / "raw")))
    d = np.load(f)
    return Data(x=torch.from_numpy(d["node_features"]), y=torch.from_numpy(d["node_labels"]),
                edge_index=torch.from_numpy(d["edges"].T), num_nodes=len(d["node_labels"]))


# Builds input/<name>.edgelist + .nodes + labels/<name>.labels from the authors' copy; drops data.x (structural-only).
def make_hetero(name):
    '''One heterophilous benchmark -> edgelist + .nodes + labels. node_features and the official masks ignored by design.'''
    if HETERO[name] is None:
        g = _npz(name)                                 # the .npz stores each undirected edge ONCE; _write_edges symmetrizes
    else:
        from torch_geometric.datasets import HeterophilousGraphDataset
        g = HeterophilousGraphDataset(root=RAW, name=HETERO[name])[0]
    n = g.num_nodes
    ne = _write_edges(f"input/{name}.edgelist", g.edge_index.numpy())   # dedup+sort: a no-op on PyG's already-symmetric copy, symmetrizes the raw .npz
    _write_nodes(f"input/{name}.edgelist", n)
    y = g.y.numpy().reshape(-1)
    Path("labels").mkdir(exist_ok=True)
    with open(f"labels/{name}.labels", "w") as f:
        for i in range(n):
            f.write(f"{i} {int(y[i])}\n")
    print(f"{name}: nodes={n} edges={ne} classes={len(set(y))} avg_degree={2 * ne / n:.2f} | "
          f"data.x ({g.x.shape[1]}-dim) IGNORED (structural-only) | official masks IGNORED (core protocol)")
    print(f"WROTE input/{name}.edgelist + .nodes + labels/{name}.labels")


# Defines command-line options: which heterophilous dataset to build (default all).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Convert the heterophilous benchmarks to project edgelist/label/.nodes files (structural-only).")
    p.add_argument('--dataset', default='all', choices=['all'] + list(HETERO), help='Which dataset to build. Default all.')
    return p.parse_args()


# Builds the requested dataset(s) from the command line.
def main(args):
    for name in HETERO:
        if args.dataset in ('all', name):
            make_hetero(name)


if __name__ == "__main__":
    main(parse_args())
