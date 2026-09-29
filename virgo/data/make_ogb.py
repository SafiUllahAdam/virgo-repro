'''Convert OGB datasets to project edgelist/label/.nodes/split files: ogbn-arxiv (node class) + ogbl-ddi (link pred), structural-only.'''
# Purely structural: OGB node features (data.x, 128-dim text) are IGNORED by design -> this project studies STRUCTURE, not attributes.
# Official OGB splits are saved verbatim (arxiv time split, ddi protein-target split) -> the datasets keep their INTENDED task.
# .nodes sidecar carries the FULL node set so isolated nodes (no training edge) survive load_graph() and still get an embedding.

import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

# PyTorch >= 2.6 flipped torch.load to weights_only=True; OGB's caches are pickled PyG objects (processed graph) and
# pickled numpy arrays (split files), which the safe unpickler rejects. Allowlist exactly those (OGB's official host).
torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, GlobalStorage, np.core.multiarray._reconstruct, np.ndarray, np.dtype]
    + [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")])

RAW = "output/ogb_raw"                         # OGB download cache: derived, never hand-edited, kept out of input/


# Dedups undirected int edges from a (2,E) or (E,2) array and writes "u v" lines; returns the edge count.
def _write_edges(path, edge_array):
    '''Write an undirected, deduplicated "u v" edgelist; drop self-loops.'''
    ei = np.asarray(edge_array)
    pairs = zip(ei[0], ei[1]) if ei.shape[0] == 2 else zip(ei[:, 0], ei[:, 1])   # (2,E) columns vs (E,2) rows
    edges = sorted({tuple(sorted((int(u), int(v)))) for u, v in pairs if int(u) != int(v)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")
    return len(edges)


# Writes a .nodes sidecar (every id 0..n-1) next to the edgelist so load_graph() restores isolated nodes.
def _write_nodes(edgelist_path, n):
    '''Write the full node set, one id per line, beside the edgelist.'''
    with open(Path(edgelist_path).with_suffix(".nodes"), "w") as f:
        for i in range(n):
            f.write(f"{i}\n")


# Builds input/ogbn_arxiv.edgelist + labels + .nodes + official split; drops the 128-dim text features (structural-only).
def make_ogbn_arxiv():
    '''ogbn-arxiv -> full transductive edgelist + labels + .nodes + official time split. data.x ignored by design.'''
    from ogb.nodeproppred import PygNodePropPredDataset
    d = PygNodePropPredDataset(name="ogbn-arxiv", root=RAW)
    g = d[0]
    n = g.num_nodes
    ne = _write_edges("input/ogbn_arxiv.edgelist", g.edge_index.numpy())
    _write_nodes("input/ogbn_arxiv.edgelist", n)
    y = g.y.numpy().reshape(-1)
    Path("labels").mkdir(exist_ok=True)
    with open("labels/ogbn_arxiv.labels", "w") as f:
        for i in range(n):
            f.write(f"{i} {int(y[i])}\n")
    idx = d.get_idx_split()
    Path("splits/ogb").mkdir(parents=True, exist_ok=True)
    np.savez("splits/ogb/ogbn_arxiv_idx.npz",
             train=idx["train"].numpy(), valid=idx["valid"].numpy(), test=idx["test"].numpy())
    print(f"ogbn-arxiv: nodes={n} edges={ne} classes={len(set(y))} "
          f"split(train/valid/test)={len(idx['train'])}/{len(idx['valid'])}/{len(idx['test'])} | data.x IGNORED (structural-only)")
    print("WROTE input/ogbn_arxiv.edgelist + .nodes + labels/ogbn_arxiv.labels + splits/ogb/ogbn_arxiv_idx.npz")


# Builds input/ogbl_ddi_train.edgelist (TRAINING edges only) + .nodes + official valid/test pos/neg pairs.
def make_ogbl_ddi():
    '''ogbl-ddi -> graph from TRAINING edges only (no valid/test leakage) + .nodes + official pos/neg pairs for Hits@20.'''
    from ogb.linkproppred import PygLinkPropPredDataset
    d = PygLinkPropPredDataset(name="ogbl-ddi", root=RAW)
    n = d[0].num_nodes
    s = d.get_edge_split()
    ne = _write_edges("input/ogbl_ddi_train.edgelist", s["train"]["edge"].numpy())   # train edges ONLY -> structure carries no test leakage
    _write_nodes("input/ogbl_ddi_train.edgelist", n)
    Path("splits/ogb").mkdir(parents=True, exist_ok=True)
    np.savez("splits/ogb/ogbl_ddi_pairs.npz",
             valid_pos=s["valid"]["edge"].numpy(), valid_neg=s["valid"]["edge_neg"].numpy(),
             test_pos=s["test"]["edge"].numpy(),   test_neg=s["test"]["edge_neg"].numpy())
    print(f"ogbl-ddi: nodes={n} train_edges={ne} "
          f"valid(pos/neg)={len(s['valid']['edge'])}/{len(s['valid']['edge_neg'])} "
          f"test(pos/neg)={len(s['test']['edge'])}/{len(s['test']['edge_neg'])} | features: NONE (structure-only benchmark)")
    print("WROTE input/ogbl_ddi_train.edgelist + .nodes + splits/ogb/ogbl_ddi_pairs.npz")


# Builds a requested OGB dataset if its files are missing; returns the path info the pipeline needs.
def ensure_ogb(name):
    '''Build ogbn_arxiv | ogbl_ddi files if absent; return an info dict (edgelist + labels/pairs/split paths).'''
    name = name.lower()
    if name == "ogbn_arxiv":
        if not Path("input/ogbn_arxiv.edgelist").exists():
            make_ogbn_arxiv()
        return {"edge_path": "input/ogbn_arxiv.edgelist", "label_path": "labels/ogbn_arxiv.labels",
                "split": "splits/ogb/ogbn_arxiv_idx.npz", "eval": "ogb", "task": "nodeclass"}
    if name == "ogbl_ddi":
        if not Path("input/ogbl_ddi_train.edgelist").exists():
            make_ogbl_ddi()
        return {"edge_path": "input/ogbl_ddi_train.edgelist", "pairs": "splits/ogb/ogbl_ddi_pairs.npz",
                "eval": "ogb", "task": "linkpred"}
    raise ValueError(f"Unknown OGB dataset '{name}'. Use: ogbn_arxiv, ogbl_ddi.")


# Defines command-line options: which OGB dataset to build (default both).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Convert OGB datasets to project edgelist/label/.nodes/split files (structural-only).")
    p.add_argument('--dataset', default='both', choices=['both', 'ogbn_arxiv', 'ogbl_ddi'], help='Which OGB dataset to build. Default both.')
    return p.parse_args()


# Builds the requested OGB dataset(s) from the command line.
def main(args):
    if args.dataset in ('both', 'ogbn_arxiv'):
        make_ogbn_arxiv()
    if args.dataset in ('both', 'ogbl_ddi'):
        make_ogbl_ddi()


if __name__ == "__main__":
    main(parse_args())
