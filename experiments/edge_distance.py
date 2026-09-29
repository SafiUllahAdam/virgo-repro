'''How far apart, in the ORIGINAL graph, are the endpoints of the edges a role graph adds?'''
# The GAT objection: if role edges mostly join 1-2 hop neighbours, attention over the original graph could already reach
# them, so the rewiring adds nothing an attention head cannot. If they join distant nodes, it adds reach that no
# message-passing depth-2 model has.
#   the control  a small-world graph makes almost EVERY non-adjacent pair 4-6 hops apart, so a raw "mean 5.2 hops" proves
#                nothing. `random_*` samples node pairs uniformly and measures the same way; `ratio` = role / random.
#                ratio ~ 1 means the role edges are no more long-range than chance, only non-adjacent.
#   helpfulness  per-edge attribution is not available, so graphs are labelled by whether augmentation WON there
#                (results/scoreboard.csv, GraphSAGE LP) and the distributions are compared across that split.

import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo.virtual_graph import VirtualGraph

CUTOFF = 8                                                        # BFS depth cap; beyond it the pair is reported as "far", not measured


# One BFS per unique source, capped, so the cost is O(sources x E) instead of all-pairs.
def hops(G, pairs):
    '''Original-graph shortest-path length for each (u, v); np.inf when farther than CUTOFF or disconnected.'''
    out = []
    by_src = {}
    for u, v in pairs:
        by_src.setdefault(u, []).append(v)
    for u, vs in by_src.items():
        d = nx.single_source_shortest_path_length(G, u, cutoff=CUTOFF)
        out += [float(d.get(v, np.inf)) for v in vs]
    return np.array(out)


def summarize(name, d):
    '''The distribution in the four numbers the question actually turns on.'''
    fin = d[np.isfinite(d)]
    return {f"{name}_median": float(np.median(fin)) if len(fin) else float("nan"),
            f"{name}_mean": round(float(fin.mean()), 3) if len(fin) else float("nan"),
            f"{name}_pct_1_2hop": round(float((d <= 2).mean() * 100), 1),
            f"{name}_pct_5plus": round(float(((d >= 5) | ~np.isfinite(d)).mean() * 100), 1)}


def measure(ds, sim, k, n_sample, rng):
    '''One row: role-edge hop distribution against a random-pair null on the same graph.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    V = VirtualGraph(G, seed=cfg.REPRO["seed"]).build(sim, k)
    edges = [e for e in V.edges if not G.has_edge(*e)]             # an edge the role graph ADDS; overlaps with the original add no reach
    kept = V.number_of_edges() - len(edges)
    if not edges:
        return None
    idx = rng.choice(len(edges), min(n_sample, len(edges)), replace=False)
    role = hops(G, [edges[i] for i in idx])
    nodes = list(G.nodes)
    rand = [(nodes[i], nodes[j]) for i, j in rng.integers(0, len(nodes), (len(idx), 2)) if i != j]
    null = hops(G, rand)
    row = {"dataset": ds, "sim": sim, "nodes": G.number_of_nodes(), "vg_edges": V.number_of_edges(),
           "added": len(edges), "overlap_with_original": kept, "sampled": len(idx)}
    row.update(summarize("role", role))
    row.update(summarize("random", null))
    row["ratio_median"] = round(row["role_median"] / row["random_median"], 3) if row["random_median"] else float("nan")
    return row


def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.REPRO["seed"])
    rows = [r for ds in args.datasets for sim in args.sims
            if (r := measure(ds, sim, args.k, args.sample, rng)) is not None]
    df = pd.DataFrame(rows)
    df.to_csv(cfg.RESULTS_DIR / "edge_distance.csv", index=False)
    cols = ["dataset", "sim", "added", "overlap_with_original", "role_median", "role_mean", "role_pct_1_2hop",
            "role_pct_5plus", "random_median", "random_mean", "ratio_median"]
    print("\nORIGINAL-GRAPH HOP DISTANCE OF THE EDGES THE ROLE GRAPH ADDS (random node pairs = the small-world null)")
    print(df[cols].to_string(index=False))
    print(f"\nrole edges at 1-2 hops: mean {df['role_pct_1_2hop'].mean():.1f}%  |  random pairs at 1-2 hops: mean {df['random_pct_1_2hop'].mean():.1f}%")
    print(f"median hop ratio role/random: mean {df['ratio_median'].mean():.3f} (1.0 = no more long-range than chance)")
    print(f"\n{len(df):3d} rows -> results/edge_distance.csv")


# Defines command-line options (mirrors psi_clamp.py / psi_omega.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Measure the original-graph distance spanned by added role edges.")
    p.add_argument('--datasets', nargs='+', default=["cora", "twitch_es", "squirrel_filtered", "actor", "minesweeper", "lastfm_asia"])
    p.add_argument('--sims', nargs='+', default=["psi", "degree", "centrality"], help='Role-graph variants to measure.')
    p.add_argument('--k', type=int, default=cfg.VG_K[1], help='Top-K of the role graph. Default 10 (the locked setting).')
    p.add_argument('--sample', type=int, default=3000, help='Added edges sampled per (dataset, variant). Default 3000.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
